"""
Training loop for HiT-HAR.

Supports:
  - AMP mixed precision
  - WandB logging
  - Early stopping
"""

import os
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self._save(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(model)
            self.counter = 0

    def _save(self, model):
        if isinstance(model, nn.DataParallel):
            self.best_model_state = {k: v.cpu() for k, v in model.module.state_dict().items()}
        else:
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}


class HiTHARTrainer:
    """Training loop for HiT-HAR model."""

    def __init__(self, model, criterion, config, args):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.args = args

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        print(f"Using device: {self.device}")

        # Move model to device
        self.model = self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)

        # Move criterion buffers to device
        self.criterion = self.criterion.to(self.device)

        # Optimizer — include criterion parameters (e.g., learnable beta) if any
        lr = config['training']['lr']
        wd = config['training']['weight_decay']
        all_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        criterion_params = list(filter(lambda p: p.requires_grad, self.criterion.parameters()))
        if criterion_params:
            # Use higher LR for beta parameters (they are scalar/few params)
            beta_lr = config['training'].get('beta_lr', lr * 10)
            self.optimizer = torch.optim.AdamW([
                {'params': all_params, 'lr': lr, 'weight_decay': wd},
                {'params': criterion_params, 'lr': beta_lr, 'weight_decay': 0.0},
            ])
        else:
            self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=wd)

        # Scheduler: warmup + cosine
        warmup_epochs = config['training'].get('warmup_epochs', 5)
        total_epochs = config['training']['epochs']
        min_factor = 0.2

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # AMP scaler
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))

        # Grad clip
        self.grad_clip = config['training'].get('grad_clip', 1.0)

        # EMA model (Exponential Moving Average)
        self.ema_decay = config['training'].get('ema_decay', 0.0)
        self.ema_model = None
        if self.ema_decay > 0:
            from copy import deepcopy
            self.ema_model = deepcopy(self.model)
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

        # Early stopping
        patience = config['training'].get('patience', 15)
        self.early_stopper = EarlyStopping(patience=patience)

        # WandB
        self.wandb_run = None

    def init_wandb(self, train_ds, val_ds):
        """Initialize WandB logging with group/tags."""
        if self.args.no_wandb or not HAS_WANDB:
            return
        tags = [f"phase{self.args.phase}", f"seed{self.args.seed}"]
        if hasattr(self.args, 'wandb_tags') and self.args.wandb_tags:
            tags.extend(self.args.wandb_tags.split(','))
        group = getattr(self.args, 'wandb_group', None)
        project = os.environ.get("WANDB_PROJECT", "hit-har")
        entity = os.environ.get("WANDB_ENTITY")
        self.wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=self.args.run_name or f"hit-har-phase{self.args.phase}",
            group=group, tags=tags,
            config={**self.config, "phase": self.args.phase, "seed": self.args.seed,
                    "train_samples": len(train_ds), "val_samples": len(val_ds)},
        )

    def train(self, train_loader, val_loader, train_ds=None, val_ds=None):
        """Run full training loop."""
        # Capture train normalization stats for checkpoint (prevents test leakage)
        if train_ds is not None:
            if hasattr(train_ds, 'global_mean') and hasattr(train_ds, 'global_std'):
                self._train_norm_stats = {
                    'mean': train_ds.global_mean,
                    'std': train_ds.global_std,
                }
            self.init_wandb(train_ds, val_ds)

        epochs = self.config['training']['epochs']
        best_epoch = 0
        start_epoch = 0

        # Resume from checkpoint if available
        resume_path = getattr(self.args, 'resume', None)
        if resume_path and Path(resume_path).exists():
            start_epoch, best_epoch = self._load_resume(resume_path)
            print(f"Resumed from epoch {start_epoch}, best_epoch={best_epoch}")

        for epoch in range(start_epoch, epochs):
            # ---- Train ----
            self.model.train()
            train_losses = []
            loss_accum = {}

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                loss, loss_dict = self._train_step(batch)
                self._update_ema()
                train_losses.append(loss.item())
                for k, v in loss_dict.items():
                    loss_accum.setdefault(k, []).append(v)

            avg_train_loss = np.mean(train_losses)
            avg_losses = {k: np.mean(v) for k, v in loss_accum.items()}
            print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}", end='')
            for k, v in avg_losses.items():
                if k != 'total_loss':
                    print(f" | {k}: {v:.4f}", end='')
            print()

            # ---- Validate ----
            val_metrics = self._validate(val_loader)
            early_stop_metric = self.config['training'].get('early_stop_metric', 'combined')
            if early_stop_metric == 'action_f1':
                val_combined_f1 = val_metrics['action_f1']
            else:
                val_combined_f1 = val_metrics['action_f1'] * 0.5 + val_metrics['scenario_f1'] * 0.5
            print(f"  Val Scenario F1: {val_metrics['scenario_f1']:.4f} | "
                  f"Action F1: {val_metrics['action_f1']:.4f} | "
                  f"Combined: {val_combined_f1:.4f}")

            # ---- Logging ----
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            if self.wandb_run is not None:
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_scenario_f1': val_metrics['scenario_f1'],
                    'val_scenario_acc': val_metrics['scenario_acc'],
                    'val_action_f1': val_metrics['action_f1'],
                    'val_action_acc': val_metrics['action_acc'],
                    'val_combined_f1': val_combined_f1,
                    'learning_rate': current_lr,
                    **{f'train_{k}': v for k, v in avg_losses.items()},
                }
                # Per-class F1
                for cls_name, f1_val in val_metrics.get('action_per_class', {}).items():
                    log_dict[f'val_action_f1_{cls_name}'] = f1_val
                for cls_name, f1_val in val_metrics.get('scenario_per_class', {}).items():
                    log_dict[f'val_scenario_f1_{cls_name}'] = f1_val
                wandb.log(log_dict)

            # ---- Early stopping ----
            self.early_stopper(val_combined_f1, self.ema_model if self.ema_model else self.model)
            if self.early_stopper.best_score == val_combined_f1:
                best_epoch = epoch + 1
            if self.early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # ---- Per-epoch checkpoint (for resume) ----
            self._save_resume(epoch + 1, best_epoch)

        # Save models
        self._save_models(best_epoch)

        if self.wandb_run is not None:
            wandb.finish()

        return self.early_stopper.best_score, best_epoch

    def _train_step(self, batch):
        """Single training step."""
        inputs = batch['inputs'].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
            model_out = self.model(inputs)

            loss, loss_dict = self.criterion(model_out, batch)

        self.scaler.scale(loss).backward()
        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss, loss_dict

    @torch.no_grad()
    def _update_ema(self):
        if self.ema_model is None:
            return
        # Update both parameters AND buffers (BatchNorm running_mean/var)
        # Skip non-float buffers like num_batches_tracked (Long dtype)
        for (name, ema_val), (_, model_val) in zip(
            self.ema_model.state_dict().items(),
            self.model.state_dict().items()
        ):
            if ema_val.is_floating_point():
                ema_val.mul_(self.ema_decay).add_(model_val, alpha=1 - self.ema_decay)
            else:
                ema_val.copy_(model_val)

    @torch.no_grad()
    def _validate(self, val_loader):
        """Run validation and compute metrics."""
        eval_model = self.ema_model if self.ema_model is not None else self.model
        eval_model.eval()

        all_s_preds, all_s_labels = [], []
        all_a_preds, all_a_labels = [], []

        for batch in val_loader:
            inputs = batch['inputs'].to(self.device)
            s_labels = batch['scenario_label']
            a_labels = batch['action_labels']

            model_out = eval_model(inputs)
            s_preds = model_out['scenario_logits'].argmax(dim=1).cpu()
            a_preds = model_out['action_logits'].argmax(dim=-1).cpu()

            all_s_preds.extend(s_preds.numpy())
            all_s_labels.extend(s_labels.numpy())

            a_flat = a_labels.view(-1)
            a_pred_flat = a_preds.view(-1)
            valid = a_flat != -1
            all_a_preds.extend(a_pred_flat[valid].numpy())
            all_a_labels.extend(a_flat[valid].numpy())

        # Compute metrics
        s_f1 = f1_score(all_s_labels, all_s_preds, average='macro', zero_division=0)
        s_acc = (np.array(all_s_preds) == np.array(all_s_labels)).mean()
        a_f1 = f1_score(all_a_labels, all_a_preds, average='macro', zero_division=0) if all_a_labels else 0.0
        a_acc = (np.array(all_a_preds) == np.array(all_a_labels)).mean() if all_a_labels else 0.0

        # Per-class F1 with explicit label ordering (dynamic for 4/5-class)
        _full_idx_to_action = {0: 'ObjTransfer', 1: 'TaskOp', 2: 'Stationary',
                               3: 'Locomotion', 4: 'Search'}
        num_actions = self.config.get('num_actions', 5)
        idx_to_action = {i: _full_idx_to_action[i] for i in range(num_actions)}
        all_class_ids = list(range(num_actions))

        action_per_class = {}
        if all_a_labels:
            a_f1_per = f1_score(all_a_labels, all_a_preds, average=None,
                                labels=all_class_ids, zero_division=0)
            for i, f1_val in enumerate(a_f1_per):
                action_per_class[idx_to_action[i]] = f1_val

        # Per-class scenario F1
        idx_to_scenario = {0: 'Carpentry', 1: 'Cleaning', 2: 'Cooking',
                           3: 'DeskWork', 4: 'Gardening', 5: 'MechRepair',
                           6: 'PlayInstrument', 7: 'WalkOutdoors'}
        all_scenario_ids = list(range(len(idx_to_scenario)))

        scenario_per_class = {}
        s_f1_per = f1_score(all_s_labels, all_s_preds, average=None,
                            labels=all_scenario_ids, zero_division=0)
        for i, f1_val in enumerate(s_f1_per):
            scenario_per_class[idx_to_scenario[i]] = f1_val

        # Log confusion matrix to WandB
        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
            try:
                action_names = [idx_to_action[i] for i in all_class_ids]
                cm = wandb.plot.confusion_matrix(
                    y_true=list(all_a_labels), preds=list(all_a_preds),
                    class_names=action_names,
                )
                wandb.log({"action_confusion_matrix": cm})
            except Exception:
                pass  # Don't crash training if WandB fails

        return {
            'scenario_f1': s_f1,
            'scenario_acc': s_acc,
            'action_f1': a_f1,
            'action_acc': a_acc,
            'action_per_class': action_per_class,
            'scenario_per_class': scenario_per_class,
        }

    def _save_resume(self, epoch, best_epoch):
        """Save checkpoint for resuming training."""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        resume_path = output_dir / 'resume_checkpoint.pth'

        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save({
            'epoch': epoch,
            'best_epoch': best_epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'early_stopper_best_score': self.early_stopper.best_score,
            'early_stopper_counter': self.early_stopper.counter,
            'early_stopper_best_model': self.early_stopper.best_model_state,
            'config': self.config,
        }, resume_path)
        print(f"  Resume checkpoint saved: {resume_path} (epoch {epoch})")

    def _load_resume(self, resume_path):
        """Load checkpoint to resume training."""
        ckpt = torch.load(resume_path, map_location=self.device)

        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])

        self.early_stopper.best_score = ckpt.get('early_stopper_best_score')
        self.early_stopper.counter = ckpt.get('early_stopper_counter', 0)
        self.early_stopper.best_model_state = ckpt.get('early_stopper_best_model')

        start_epoch = ckpt['epoch']
        best_epoch = ckpt.get('best_epoch', 0)
        return start_epoch, best_epoch

    def _save_models(self, best_epoch):
        """Save best and last model checkpoints."""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Include norm_stats so test evaluation uses train normalization (no leakage)
        norm_stats = getattr(self, '_train_norm_stats', None)

        if self.early_stopper.best_model_state:
            save_path = output_dir / 'best_model.pth'
            torch.save({
                'model_state_dict': self.early_stopper.best_model_state,
                'best_score': self.early_stopper.best_score,
                'best_epoch': best_epoch,
                'config': self.config,
                'norm_stats': norm_stats,
            }, save_path)
            print(f"Best model saved: {save_path} (score: {self.early_stopper.best_score:.4f})")

        # Save last model
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        save_path = output_dir / 'last_model.pth'
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'config': self.config,
            'norm_stats': norm_stats,
        }, save_path)
        print(f"Last model saved: {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint for fine-tuning."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state_dict = ckpt.get('model_state_dict', ckpt)

        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
