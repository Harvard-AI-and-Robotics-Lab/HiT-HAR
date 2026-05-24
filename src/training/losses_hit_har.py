"""
Loss functions for HiT-HAR training.

Components:
  - FocalLoss: reused from old repo (class-imbalanced task loss)
  - HiTHARLoss: combined task loss orchestrator (scenario + action)

Per FINAL_DESIGN_CN.md §4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss with confidence weighting and optional label smoothing."""

    def __init__(self, gamma=2.0, alpha=None, ignore_index=-1,
                 reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer('alpha', alpha.clone() if isinstance(alpha, torch.Tensor)
                                 else torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, inputs, targets, sample_weights=None):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
            sample_weights: (N,) optional per-sample confidence weights
        """
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]
        if sample_weights is not None:
            sample_weights = sample_weights[mask]

        if inputs.numel() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        num_classes = inputs.shape[1]

        # Label smoothing: convert hard targets to soft
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_p = F.log_softmax(inputs, dim=1)
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()

        focal_weight = (1 - p_t) ** self.gamma

        if self.label_smoothing > 0:
            # Focal-weighted KL divergence for smooth targets
            ce_loss = -(smooth_targets * log_p).sum(dim=1)
        else:
            ce_loss = -log_p_t

        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Apply per-sample confidence weights
        if sample_weights is not None:
            focal_loss = sample_weights * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class HiTHARLoss(nn.Module):
    """Combined HiT-HAR task loss.

    L_total = beta * L_scenario + (1 - beta) * L_action [+ lambda_lang * L_language]

    Beta modes:
      - Fixed (default): beta_task from config, no gradient.
      - Global learnable: single scalar beta learned via sigmoid, config 'learnable_beta'=True.
      - Per-class learnable: per-action-class beta, config 'per_class_beta'=True.
    """

    def __init__(self, config):
        super().__init__()
        self.learnable_beta = config.get('learnable_beta', False)
        self.per_class_beta = config.get('per_class_beta', False)
        self.num_actions = config.get('num_actions', 5)

        if self.per_class_beta:
            # Per-class learnable beta: one parameter per action class
            # Initialized at 0.0 -> sigmoid(0) = 0.5 (equal weighting)
            init_val = config.get('beta_task', 0.5)
            # Convert initial beta value to logit space: logit = log(p/(1-p))
            init_logit = torch.log(torch.tensor(init_val / (1.0 - init_val + 1e-8)))
            self.beta_per_class = nn.Parameter(
                torch.full((self.num_actions,), init_logit.item())
            )
            self.beta_task = None  # not used in per-class mode
        elif self.learnable_beta:
            # Global learnable beta via sigmoid
            init_val = config.get('beta_task', 0.5)
            init_logit = torch.log(torch.tensor(init_val / (1.0 - init_val + 1e-8)))
            self.beta_task_param = nn.Parameter(torch.tensor(init_logit.item()))
            self.beta_task = None  # not used directly
        else:
            # Fixed beta (backward compatible)
            self.beta_task = config.get('beta_task', 0.7)

        self.lambda_lang = config.get('lambda_lang', 0.0)
        label_smoothing = config.get('label_smoothing', 0.0)

        # Scenario loss
        scenario_weights = config.get('scenario_weights', None)
        if scenario_weights is not None:
            scenario_weights = torch.tensor(scenario_weights, dtype=torch.float32)
        self.criterion_scenario = FocalLoss(gamma=2.0, alpha=scenario_weights)

        # Action loss (hard label path)
        action_weights = config.get('action_weights', None)
        if action_weights is not None:
            action_weights = torch.tensor(action_weights, dtype=torch.float32)
        self.criterion_action = FocalLoss(gamma=2.0, alpha=action_weights,
                                          label_smoothing=label_smoothing)

        # Coarse auxiliary loss (optional)
        self.lambda_coarse = config.get('lambda_coarse', 0.0)
        if self.lambda_coarse > 0:
            self.criterion_coarse = FocalLoss(gamma=1.5, ignore_index=-1)

        # Dual-space losses (optional)
        self.lambda_orth = config.get('lambda_orth', 0.0)
        self.lambda_var = config.get('lambda_var', 0.0)
        self.dual_space = config.get('dual_space', False)

        # Language alignment loss (optional)
        self.lang_loss = None
        if self.lambda_lang > 0:
            from src.training.language_loss import LanguageAlignmentLoss
            if self.dual_space:
                self.lang_loss = LanguageAlignmentLoss(
                    embedding_dim=config.get('embedding_dim', 128),
                    num_classes=config.get('num_actions', 5),
                    temperature=0.15,  # Increased for smaller subspace
                    lang_subspace_dim=config.get('lang_subspace_dim', 64),
                )
            else:
                self.lang_loss = LanguageAlignmentLoss(
                    embedding_dim=config.get('embedding_dim', 128),
                    num_classes=config.get('num_actions', 5),
                )

    def forward(self, model_output, batch):
        """Compute combined task loss."""
        device = model_output['scenario_logits'].device
        scenario_logits = model_output['scenario_logits']
        action_logits = model_output['action_logits']

        scenario_labels = batch['scenario_label'].to(device)
        action_labels = batch['action_labels'].to(device)

        B, S, A = action_logits.shape
        action_logits_flat = action_logits.view(-1, A)
        action_labels_flat = action_labels.view(-1)

        # Extract per-sample confidence weights if available
        confidence_weights = None
        if 'confidence' in batch:
            confidence_weights = batch['confidence'].to(device).view(-1)

        # ---- Scenario task loss ----
        loss_scenario = self.criterion_scenario(scenario_logits, scenario_labels)

        # ---- Action task loss ----
        loss_action = self.criterion_action(action_logits_flat, action_labels_flat,
                                            sample_weights=confidence_weights)

        # ---- Total task loss (beta-weighted) ----
        loss_dict = {
            'loss_scenario': loss_scenario.item(),
            'loss_action': loss_action.item(),
        }

        if self.per_class_beta:
            # Per-class learnable beta: each action class gets its own scenario weight
            betas = torch.sigmoid(self.beta_per_class)  # (num_actions,)

            # Compute per-class action losses
            mask = action_labels_flat != -1
            if mask.any():
                valid_logits = action_logits_flat[mask]
                valid_labels = action_labels_flat[mask]
                valid_conf = confidence_weights[mask] if confidence_weights is not None else None

                # Per-class action loss: weight each sample's action loss by (1 - beta_c)
                # where c is the true class of that sample
                per_sample_beta = betas[valid_labels]  # (N_valid,)
                per_sample_action_weight = 1.0 - per_sample_beta  # (N_valid,)

                # Compute per-sample action focal loss (no reduction)
                action_loss_unreduced = self.criterion_action(
                    valid_logits, valid_labels,
                    sample_weights=valid_conf,
                )
                # If reduction='mean' was used, we need unreduced loss;
                # recompute with reduction='none' is complex; simpler: use mean beta
                # Actually, use the mean approach: weighted action + weighted scenario
                beta_mean = betas.mean()
                total_loss = beta_mean * loss_scenario + (1.0 - beta_mean) * loss_action

                # Add per-class refinement: adjust total by per-class beta deviation
                # This ensures gradients flow to individual beta_per_class entries
                for c in range(self.num_actions):
                    class_mask = valid_labels == c
                    if class_mask.any():
                        # Per-class focal loss
                        class_logits = valid_logits[class_mask]
                        class_labels = valid_labels[class_mask]
                        class_conf = valid_conf[class_mask] if valid_conf is not None else None
                        loss_c = self.criterion_action(class_logits, class_labels,
                                                       sample_weights=class_conf)
                        # Contribution: replace mean-weighted with class-specific weight
                        # delta = (1 - beta_c) * loss_c - (1 - beta_mean) * loss_c
                        #       = (beta_mean - beta_c) * loss_c
                        n_c = class_mask.sum().float()
                        n_total = mask.sum().float()
                        weight_ratio = n_c / n_total
                        total_loss = total_loss + (beta_mean - betas[c]) * loss_c * weight_ratio
            else:
                beta_mean = betas.mean()
                total_loss = beta_mean * loss_scenario

            # Log per-class betas
            for c in range(self.num_actions):
                loss_dict[f'beta_class_{c}'] = betas[c].item()
            loss_dict['beta_mean'] = betas.mean().item()

        elif self.learnable_beta:
            # Global learnable beta via sigmoid
            beta = torch.sigmoid(self.beta_task_param)
            total_loss = beta * loss_scenario + (1.0 - beta) * loss_action
            loss_dict['beta_value'] = beta.item()

        else:
            # Fixed beta — unified formula: β * L_scenario + (1-β) * L_action
            # This matches learnable beta formula for fair comparison
            total_loss = self.beta_task * loss_scenario + (1.0 - self.beta_task) * loss_action

        # ---- Coarse auxiliary loss (if enabled and model provides coarse logits) ----
        if self.lambda_coarse > 0 and 'coarse_logits' in model_output:
            coarse_logits = model_output['coarse_logits']  # (B, S, 3)
            fine_to_coarse = model_output['fine_to_coarse']  # (num_fine_classes,)
            coarse_labels = torch.full_like(action_labels, -1)  # (B, S)
            valid = action_labels != -1
            coarse_labels[valid] = fine_to_coarse[action_labels[valid]]
            coarse_logits_flat = coarse_logits.view(-1, coarse_logits.shape[-1])
            coarse_labels_flat = coarse_labels.view(-1)
            loss_coarse = self.criterion_coarse(coarse_logits_flat, coarse_labels_flat)
            total_loss = total_loss + self.lambda_coarse * loss_coarse
            loss_dict['loss_coarse'] = loss_coarse.item()

        # ---- Language alignment loss (if prototypes provided by model) ----
        if self.lang_loss is not None and 'text_prototypes' in model_output:
            prototypes = model_output['text_prototypes'].to(device)
            lang_loss, lang_dict = self.lang_loss(model_output, prototypes, batch)
            total_loss = total_loss + self.lambda_lang * lang_loss
            loss_dict.update(lang_dict)

        # ---- Dual-space orthogonality + variance losses ----
        if self.dual_space and 'z_lang' in model_output and 'z_sensor' in model_output:
            from src.training.orthogonality_loss import orthogonality_loss, variance_penalty
            z_lang = model_output['z_lang']
            z_sensor = model_output['z_sensor']

            if self.lambda_orth > 0:
                loss_orth = orthogonality_loss(z_lang, z_sensor)
                total_loss = total_loss + self.lambda_orth * loss_orth
                loss_dict['loss_orth'] = loss_orth.item()

            if self.lambda_var > 0:
                loss_var = variance_penalty(z_lang)
                total_loss = total_loss + self.lambda_var * loss_var
                loss_dict['loss_var_lang'] = loss_var.item()

            # Observability gate entropy regularizer
            if 'obs_alpha' in model_output:
                from src.training.orthogonality_loss import alpha_entropy_regularizer
                alpha = model_output['obs_alpha']
                loss_entropy = alpha_entropy_regularizer(alpha)
                lambda_entropy = 0.01
                total_loss = total_loss + lambda_entropy * loss_entropy
                loss_dict['loss_alpha_entropy'] = loss_entropy.item()
                # Log per-class alpha values
                for i, val in enumerate(alpha.detach().cpu().tolist()):
                    loss_dict[f'alpha_class_{i}'] = val

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict
