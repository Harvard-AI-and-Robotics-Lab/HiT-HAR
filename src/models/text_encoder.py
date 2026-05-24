# src/models/text_encoder.py
"""Frozen text encoder with trainable projection for language-anchored HAR.

Uses all-mpnet-base-v2 (768-dim) frozen, projects to IMU embedding space.
Class prototypes are pre-computed from taxonomy descriptions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


# Motion-native class descriptions: what a head-mounted IMU would actually measure
CLASS_DESCRIPTIONS = {
    "Object Transfer": (
        "Head tilts downward to look at source, then reorients toward destination. "
        "Brief vertical acceleration during bending or reaching. "
        "Periodic head rotation between two focal points with pauses at each. "
        "Gyroscope shows alternating yaw as gaze shifts between pickup and placement locations."
    ),
    "Task Operation": (
        "Head maintains steady forward orientation with minimal angular velocity. "
        "Sustained slight downward tilt while focusing on a work surface. "
        "Low-amplitude, high-frequency micro-vibrations from manual tool use transmitted to head. "
        "Accelerometer shows stable gravity alignment with occasional small perturbations."
    ),
    "Stationary": (
        "Near-zero acceleration magnitude beyond gravity. Minimal gyroscope activity. "
        "Head orientation is stable with only slow drift from postural sway. "
        "No periodic patterns, no sudden orientation changes. "
        "The IMU signal is dominated by sensor noise rather than intentional motion."
    ),
    "Locomotion": (
        "Periodic vertical acceleration from gait cycle — heel strike creates "
        "rhythmic head bobbing at walking frequency (1.5-2.5 Hz). "
        "Forward acceleration component with sinusoidal pattern. "
        "Gyroscope shows periodic roll and pitch oscillations synchronized with stride."
    ),
    "Search": (
        "Active head rotation with high angular velocity in yaw axis. "
        "Rapid orientation changes as gaze sweeps across the environment. "
        "Gyroscope shows bursts of rotational activity followed by brief fixation pauses. "
        "Distinct from stationary by the magnitude and frequency of rotational signals."
    ),
}

# Action name → class ID mapping (must match paired_dataset.py ACTION_MAP)
CLASS_ORDER = ["Object Transfer", "Task Operation", "Stationary", "Locomotion", "Search"]


class TextAnchorEncoder(nn.Module):
    """Frozen text encoder with trainable projection to IMU embedding space.

    Pre-computes class prototype embeddings from taxonomy descriptions.
    Optionally encodes instance narrations for fine-grained alignment.
    """

    def __init__(self, projection_dim=128, text_model_name='all-mpnet-base-v2',
                 class_names=None):
        super().__init__()
        assert HAS_SBERT, "sentence-transformers required: pip install sentence-transformers"

        self.projection_dim = projection_dim
        self.class_names = class_names if class_names is not None else CLASS_ORDER

        # Frozen text encoder
        self.text_model = SentenceTransformer(text_model_name)
        self.text_dim = self.text_model.get_sentence_embedding_dimension()
        # Fallback: probe dimension by encoding a dummy sentence
        if self.text_dim is None:
            dummy = self.text_model.encode(["test"], convert_to_tensor=True)
            self.text_dim = dummy.shape[-1]
        # Freeze all parameters
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Trainable projection: text_dim → projection_dim
        self.projection = nn.Sequential(
            nn.Linear(self.text_dim, self.text_dim // 2),
            nn.GELU(),
            nn.Linear(self.text_dim // 2, projection_dim),
        )

        # Pre-compute and cache class prototypes
        self._class_prototypes = None

    @torch.no_grad()
    def _encode_texts(self, texts):
        """Encode texts using frozen sentence transformer. Returns (N, text_dim)."""
        embeddings = self.text_model.encode(texts, convert_to_tensor=True,
                                            show_progress_bar=False)
        return embeddings.float()

    def get_class_prototypes(self):
        """Get L2-normalized class prototype embeddings. Shape: (num_classes, projection_dim).

        Cached after first call. Projection is applied (trainable).
        Uses self.class_names (defaults to CLASS_ORDER) for dynamic class subset support.
        """
        if self._class_prototypes is None or self.training:
            descriptions = [CLASS_DESCRIPTIONS[cls] for cls in self.class_names]
            raw_embeds = self._encode_texts(descriptions)  # (num_classes, text_dim)
            # Move to same device as projection layer (text_model returns CPU tensors)
            raw_embeds = raw_embeds.to(next(self.projection.parameters()).device)
            projected = self.projection(raw_embeds)          # (5, projection_dim)
            projected = F.normalize(projected, dim=1)
            if not self.training:
                self._class_prototypes = projected
            return projected
        return self._class_prototypes

    def encode_narrations(self, narrations):
        """Encode instance narrations. Shape: (N, projection_dim).

        Used for instance-level alignment (optional, noise-aware).
        """
        raw = self._encode_texts(narrations)     # (N, text_dim)
        raw = raw.to(next(self.projection.parameters()).device)
        projected = self.projection(raw)          # (N, projection_dim)
        return F.normalize(projected, dim=1)

    def invalidate_cache(self):
        """Call when projection weights change (e.g., after optimizer step)."""
        self._class_prototypes = None
