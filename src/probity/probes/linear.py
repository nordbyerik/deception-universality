import torch
import torch.nn as nn

from .base import BaseProbe
from .config import LinearProbeConfig


class LinearProbe(BaseProbe[LinearProbeConfig]):
    """Simple linear probe for regression or finding directions.

    Learns a linear transformation Wx + b. The direction is derived from W.
    Operates on original activation space.
    """

    def __init__(self, config: LinearProbeConfig):
        super().__init__(config)
        self.linear = nn.Linear(config.input_size, config.output_size, bias=config.bias)

        # Initialize weights (optional, can use default PyTorch init)
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="linear")
        if config.bias and self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input x is expected in the original activation space."""
        # Standardization is handled by the trainer externally if needed
        return self.linear(x)

    def _get_raw_direction_representation(self) -> torch.Tensor:
        """Return the raw linear layer weights."""
        return self.linear.weight.data

    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        """Set the raw linear layer weights."""
        if self.linear.weight.shape != vector.shape:
            # Reshape if necessary (e.g., loaded [dim] but need [1, dim])
            if (
                self.linear.weight.dim() == 2
                and self.linear.weight.shape[0] == 1
                and vector.dim() == 1
            ):
                vector = vector.unsqueeze(0)
            elif (
                self.linear.weight.dim() == 1
                and vector.dim() == 2
                and vector.shape[0] == 1
            ):
                vector = vector.squeeze(0)
            else:
                raise ValueError(
                    f"Shape mismatch loading vector. Probe weight: {self.linear.weight.shape}, Loaded vector: {vector.shape}"
                )
        with torch.no_grad():
            self.linear.weight.copy_(vector)

    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe direction, applying normalization."""
        # Start with raw weights (already in original activation space)
        direction = self._get_raw_direction_representation().clone()

        # Normalize if requested and configured
        should_normalize = normalized and self.config.normalize_weights
        if should_normalize:
            if self.config.output_size > 1:
                # Normalize each output direction independently
                norms = torch.norm(direction, dim=1, keepdim=True)
                direction = direction / (norms + 1e-8)
            else:
                # Normalize the single direction vector
                norm = torch.norm(direction)
                direction = direction / (norm + 1e-8)

        # Squeeze if single output dimension for convenience
        if self.config.output_size == 1:
            direction = direction.squeeze(0)

        return direction

    # get_loss_fn remains specific to LinearProbe, not moved to base
    def get_loss_fn(self) -> nn.Module:
        """Selects loss function based on config."""
        if self.config.loss_type == "mse":
            return nn.MSELoss()
        elif self.config.loss_type == "cosine":
            # CosineEmbeddingLoss expects targets y = 1 or -1
            # Input: (x1, x2, y) -> computes loss based on y * cos(x1, x2)
            # Here, pred is x1, target direction (implicit) is x2, label is y
            # We might need a wrapper if target vectors aren't directly available
            print(
                "Warning: Cosine loss in LinearProbe assumes target vectors are handled externally."
            )
            return nn.CosineEmbeddingLoss()
        elif self.config.loss_type == "l1":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
