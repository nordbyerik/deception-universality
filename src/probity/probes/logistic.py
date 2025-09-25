import torch
import torch.nn as nn
from typing import Optional

from .base import BaseProbe
from .config import LogisticProbeConfig, MultiClassLogisticProbeConfig


class LogisticProbe(BaseProbe[LogisticProbeConfig]):
    """Logistic regression probe implemented using nn.Linear. Operates on original activation space."""

    def __init__(self, config: LogisticProbeConfig):
        super().__init__(config)
        # Logistic regression is essentially a linear layer followed by sigmoid (handled by loss)
        self.linear = nn.Linear(config.input_size, config.output_size, bias=config.bias)

        # Initialize weights (zeros often work well for logistic init)
        nn.init.zeros_(self.linear.weight)
        if config.bias and self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits. Input x is expected in the original activation space."""
        # Standardization is handled by the trainer externally if needed
        return self.linear(x)

    def _get_raw_direction_representation(self) -> torch.Tensor:
        """Return the raw linear layer weights."""
        return self.linear.weight.data

    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        """Set the raw linear layer weights."""
        if self.linear.weight.shape != vector.shape:
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
                norms = torch.norm(direction, dim=1, keepdim=True)
                direction = direction / (norms + 1e-8)
            else:
                norm = torch.norm(direction)
                direction = direction / (norm + 1e-8)

        # Squeeze if single output dimension
        if self.config.output_size == 1:
            direction = direction.squeeze(0)

        return direction

    def get_loss_fn(self, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
        """Get binary cross entropy loss with logits.

        Args:
            pos_weight: Optional weight for positive class (for class imbalance).
        """
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


class MultiClassLogisticProbe(BaseProbe[MultiClassLogisticProbeConfig]):
    """Multi-class logistic regression probe (Softmax Regression)."""

    def __init__(self, config: MultiClassLogisticProbeConfig):
        super().__init__(config)
        if config.output_size <= 1:
            raise ValueError("MultiClassLogisticProbe requires output_size > 1.")
        self.linear = nn.Linear(config.input_size, config.output_size, bias=config.bias)

        # Initialize weights
        nn.init.zeros_(self.linear.weight)
        if config.bias and self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits for each class."""
        return self.linear(x)

    def _get_raw_direction_representation(self) -> torch.Tensor:
        """Return the raw linear layer weights (weight matrix)."""
        return self.linear.weight.data

    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        """Set the raw linear layer weights (weight matrix)."""
        if self.linear.weight.shape != vector.shape:
            raise ValueError(
                f"Shape mismatch loading vector. Probe weight: "
                f"{self.linear.weight.shape}, Loaded vector: {vector.shape}"
            )
        with torch.no_grad():
            self.linear.weight.copy_(vector)

    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe directions (weight matrix), applying normalization per class."""
        # Start with raw weights
        directions = self._get_raw_direction_representation().clone()

        # Normalize if requested and configured
        should_normalize = normalized and self.config.normalize_weights
        if should_normalize:
            # Normalize each class direction (row) independently
            norms = torch.norm(directions, dim=1, keepdim=True)
            directions = directions / (norms + 1e-8)

        return directions  # Shape [output_size, input_size]

    def get_loss_fn(self, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        """Get cross entropy loss.

        Args:
            class_weights: Optional weights for each class (for class imbalance).
                           Shape [output_size].
        """
        return nn.CrossEntropyLoss(weight=class_weights)
