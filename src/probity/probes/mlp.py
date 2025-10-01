import torch
import torch.nn as nn
from typing import Optional

from .base import BaseProbe
from .config import LogisticProbeConfig, MultiClassLogisticProbeConfig

class MLPProbeConfig(BaseProbeConfig):
    """Configuration for MLP probe."""
    hidden_size: int = 128
    num_hidden_layers: int = 1
    activation: str = "relu"  # "relu", "gelu", "tanh"
    dropout: float = 0.0
    
    
class MLPProbe(BaseProbe[MLPProbeConfig]):
    """Multi-layer perceptron probe for non-linear feature extraction."""
    
    def __init__(self, config: MLPProbeConfig):
        super().__init__(config)
        
        # Build MLP layers
        layers = []
        current_size = config.input_size
        
        # Hidden layers
        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(current_size, config.hidden_size, bias=config.bias))
            layers.append(self._get_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            current_size = config.hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_size, config.output_size, bias=config.bias))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization often works well
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits."""
        return self.mlp(x)
    
    def _get_raw_direction_representation(self) -> torch.Tensor:
        """For MLP, return the first layer weights as a rough 'direction'."""
        # This is a bit of a hack since MLPs don't have a single direction
        # But can be useful for visualization/comparison purposes
        first_linear = next(m for m in self.mlp.modules() if isinstance(m, nn.Linear))
        return first_linear.weight.data
    
    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        """Set first layer weights. Less meaningful for MLPs but included for interface."""
        first_linear = next(m for m in self.mlp.modules() if isinstance(m, nn.Linear))
        with torch.no_grad():
            if first_linear.weight.shape != vector.shape:
                raise ValueError(
                    f"Shape mismatch. First layer: {first_linear.weight.shape}, Vector: {vector.shape}"
                )
            first_linear.weight.copy_(vector)
    
    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get first layer weights as a proxy for 'direction'."""
        direction = self._get_raw_direction_representation().clone()
        
        if normalized and self.config.normalize_weights:
            if direction.dim() == 2:
                norms = torch.norm(direction, dim=1, keepdim=True)
                direction = direction / (norms + 1e-8)
            else:
                norm = torch.norm(direction)
                direction = direction / (norm + 1e-8)
        
        return direction
    
    def get_loss_fn(self, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
        """Get binary cross entropy loss with logits."""
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)