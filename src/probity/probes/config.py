from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal
import torch


@dataclass
class ProbeConfig:
    """Base configuration for probes with metadata."""

    # Core configuration
    input_size: int
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Metadata fields (from former ProbeVector)
    model_name: str = "unknown_model"
    hook_point: str = "unknown_hook"
    hook_layer: int = 0
    hook_head_index: Optional[int] = None
    name: str = "unnamed_probe"

    # Dataset information
    dataset_path: Optional[str] = None
    prepend_bos: bool = True
    context_size: int = 128

    # Technical settings
    dtype: str = "float32"

    # Additional metadata
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinearProbeConfig(ProbeConfig):
    """Configuration for linear probe."""

    loss_type: Literal["mse", "cosine", "l1"] = "mse"
    normalize_weights: bool = True
    bias: bool = False
    output_size: int = 1  # Number of output dimensions


@dataclass
class LogisticProbeConfig(ProbeConfig):
    """Configuration for logistic regression probe."""

    normalize_weights: bool = True
    bias: bool = True
    output_size: int = 1  # Number of output dimensions


@dataclass
class MultiClassLogisticProbeConfig(ProbeConfig):
    """Configuration for multi-class logistic regression probe."""

    output_size: int = 2  # Must be specified, > 1
    normalize_weights: bool = True
    bias: bool = True


@dataclass
class KMeansProbeConfig(ProbeConfig):
    """Configuration for K-means clustering probe."""

    n_clusters: int = 2
    n_init: int = 10
    normalize_weights: bool = True
    random_state: int = 42


@dataclass
class PCAProbeConfig(ProbeConfig):
    """Configuration for PCA-based probe."""

    n_components: int = 1
    normalize_weights: bool = True


@dataclass
class MeanDiffProbeConfig(ProbeConfig):
    """Configuration for mean difference probe."""

    normalize_weights: bool = True


# Configs for SklearnLogisticProbe
@dataclass
class LogisticProbeConfigBase(ProbeConfig):
    """Base config shared by sklearn implementations."""

    standardize: bool = True  # Internal standardization for sklearn
    normalize_weights: bool = True
    bias: bool = True
    output_size: int = 1  # Usually 1 for logistic


@dataclass
class SklearnLogisticProbeConfig(LogisticProbeConfigBase):
    """Config for sklearn-based probe."""

    max_iter: int = 100
    random_state: int = 42
    solver: str = "lbfgs"  # Example of adding solver
