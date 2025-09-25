# __init__.py for probity.probes

# Import probe configuration classes
from .config import (
    ProbeConfig,
    LinearProbeConfig,
    LogisticProbeConfig,
    MultiClassLogisticProbeConfig,
    KMeansProbeConfig,
    PCAProbeConfig,
    MeanDiffProbeConfig,
    LogisticProbeConfigBase,  # Base for sklearn
    SklearnLogisticProbeConfig,
)

# Import the base probe class
from .base import BaseProbe

# Import concrete probe implementations
from .linear import LinearProbe
from .logistic import LogisticProbe, MultiClassLogisticProbe
from .directional import (
    DirectionalProbe,  # Base for non-learned direction probes
    KMeansProbe,
    PCAProbe,
    MeanDifferenceProbe,
)
from .sklearn_logistic import SklearnLogisticProbe

# Import the ProbeSet class
from .probe_set import ProbeSet


# Define __all__ for explicit public API
__all__ = [
    # Configs
    "ProbeConfig",
    "LinearProbeConfig",
    "LogisticProbeConfig",
    "MultiClassLogisticProbeConfig",
    "KMeansProbeConfig",
    "PCAProbeConfig",
    "MeanDiffProbeConfig",
    "LogisticProbeConfigBase",
    "SklearnLogisticProbeConfig",
    # Base Class
    "BaseProbe",
    # Concrete Probes
    "LinearProbe",
    "LogisticProbe",
    "MultiClassLogisticProbe",
    "DirectionalProbe",
    "KMeansProbe",
    "PCAProbe",
    "MeanDifferenceProbe",
    "SklearnLogisticProbe",
    # Probe Collection
    "ProbeSet",
]
