from abc import abstractmethod
import torch
import numpy as np
from typing import Optional, Generic, TypeVar
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .base import BaseProbe, T, _get_config_attr  # Re-use T and helper from base
from .config import KMeansProbeConfig, PCAProbeConfig, MeanDiffProbeConfig


class DirectionalProbe(BaseProbe[T]):
    """Base class for probes computing direction directly (KMeans, PCA, MeanDiff).
    Stores the final direction in the original activation space.
    """

    # Explicitly type hint the buffer
    direction_vector: torch.Tensor

    def __init__(self, config: T):
        super().__init__(config)
        # Stores the final direction (in original activation space)
        # Initialize buffer with zeros matching input size and dtype
        input_size = _get_config_attr(config, "input_size")
        if input_size is None:
            raise ValueError("Config must have input_size for DirectionalProbe")
        self.register_buffer(
            "direction_vector",
            torch.zeros(input_size, dtype=self.dtype),
            persistent=True,
        )
        # Flag to indicate if fit has been run and direction is valid
        self.has_fit: bool = False

    @abstractmethod
    def fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fit the probe (e.g., run KMeans/PCA) and compute the initial direction.
        The input x may be standardized by the trainer.

        Returns:
            The computed direction tensor *before* potential unscaling by the trainer.
            The internal `direction_vector` buffer should NOT be set here.
        """
        pass

    def _get_raw_direction_representation(self) -> torch.Tensor:
        """Return the computed final internal direction (in original activation space)."""
        # Check if the buffer has been initialized and fit has run
        if (
            not hasattr(self, "direction_vector")
            or self.direction_vector is None
            or not self.has_fit
        ):
            print(
                f"Warning: Accessing direction for probe {self.name} before fit() or after loading incompatible state. Returning zero vector."
            )
            # Return a zero vector of the correct shape and device
            input_size = _get_config_attr(self.config, "input_size")
            device = _get_config_attr(self.config, "device")
            if input_size is None:
                # Should not happen if constructor succeeded, but handle defensively
                raise ValueError("Cannot determine input_size to create zero vector.")
            return torch.zeros(input_size, dtype=self.dtype, device=device)
        return self.direction_vector

    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        """Set the final internal direction (in original activation space)."""
        input_size = _get_config_attr(self.config, "input_size")
        device = _get_config_attr(self.config, "device")
        if input_size is None or device is None:
            raise ValueError(
                "Config missing input_size or device for setting direction."
            )

        # Check if buffer exists and shape mismatch
        if hasattr(self, "direction_vector") and self.direction_vector is not None:
            if self.direction_vector.shape != vector.shape:
                # Allow setting if current vector is the initial zeros and target shape matches config
                is_zero_buffer = torch.all(self.direction_vector == 0)
                matches_config_shape = (
                    vector.ndim == 1 and vector.shape[0] == input_size
                )

                if is_zero_buffer and matches_config_shape:
                    print(f"Info: Initializing direction vector for {self.name}.")
                else:
                    raise ValueError(
                        f"Shape mismatch loading vector for {self.name}. Probe direction: {self.direction_vector.shape}, Loaded vector: {vector.shape}"
                    )

        # Ensure the vector is on the correct device and dtype
        vector = vector.to(device=device, dtype=self.dtype)

        # Register or update the buffer
        if not hasattr(self, "direction_vector") or self.direction_vector is None:
            # Register the buffer if it doesn't exist or is None
            self.register_buffer("direction_vector", vector.clone())
        else:
            # Use copy_ for in-place update of existing buffer
            with torch.no_grad():
                self.direction_vector.copy_(vector)

        # Mark that the direction is now set (either by loading or fitting)
        self.has_fit = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input onto the final (normalized) direction. Input x is in original activation space."""
        # Standardization is handled by the trainer externally if needed

        # Get the final interpretable direction (normalized by default)
        direction = self.get_direction(normalized=True)

        # Ensure consistent dtypes for matmul
        x = x.to(dtype=self.dtype)
        direction = direction.to(dtype=self.dtype)

        # Project onto the direction
        # Handle potential dimension mismatch (e.g., x: [B, D], direction: [D])
        if direction.dim() == 1 and x.dim() >= 2:
            # Standard case: project batch onto single vector -> [B]
            return torch.matmul(x, direction)
        elif direction.dim() == x.dim() and direction.shape[-1] == x.shape[-1]:
            # If direction has batch dim matching x, do batch matmul or elementwise product and sum
            # This case is less common for typical directional probes
            # Assuming elementwise product and sum over last dim: [B, D] * [B, D] -> [B]
            return torch.sum(x * direction, dim=-1)
        else:
            # Fallback or error for unexpected shapes
            try:
                # Attempt standard matmul, let PyTorch handle errors
                return torch.matmul(x, direction)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Shape mismatch during forward pass for {self.name}. Input: {x.shape}, Direction: {direction.shape}. Original error: {e}"
                )

    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the computed direction (already in original activation space), applying normalization."""
        direction = self._get_raw_direction_representation().clone()

        # Normalize if requested and configured
        should_normalize = normalized and getattr(
            self.config, "normalize_weights", False
        )
        if should_normalize:
            norm = torch.norm(direction)
            # Avoid division by zero if norm is very small
            if norm > 1e-8:
                direction = direction / norm
            else:
                # If norm is zero (or close), return the zero vector
                direction = torch.zeros_like(direction)

        return direction


class KMeansProbe(DirectionalProbe[KMeansProbeConfig]):
    """K-means clustering based probe."""

    def __init__(self, config: KMeansProbeConfig):
        super().__init__(config)
        # Sklearn model stored internally, not part of state_dict
        self.kmeans_model: Optional[KMeans] = None

    def fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fit K-means and compute direction from centroids.
        Input x may be standardized by the trainer.
        Returns the computed direction tensor *before* potential unscaling.
        """
        if y is None:
            raise ValueError(
                "KMeansProbe requires labels (y) to determine centroid polarity."
            )

        # K-means expects float32
        x_np = x.cpu().numpy().astype(np.float32)
        y_np = y.cpu().numpy()

        self.kmeans_model = KMeans(
            n_clusters=self.config.n_clusters,
            n_init=self.config.n_init,
            random_state=self.config.random_state,
            init="k-means++",  # Specify init strategy
        )

        # Fit K-means
        try:
            cluster_assignments = self.kmeans_model.fit_predict(x_np)
            centroids = self.kmeans_model.cluster_centers_  # Shape: [n_clusters, dim]
        except Exception as e:
            raise RuntimeError(f"Sklearn KMeans fitting failed: {e}")

        # Determine positive and negative centroids based on label correlation
        # Ensure y_np is 1D
        if y_np.ndim > 1:
            y_np = y_np.squeeze()

        # Handle case where a cluster might be empty (highly unlikely with k-means++)
        cluster_labels_mean = np.full(
            self.config.n_clusters, np.nan
        )  # Initialize with NaN
        for i in range(self.config.n_clusters):
            mask = cluster_assignments == i
            if np.any(mask):
                # Calculate mean label only for assigned points
                cluster_labels_mean[i] = np.mean(y_np[mask])
            # No else needed, remains NaN if empty

        # Check for NaN means (empty clusters) before argmax/argmin
        if np.isnan(cluster_labels_mean).all():
            raise ValueError(
                "All K-means clusters were empty or could not calculate mean labels."
            )
        elif np.isnan(cluster_labels_mean).any():
            print("Warning: One or more K-means clusters were empty.")
            # Fallback logic might be needed here if empty clusters are problematic

        # Find centroids most correlated with positive (1) and negative (0) labels
        # Use nanargmax/nanargmin to handle potential empty clusters gracefully
        try:
            pos_centroid_idx = np.nanargmax(cluster_labels_mean)
            neg_centroid_idx = np.nanargmin(cluster_labels_mean)
        except ValueError:
            # This happens if all means are NaN
            raise ValueError(
                "Could not determine positive/negative centroids due to all NaN means (likely all clusters empty)."
            )

        # Check if the same cluster was chosen for both (e.g., only one non-empty cluster)
        if pos_centroid_idx == neg_centroid_idx:
            print(
                "Warning: Could not distinguish positive/negative K-means centroids based on labels (e.g., only one cluster had points or means were equal)."
            )
            # Fallback: Use first two centroids if available and distinct?
            if self.config.n_clusters >= 2 and len(np.unique(cluster_assignments)) >= 2:
                # Find the two clusters with the most points maybe?
                unique_clusters, counts = np.unique(
                    cluster_assignments, return_counts=True
                )
                if len(unique_clusters) >= 2:
                    sorted_indices = np.argsort(counts)[::-1]
                    # Try using the two largest clusters
                    idx1, idx2 = (
                        unique_clusters[sorted_indices[0]],
                        unique_clusters[sorted_indices[1]],
                    )
                    # Arbitrarily assign pos/neg, or maybe based on their (potentially equal) means?
                    print(f"Fallback: Using clusters {idx1} and {idx2} based on size.")
                    # Re-calculate difference based on these indices
                    pos_centroid = centroids[idx1]
                    neg_centroid = centroids[idx2]
                    initial_direction_np = pos_centroid - neg_centroid
                else:  # Only one cluster actually had points
                    raise ValueError(
                        "Cannot compute difference: Only one K-means cluster contained data points."
                    )
            else:
                raise ValueError(
                    f"Cannot compute difference: Only {len(np.unique(cluster_assignments))} K-means cluster(s) found with data, or means were indistinguishable."
                )
        else:
            pos_centroid = centroids[pos_centroid_idx]
            neg_centroid = centroids[neg_centroid_idx]
            # Direction is from negative to positive centroid
            initial_direction_np = pos_centroid - neg_centroid

        # Return the initial direction tensor (trainer will handle unscaling and setting)
        initial_direction_tensor = torch.tensor(
            initial_direction_np, device=self.config.device, dtype=self.dtype
        )

        # Mark fit as having run (direction_vector buffer is set by trainer later)
        # self.has_fit = True # DO NOT SET here, trainer does after unscaling
        return initial_direction_tensor


class PCAProbe(DirectionalProbe[PCAProbeConfig]):
    """PCA-based probe."""

    def __init__(self, config: PCAProbeConfig):
        super().__init__(config)
        self.pca_model: Optional[PCA] = None  # Store sklearn model if needed later

    def fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fit PCA and determine direction sign based on correlation with labels.
        Input x may be standardized by the trainer.
        Returns the computed direction tensor *before* potential unscaling.
        """
        # PCA works best with float32 or float64
        x_np = x.cpu().numpy().astype(np.float32)

        # Ensure n_components is valid
        n_samples, n_features = x_np.shape
        n_components = min(self.config.n_components, n_samples, n_features)
        if n_components != self.config.n_components:
            print(
                f"Warning: n_components reduced from {self.config.n_components} to {n_components} due to data shape {x_np.shape}"
            )

        if n_components == 0:
            raise ValueError(
                f"Cannot perform PCA with 0 components (data shape {x_np.shape})."
            )

        self.pca_model = PCA(n_components=n_components)

        # Fit PCA
        try:
            self.pca_model.fit(x_np)
            # Components are rows in pca_model.components_
            # Shape: [n_components, dim]
            components = self.pca_model.components_
        except Exception as e:
            raise RuntimeError(f"Sklearn PCA fitting failed: {e}")

        # Get the first principal component
        pc1 = components[0]  # Shape: [dim]

        # Determine sign based on correlation with labels if provided
        if y is not None:
            y_np = y.cpu().numpy()
            if y_np.ndim > 1:
                y_np = y_np.squeeze()

            # Ensure y_np has the same number of samples as x_np
            if len(y_np) != n_samples:
                raise ValueError(
                    f"PCA fit input x ({n_samples} samples) and labels y ({len(y_np)} samples) have different lengths."
                )

            # Project potentially standardized data onto the first component
            projections = np.dot(x_np, pc1.T)  # Shape: [batch]

            # Calculate correlation between projections and labels
            try:
                # Ensure labels are numeric for correlation
                correlation = np.corrcoef(projections, y_np.astype(float))[0, 1]
                # Flip component sign if correlation is negative
                sign = np.sign(correlation) if not np.isnan(correlation) else 1.0
                pc1 = pc1 * sign
            except ValueError as e:
                print(
                    f"Warning: Could not calculate correlation between PCA projections and labels: {e}. Using original PC sign."
                )
            except IndexError:
                print(
                    "Warning: Could not extract correlation coefficient. Using original PC sign."
                )

        # Return the initial direction (first PC, potentially sign-corrected)
        initial_direction_tensor = torch.tensor(
            pc1, device=self.config.device, dtype=self.dtype
        )

        # Mark fit as having run (direction_vector buffer is set by trainer later)
        # self.has_fit = True # DO NOT SET here
        return initial_direction_tensor


class MeanDifferenceProbe(DirectionalProbe[MeanDiffProbeConfig]):
    """Probe finding direction through mean difference between classes."""

    def fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute direction as difference between class means.
        Input x may be standardized by the trainer.
        Returns the computed direction tensor *before* potential unscaling.
        """
        if y is None:
            raise ValueError("MeanDifferenceProbe requires labels (y).")

        # Ensure consistent dtypes and device
        x = x.to(dtype=self.dtype, device=self.config.device)
        y = y.to(device=self.config.device)  # Let mask handle dtype comparison

        # Calculate means for positive (1) and negative (0) classes
        # Ensure y is boolean or integer {0, 1} for masking
        pos_mask = (y == 1).squeeze()
        neg_mask = (y == 0).squeeze()

        # Check if masks are valid
        if not torch.any(pos_mask):
            raise ValueError(
                "MeanDifferenceProbe requires data for the positive class (label 1)."
            )
        if not torch.any(neg_mask):
            raise ValueError(
                "MeanDifferenceProbe requires data for the negative class (label 0)."
            )

        # Filter data and compute means
        pos_data = x[pos_mask]
        neg_data = x[neg_mask]

        # Ensure data is not empty before computing mean
        if pos_data.shape[0] == 0 or neg_data.shape[0] == 0:
            # This should be caught by torch.any checks above, but double-check
            raise ValueError("One class has no data after masking.")

        pos_mean = pos_data.mean(dim=0)
        neg_mean = neg_data.mean(dim=0)

        # Direction from negative to positive mean
        # This initial direction is potentially in the standardized space
        initial_direction_tensor = pos_mean - neg_mean

        # Return the initial direction
        # self.has_fit = True # DO NOT SET here
        return initial_direction_tensor
