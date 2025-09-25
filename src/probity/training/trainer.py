from abc import ABC
from dataclasses import dataclass

# from pathlib import Path # Unused
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from typing import Optional, Dict, List, Tuple, Literal  # Removed Any, Type
from tqdm.auto import tqdm
import math

from probity.collection.activation_store import ActivationStore

# Probes now expect non-standardized data for forward pass and store unscaled directions
from probity.probes import (
    BaseProbe,
    DirectionalProbe,
    MultiClassLogisticProbe,  # Add import
    LogisticProbe,  # Added back for instanceof check
    LinearProbe,  # Added for instanceof check
)


@dataclass
class BaseTrainerConfig:
    """Enhanced base configuration shared by all trainers."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: Optional[str] = None
    batch_size: int = 32
    learning_rate: float = 1e-3
    end_learning_rate: float = 1e-5  # For LR scheduling
    weight_decay: float = 0.01
    num_epochs: int = 10
    show_progress: bool = True
    optimizer_type: Literal["Adam", "SGD", "AdamW"] = "Adam"
    handle_class_imbalance: bool = True
    standardize_activations: bool = False  # Option to standardize *during* training


class BaseProbeTrainer(ABC):
    """Enhanced abstract base class for all probe trainers. Handles standardization during training."""

    def __init__(self, config: BaseTrainerConfig):
        self.config = config
        # Store standardization stats if standardization is enabled
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None

    def _get_lr_scheduler(
        self, optimizer: optim.Optimizer, start_lr: float, end_lr: float, num_steps: int
    ) -> optim.lr_scheduler.LRScheduler:
        """Create exponential learning rate scheduler."""
        if start_lr <= 0 or end_lr <= 0:
            # Handle cases where LR might be zero or negative, default to constant LR
            print("Warning: start_lr or end_lr <= 0, using constant LR scheduler.")
            return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        # Ensure num_steps is positive
        if num_steps <= 0:
            print("Warning: num_steps <= 0 for LR scheduler, using constant LR.")
            return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        # Avoid log(0) or division by zero
        if start_lr == end_lr:
            gamma = 1.0
        else:
            # Ensure end_lr / start_lr is positive
            ratio = end_lr / start_lr
            if ratio <= 0:
                print(
                    f"Warning: Invalid learning rate range ({start_lr} -> {end_lr}). "
                    f"Using constant LR."
                )
                return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
            gamma = math.exp(math.log(ratio) / num_steps)

        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    def _calculate_pos_weights(self, y: torch.Tensor) -> torch.Tensor:
        """Calculate positive weights for handling class imbalance.

        Handles both single-output (y shape: [N, 1]) and
        multi-output (y shape: [N, C]) cases.
        """
        if y.dim() == 1:
            y = y.unsqueeze(1)

        # Calculate weights for each output dimension
        num_pos = y.sum(dim=0)
        num_neg = len(y) - num_pos
        weights = num_neg / (num_pos + 1e-8)  # Add epsilon to prevent division by zero

        return weights

    def _calculate_class_weights(
        self, y: torch.Tensor, num_classes: int
    ) -> Optional[torch.Tensor]:
        """Calculate class weights for multi-class CrossEntropyLoss."""
        if y.dim() == 2 and y.shape[1] == 1:
            y = y.squeeze(1)  # Convert [N, 1] to [N]
        if y.dim() != 1:
            print("Warning: Cannot calculate class weights for non-1D target tensor.")
            return None

        # Check dtype before attempting conversion or calculations
        if y.dtype != torch.long:
            print(
                f"Warning: Cannot calculate class weights for non-Long target tensor "
                f"(dtype: {y.dtype})."
            )
            return None
            # Note: We no longer attempt conversion for non-long types.
            # If conversion is desired, the calling code should handle it.

        counts = torch.bincount(y, minlength=num_classes)
        # Avoid division by zero for classes with zero samples
        total_samples = counts.sum()
        if total_samples == 0:
            return None  # No samples to calculate weights from

        # Calculate weights: (total_samples / (num_classes * count_for_class))
        # This gives higher weight to less frequent classes.
        weights = total_samples / (num_classes * (counts + 1e-8))

        # Handle cases where a class might have 0 samples (weight will be large but finite due to epsilon)
        # Clamp weights to avoid extreme values? Optional.
        # weights = torch.clamp(weights, min=0.1, max=10.0)

        return weights

    def _create_optimizer(self, model: torch.nn.Module) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_class = getattr(optim, self.config.optimizer_type, None)
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

        # Filter out parameters that do not require gradients
        params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())

        if self.config.optimizer_type in ["Adam", "AdamW"]:
            optimizer = optimizer_class(
                params_to_optimize,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "SGD":
            optimizer = optimizer_class(
                params_to_optimize,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                # Add momentum? momentum=0.9 might be a good default
            )
        else:
            # Should be caught by getattr check, but belts and braces
            raise ValueError(
                f"Unsupported optimizer type: {self.config.optimizer_type}"
            )

        return optimizer

    def prepare_data(
        self,
        activation_store: ActivationStore,
        position_key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for training, optionally applying standardization.

        Computes standardization statistics if standardize_activations is True.
        Returns:
            X_train: Activations potentially standardized for training.
            y: Labels.
            X_orig: Original, non-standardized activations.
        """
        X_orig, y = activation_store.get_probe_data(position_key)
        X_train = X_orig  # Default to original if no standardization

        # Apply standardization only if configured
        if self.config.standardize_activations:
            # Compute statistics if not already done (e.g., first call)
            if self.feature_mean is None or self.feature_std is None:
                self.feature_mean = X_orig.mean(dim=0, keepdim=True)
                # Add epsilon for numerical stability
                self.feature_std = X_orig.std(dim=0, keepdim=True) + 1e-8

                # Move statistics to the correct device
                if self.config.device:
                    target_device = torch.device(self.config.device)
                    self.feature_mean = self.feature_mean.to(target_device)
                    self.feature_std = self.feature_std.to(target_device)

            # Apply standardization to create X_train
            if self.feature_mean is not None and self.feature_std is not None:
                # Ensure X_orig is on the same device as stats before operation
                if hasattr(self.feature_mean, "device"):
                    X_orig_dev = X_orig.to(self.feature_mean.device)
                    X_train = (X_orig_dev - self.feature_mean) / self.feature_std
                else:
                    # Fallback if stats don't have device info (shouldn't happen)
                    X_train = (X_orig - self.feature_mean) / self.feature_std

        return X_train, y, X_orig  # Return both training and original activations

    # Removed transfer_stats_to_model


@dataclass
class SupervisedTrainerConfig(BaseTrainerConfig):
    """Enhanced config for supervised training methods."""

    train_ratio: float = 0.8
    patience: int = 5
    min_delta: float = 1e-4


class SupervisedProbeTrainer(BaseProbeTrainer):
    """Enhanced trainer for supervised probes with progress tracking and LR scheduling."""

    def __init__(self, config: SupervisedTrainerConfig):
        super().__init__(config)
        # Ensure self.config has the more specific type
        if not isinstance(config, SupervisedTrainerConfig):
            raise TypeError("SupervisedProbeTrainer requires a SupervisedTrainerConfig")
        self.config: SupervisedTrainerConfig = config

    def prepare_supervised_data(
        self,
        activation_store: ActivationStore,
        position_key: str,
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare train/val splits with DataLoader creation.
        DataLoaders yield batches of (X_train, y, X_orig).
        """
        X_train_all, y_all, X_orig_all = self.prepare_data(
            activation_store, position_key
        )

        # Split data
        n_total = len(X_orig_all)
        n_train = int(n_total * self.config.train_ratio)
        # Ensure validation set is not empty
        if n_train == n_total:
            n_train = max(
                0, n_total - 1
            )  # Keep at least one sample for validation if possible
            if n_train == 0 and n_total > 0:
                print(
                    "Warning: Only one data point available. Using it for training and validation."
                )
                n_train = 1
            elif n_total > 0:
                print(
                    "Warning: train_ratio resulted in no validation data. "
                    "Adjusting to keep one sample for validation."
                )

        # Ensure train set is not empty if n_total > 0
        if n_train == 0 and n_total > 0:
            print(
                "Warning: train_ratio resulted in no training data. Using all data for training."
            )
            n_train = n_total

        # Generate random permutation
        indices = torch.randperm(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Handle edge case: If either split is empty (can happen if n_total is very small)
        if len(train_indices) == 0 and n_total > 0:
            train_indices = indices  # Use all data for training
            val_indices = indices  # Use all data for validation too (less ideal)
            print(
                "Warning: No training samples after split. Using all data for training and validation."
            )
        elif len(val_indices) == 0 and n_total > 0:
            val_indices = (
                train_indices  # Use training data for validation if val split is empty
            )
            print(
                "Warning: No validation samples after split. "
                "Using training data for validation."
            )

        X_train_split, X_val_split = (
            X_train_all[train_indices],
            X_train_all[val_indices],
        )
        X_orig_train_split, X_orig_val_split = (
            X_orig_all[train_indices],
            X_orig_all[val_indices],
        )
        y_train_split, y_val_split = y_all[train_indices], y_all[val_indices]

        # Create dataloaders
        # Handle both single and multi-dimensional labels
        y_train_split = (
            y_train_split if y_train_split.dim() > 1 else y_train_split.unsqueeze(1)
        )
        y_val_split = y_val_split if y_val_split.dim() > 1 else y_val_split.unsqueeze(1)

        # Note: We don't move tensors to device here because DataLoader will
        # create copies that could waste memory. Instead, we move tensors to
        # device in training loop just before using them.
        train_dataset = TensorDataset(X_train_split, y_train_split, X_orig_train_split)
        val_dataset = TensorDataset(X_val_split, y_val_split, X_orig_val_split)

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        return train_loader, val_loader

    def train_epoch(
        self,
        model: torch.nn.Module,  # Use base Module type here, checked in train
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epoch: int,
        num_epochs: int,
        is_multi_class: bool = False,  # Flag for multi-class loss
    ) -> float:
        """Run one epoch of training with progress tracking. Uses X_train for training."""
        model.train()
        total_loss = 0

        # Create progress bar for batches
        batch_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not self.config.show_progress,
            leave=False,
        )

        for (
            batch_x_train,
            batch_y,
            _,
        ) in batch_pbar:  # Ignore X_orig during training pass
            optimizer.zero_grad()
            batch_x_train = batch_x_train.to(self.config.device)
            batch_y = batch_y.to(self.config.device)

            # --- Adjust target shape/type for loss ---
            if is_multi_class:
                # CrossEntropyLoss expects Long targets of shape [N]
                if batch_y.dim() == 2 and batch_y.shape[1] == 1:
                    batch_y = batch_y.squeeze(1)
                batch_y = batch_y.long()  # Ensure Long type
            else:  # BCEWithLogitsLoss expects Float targets
                batch_y = batch_y.float()  # Ensure Float type
            # -----------------------------------------

            # Model forward pass uses the potentially standardized data
            outputs = model(batch_x_train)
            loss = loss_fn(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        return total_loss / len(train_loader)

    def train(
        self,
        model: BaseProbe,  # Expect a BaseProbe instance
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """Train model, potentially unscale direction if standardization was used."""
        # Ensure model is a BaseProbe instance
        if not isinstance(model, BaseProbe):
            raise TypeError(
                "SupervisedProbeTrainer expects model to be an instance of BaseProbe"
            )

        # Ensure model is on the correct device
        target_device = torch.device(self.config.device)
        model.to(target_device)

        # Determine if this is a multi-class probe
        is_multi_class = isinstance(model, MultiClassLogisticProbe)
        # num_classes = model.config.output_size if hasattr(model.config, 'output_size') else 1 # Removed

        # Standardization stats are managed by the trainer, not transferred

        optimizer = self._create_optimizer(model)
        scheduler = self._get_lr_scheduler(
            optimizer,
            self.config.learning_rate,
            self.config.end_learning_rate,
            self.config.num_epochs,
        )

        # --- Set up loss function ---
        loss_fn: nn.Module
        all_train_y: Optional[torch.Tensor] = None
        if self.config.handle_class_imbalance:
            # Calculate all labels only once if needed for any weight calculation
            all_train_y = torch.cat([y for _, y, _ in train_loader])

        if isinstance(model, MultiClassLogisticProbe):
            weights_arg: Optional[torch.Tensor] = None
            if self.config.handle_class_imbalance and all_train_y is not None:
                num_classes = model.config.output_size
                class_weights = self._calculate_class_weights(all_train_y, num_classes)
                if class_weights is not None:
                    weights_arg = class_weights.to(target_device)
            # Pass `class_weights` keyword arg
            loss_fn = model.get_loss_fn(class_weights=weights_arg)

        elif isinstance(model, LogisticProbe):
            weights_arg: Optional[torch.Tensor] = None
            if self.config.handle_class_imbalance and all_train_y is not None:
                pos_weight = self._calculate_pos_weights(all_train_y)
                if pos_weight is not None:
                    weights_arg = pos_weight.to(target_device)
            # Pass `pos_weight` keyword arg
            loss_fn = model.get_loss_fn(pos_weight=weights_arg)

        elif isinstance(model, LinearProbe):
            # LinearProbe loss (MSE, L1, Cosine) doesn't use these weights
            loss_fn = model.get_loss_fn()
            if self.config.handle_class_imbalance:
                print(
                    f"Warning: Class imbalance handling enabled, but may not be effective "
                    f"for LinearProbe with loss type '{model.config.loss_type}'."
                )
        else:
            # Fallback for other BaseProbe types that might be supervised
            probe_type_name = type(model).__name__
            print(
                f"Warning: Unknown supervised probe type '{probe_type_name}'. "
                f"Attempting default BCEWithLogitsLoss. Ensure this is appropriate."
            )
            weights_arg: Optional[torch.Tensor] = None
            if self.config.handle_class_imbalance and all_train_y is not None:
                pos_weight = self._calculate_pos_weights(all_train_y)
                if pos_weight is not None:
                    weights_arg = pos_weight.to(target_device)
            # Assume BCE loss for unknown types
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights_arg)
        # --- End Loss Function Setup ---

        # Training history
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        # Create progress bar for epochs
        epoch_pbar = tqdm(
            range(self.config.num_epochs),
            desc="Training",
            disable=not self.config.show_progress,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in epoch_pbar:
            # Train epoch uses X_train
            train_loss = self.train_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                epoch,
                self.config.num_epochs,
                is_multi_class=is_multi_class,  # Pass flag
            )
            history["train_loss"].append(train_loss)

            # Validation uses X_orig
            if val_loader is not None:
                val_loss = self.validate(
                    model, val_loader, loss_fn, is_multi_class=is_multi_class
                )
                history["val_loss"].append(val_loss)

                # Early stopping check
                if val_loss < best_val_loss - self.config.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state? Optional
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

                epoch_pbar.set_postfix(
                    {
                        "Train Loss": f"{train_loss:.6f}",
                        "Val Loss": f"{val_loss:.6f}",
                        "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )
            else:
                epoch_pbar.set_postfix(
                    {
                        "Train Loss": f"{train_loss:.6f}",
                        "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            history["learning_rate"].append(scheduler.get_last_lr()[0])
            scheduler.step()

        # --- Post-Training Direction Unscaling ---
        if self.config.standardize_activations and self.feature_std is not None:
            print("Unscaling probe direction...")
            with torch.no_grad():
                # Get the direction learned on standardized data
                learned_direction = model._get_raw_direction_representation()

                # Unscale the direction
                # Ensure std dev matches direction dims for division
                std_dev = self.feature_std.squeeze().to(learned_direction.device)

                # Handle potential shape mismatches (e.g., [1, dim] vs [dim])
                if (
                    learned_direction.dim() == 2
                    and learned_direction.shape[0] == 1
                    and std_dev.dim() == 1
                ):
                    # Common case for Linear/Logistic: weights are [1, dim], std_dev is [dim]
                    unscaled_direction = learned_direction / std_dev.unsqueeze(0)
                elif learned_direction.shape == std_dev.shape:
                    unscaled_direction = learned_direction / std_dev
                elif (
                    learned_direction.dim() == 1
                    and std_dev.dim() == 1
                    and learned_direction.shape[0] == std_dev.shape[0]
                ):
                    # Case for single vector directions (like maybe directional probes before squeeze?)
                    unscaled_direction = learned_direction / std_dev
                else:
                    print(
                        f"Warning: Shape mismatch during final unscaling. Direction: {learned_direction.shape}, StdDev: {std_dev.shape}. Skipping unscaling."
                    )
                    unscaled_direction = (
                        learned_direction  # Keep original if shapes mismatch
                    )

                # Update the probe's internal representation with the unscaled direction
                model._set_raw_direction_representation(unscaled_direction)

        return history

    def validate(
        self,
        model: torch.nn.Module,  # Use base Module type here
        val_loader: DataLoader,
        loss_fn: torch.nn.Module,
        is_multi_class: bool = False,  # Flag for multi-class loss
    ) -> float:
        """Run validation with progress tracking. Uses X_orig for validation."""
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for _, batch_y, batch_x_orig in val_loader:  # Use X_orig for validation
                batch_x_orig = batch_x_orig.to(self.config.device)
                batch_y = batch_y.to(self.config.device)

                # --- Adjust target shape/type for loss ---
                if is_multi_class:
                    # CrossEntropyLoss expects Long targets of shape [N]
                    if batch_y.dim() == 2 and batch_y.shape[1] == 1:
                        batch_y = batch_y.squeeze(1)
                    batch_y = batch_y.long()  # Ensure Long type
                else:  # BCEWithLogitsLoss expects Float targets
                    batch_y = batch_y.float()  # Ensure Float type
                # -----------------------------------------

                # Model forward pass uses original (non-standardized) data
                # Assumes the probe direction has been unscaled if needed after training
                outputs = model(batch_x_orig)
                loss = loss_fn(outputs, batch_y)
                total_loss += loss.item()

        return total_loss / len(val_loader)


@dataclass
class DirectionalTrainerConfig(BaseTrainerConfig):
    """Configuration for training direction-finding probes."""

    pass  # Uses base config settings


class DirectionalProbeTrainer(BaseProbeTrainer):
    """Trainer for probes that find directions through direct computation (KMeans, PCA, MeanDiff)."""

    def __init__(self, config: DirectionalTrainerConfig):
        super().__init__(config)
        # Ensure self.config has the more specific type
        if not isinstance(config, DirectionalTrainerConfig):
            raise TypeError(
                "DirectionalProbeTrainer requires a DirectionalTrainerConfig"
            )
        self.config: DirectionalTrainerConfig = config

    def prepare_supervised_data(
        self,
        activation_store: ActivationStore,
        position_key: str,
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for directional probe computation.
        Returns two identical DataLoaders yielding (X_train, y, X_orig).
        """
        X_train_all, y_all, X_orig_all = self.prepare_data(
            activation_store, position_key
        )

        # Create a dataset with all data
        # Handle single/multi-dimensional labels
        y_all = y_all if y_all.dim() > 1 else y_all.unsqueeze(1)
        dataset = TensorDataset(X_train_all, y_all, X_orig_all)  # Include X_orig

        # Create loader for all data
        # Shuffle False might be better if order matters, but usually doesn't for these methods
        # Use full dataset length as batch size for single fit step
        batch_size_fit = len(dataset)
        if batch_size_fit == 0:
            print("Warning: Dataset is empty for DirectionalProbeTrainer")
            # Return empty loaders to avoid errors
            return DataLoader([]), DataLoader([])

        all_data_loader = DataLoader(
            dataset,
            batch_size=batch_size_fit,
            shuffle=False,  # No need to shuffle if using full batch
        )

        # Return the same loader twice to maintain the trainer interface
        return all_data_loader, all_data_loader

    def train(
        self,
        model: DirectionalProbe,  # Expect DirectionalProbe instance
        train_loader: DataLoader,
        val_loader: Optional[
            DataLoader
        ] = None,  # val_loader is ignored but kept for API consistency
    ) -> Dict[str, List[float]]:
        """Train method for directional probes.
        1. Accumulate all data (X_train, y, X_orig)
        2. Call probe's fit method with X_train, which returns the initial direction
        3. Unscale the initial direction if standardization was used
        4. Set the probe's final direction buffer
        5. Compute metrics using X_orig and the final probe state
        """
        # Ensure model is a DirectionalProbe instance
        if not isinstance(model, DirectionalProbe):
            raise TypeError(
                "DirectionalProbeTrainer expects model to be an instance of DirectionalProbe"
            )

        # Ensure model is on the correct device
        target_device = torch.device(self.config.device)
        model.to(target_device)

        # Standardization stats are managed by trainer

        # Accumulate all data (loader should have batch_size=len(dataset))
        try:
            x_train_tensor, y_train_tensor, x_orig_tensor = next(iter(train_loader))
        except StopIteration:
            print("Warning: Training loader is empty.")
            return {"train_loss": [], "val_loss": []}  # Return empty history

        x_train_tensor = x_train_tensor.to(target_device)
        y_train_tensor = y_train_tensor.to(target_device)
        x_orig_tensor = x_orig_tensor.to(target_device)  # Keep original data too

        # Fit the probe using potentially standardized data (X_train)
        # The fit method now returns the initial direction (potentially scaled)
        initial_direction = model.fit(x_train_tensor, y_train_tensor)

        # --- Unscale Direction ---
        final_direction = initial_direction  # Default if no standardization
        if self.config.standardize_activations and self.feature_std is not None:
            print("Unscaling directional probe direction...")
            std_dev = self.feature_std.squeeze().to(initial_direction.device)
            # Assume initial_direction is [dim]
            if initial_direction.shape == std_dev.shape:
                final_direction = initial_direction / std_dev
            # Add case for [1, dim] direction and [dim] std_dev
            elif (
                initial_direction.dim() == 2
                and initial_direction.shape[0] == 1
                and std_dev.dim() == 1
            ):
                final_direction = initial_direction / std_dev.unsqueeze(0)
            else:
                print(
                    f"Warning: Shape mismatch during directional probe unscaling. "
                    f"Direction: {initial_direction.shape}, StdDev: {std_dev.shape}. "
                    f"Skipping unscaling."
                )

        # Set the final, unscaled direction in the probe's buffer
        model._set_raw_direction_representation(final_direction)

        # --- Compute Metrics ---
        # Metrics should be computed using the *original* data and the *final* probe state
        history: Dict[str, List[float]] = {
            "train_loss": [],
            # Validation loss is same as train loss since we use all data
            # and val_loader is ignored
            "val_loss": [],
        }

        # Calculate loss using original data and final probe state
        with torch.no_grad():
            # Probe forward uses the final (unscaled) direction set above
            preds = model(x_orig_tensor)
            # Ensure y_train_tensor matches expected shape for loss
            y_target = y_train_tensor.float()
            if preds.dim() == 1 and y_target.dim() == 2 and y_target.shape[1] == 1:
                y_target = y_target.squeeze(1)
            elif preds.shape != y_target.shape:
                # Attempt to align shapes if possible (e.g., [N] vs [N, 1])
                try:
                    y_target = y_target.view_as(preds)
                except RuntimeError:
                    print(
                        f"Warning: Could not align prediction ({preds.shape}) and "
                        f"target ({y_target.shape}) shapes for loss calculation."
                    )
                    # Fallback or skip loss calculation
                    history["train_loss"].append(float("nan"))
                    history["val_loss"].append(float("nan"))
                    return history

            # Use BCEWithLogitsLoss as it's common for binary classification probes
            # Use original y labels
            try:
                # Use the appropriate loss based on the probe type
                if isinstance(model, MultiClassLogisticProbe):
                    loss_fn = nn.CrossEntropyLoss()
                    y_target = y_target.long().squeeze()  # Ensure long and [N] shape
                else:
                    loss_fn = nn.BCEWithLogitsLoss()
                    # y_target is already float

                loss = loss_fn(preds, y_target)
                loss_item = loss.item()
            except Exception as e:
                print(f"Error during loss calculation: {e}")

        history["train_loss"].append(loss_item)
        history["val_loss"].append(loss_item)  # Use same loss for val

        return history
