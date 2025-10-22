#!/usr/bin/env python3

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import pickle
import argparse
import h5py
import hashlib
import threading
import time
import functools

from data.deception_detection.deception_detection.repository import DatasetRepository
from data.deception_detection.deception_detection.utils import preprocess_dialogue
from data.deception_detection.deception_detection.tokenized_data import TokenizedDataset
from data.deception_detection.deception_detection.data import DialogueDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training environment."""
    if not dist.is_initialized():
        # Check if we're in a SLURM environment
        if "SLURM_PROCID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])
            local_rank = int(os.environ["SLURM_LOCALID"])
        # Check if we're using torchrun/torch.distributed.launch
        elif "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            # Not in a distributed environment - return None to disable FSDP
            logger.warning(
                "FSDP requested but not running with torchrun/SLURM. "
                "Falling back to single-GPU mode. "
                "To use FSDP, run with: torchrun --nproc_per_node=NUM_GPUS main_direct_extraction.py --use-fsdp"
            )
            return None, None, None

        # Set device
        torch.cuda.set_device(local_rank)

        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

    return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_available_gpus():
    """Detect available GPUs and their memory."""
    if not torch.cuda.is_available():
        return []

    gpu_info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3  # GB
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        free_memory = total_memory - allocated

        gpu_info.append(
            {
                "id": i,
                "name": props.name,
                "total_gb": total_memory,
                "free_gb": free_memory,
                "allocated_gb": allocated,
            }
        )
    return gpu_info


def log_gpu_info(gpu_info):
    """Log GPU availability and usage."""
    if not gpu_info:
        logger.info("No GPUs available")
        return

    logger.info(f"Found {len(gpu_info)} GPU(s):")
    for gpu in gpu_info:
        logger.info(
            f"  GPU {gpu['id']} ({gpu['name']}): {gpu['free_gb']:.2f}GB free / {gpu['total_gb']:.2f}GB total"
        )


class GPUMonitor:
    """Monitor GPU utilization in background thread."""

    def __init__(self, interval=30):
        self.interval = interval
        self.running = False
        self.thread = None

    def start(self):
        """Start monitoring."""
        if not torch.cuda.is_available():
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started GPU monitoring (reporting every {self.interval}s)")

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            time.sleep(self.interval)
            if self.running:
                gpu_info = get_available_gpus()
                logger.info("=== GPU Utilization Update ===")
                for gpu in gpu_info:
                    utilization_pct = (
                        (gpu["allocated_gb"] / gpu["total_gb"]) * 100
                        if gpu["total_gb"] > 0
                        else 0
                    )
                    logger.info(
                        f"  GPU {gpu['id']}: {gpu['allocated_gb']:.2f}GB / {gpu['total_gb']:.2f}GB "
                        f"({utilization_pct:.1f}% utilized)"
                    )


def calculate_optimal_batch_size(gpu_info, base_batch_size=4):
    """Calculate optimal batch size based on available GPU memory and count."""
    if not gpu_info:
        return base_batch_size

    num_gpus = len(gpu_info)
    min_free_memory = min(gpu["free_gb"] for gpu in gpu_info)

    # Increase batch size based on number of GPUs and available memory
    # With model parallelism, we want to keep the pipeline full
    if num_gpus > 1:
        # Pipeline depth: process multiple batches simultaneously
        optimal_batch_size = base_batch_size * num_gpus
        logger.info(
            f"Using pipeline batch size: {optimal_batch_size} (base={base_batch_size} × {num_gpus} GPUs)"
        )
    else:
        optimal_batch_size = base_batch_size
        logger.info(f"Single GPU mode, using batch size: {optimal_batch_size}")

    return optimal_batch_size


@dataclass
class ActivationCache:
    activations: torch.Tensor
    labels: torch.Tensor
    layer_idx: int
    hook_name: str


class DirectActivationExtractor:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_layers: Optional[int] = None,
        use_fsdp: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.use_fsdp = use_fsdp

        # Detect available GPUs
        self.gpu_info = get_available_gpus()
        log_gpu_info(self.gpu_info)

        if use_fsdp:
            # Setup distributed environment for FSDP
            result = setup_distributed()
            if result[0] is None:
                # Distributed setup failed, fall back to pipeline parallelism
                logger.info("Falling back to pipeline parallelism mode...")
                use_fsdp = False
                self.use_fsdp = False
                self.rank = 0
                self.world_size = 1
                self.local_rank = 0
            else:
                self.rank, self.world_size, self.local_rank = result
                logger.info(f"FSDP: Rank {self.rank}/{self.world_size}, Local Rank {self.local_rank}")

                # Load model on CPU first, then wrap with FSDP
                logger.info(f"Loading model {model_name} with FSDP...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
                model.eval()

                # Wrap model with FSDP
                # Auto-wrap policy based on size (100M parameters per shard)
                auto_wrap_policy = functools.partial(
                    size_based_auto_wrap_policy, min_num_params=100_000_000
                )

                self.model = FSDP(
                    model,
                    auto_wrap_policy=auto_wrap_policy,
                    device_id=torch.cuda.current_device(),
                    mixed_precision=None,  # We'll use fp16 from model
                    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                    cpu_offload=None,  # Keep everything on GPU for inference
                )

                logger.info(f"Model wrapped with FSDP on rank {self.rank}")

        # Load model with pipeline parallelism if not using FSDP
        if not self.use_fsdp:
            # Original pipeline parallelism approach without quantization
            # Set rank to 0 for non-distributed mode
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

            logger.info(f"Loading model {model_name} with pipeline parallelism...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            self.model.eval()

            # Log which layers are on which devices
            if hasattr(self.model, "hf_device_map"):
                logger.info("Model layer distribution across GPUs:")
                device_layers = {}
                for layer_name, device in self.model.hf_device_map.items():
                    if device not in device_layers:
                        device_layers[device] = []
                    device_layers[device].append(layer_name)
                for device, layers in sorted(device_layers.items()):
                    logger.info(f"  {device}: {len(layers)} components")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_config = AutoConfig.from_pretrained(model_name)
        self.num_layers = model_config.num_hidden_layers
        self.hidden_size = model_config.hidden_size
        logger.info(
            f"Model loaded: {self.num_layers} layers, hidden size {self.hidden_size}"
        )

        self.max_layers = max_layers
        self.num_gpus = len(self.gpu_info)

    def select_layers(self) -> List[int]:
        if self.max_layers is None:
            return list(range(self.num_layers))
        else:
            layer_indices = [0, self.num_layers - 1]
            if self.num_layers > 2:
                middle_layers = np.linspace(
                    1,
                    self.num_layers - 2,
                    min(self.max_layers - 2, self.num_layers - 2),
                    dtype=int,
                )
                layer_indices.extend(middle_layers.tolist())
            return sorted(list(set(layer_indices)))

    def extract_activations_all_layers(
        self,
        dataset: DialogueDataset,
        layer_indices: List[int],
        cache_file: str,
        batch_size: int = 4,
        max_length: int = 512,
        pipeline_depth: Optional[int] = None,
    ) -> None:
        """Extract activations from all specified layers in a single pass and save incrementally to HDF5.

        Args:
            dataset: The dataset to extract activations from
            layer_indices: List of layer indices to extract
            cache_file: Path to HDF5 cache file
            batch_size: Batch size for processing
            max_length: Max sequence length
            pipeline_depth: Number of batches to process in parallel (default: num_gpus for pipeline parallelism)
        """
        # Use their TokenizedDataset to get proper detection masks
        tokenized = TokenizedDataset.from_dataset(
            dataset,
            self.tokenizer,
            max_length=max_length,
        )

        # Note: pipeline_depth parameter is ignored - we process batches sequentially
        # FSDP handles data parallelism, pipeline parallelism handles layer distribution

        # Start GPU monitoring
        gpu_monitor = GPUMonitor(interval=30)
        gpu_monitor.start()

        # For FSDP, only rank 0 should write to cache
        should_write = not self.use_fsdp or (self.use_fsdp and self.rank == 0)

        # Initialize HDF5 file with datasets for each layer (only rank 0)
        if should_write:
            with h5py.File(cache_file, "w") as f:
                # Store metadata
                f.attrs["num_layers"] = len(layer_indices)
                f.attrs["layer_indices"] = layer_indices
                f.attrs["hidden_size"] = self.hidden_size

                # Create extensible datasets for each layer
                for layer_idx in layer_indices:
                    f.create_dataset(
                        f"layer_{layer_idx}",
                        shape=(0, self.hidden_size),
                        maxshape=(None, self.hidden_size),
                        dtype="float32",
                        chunks=True,
                        compression="gzip",
                        compression_opts=4,
                    )

                # Create dataset for labels (shared across all layers)
                f.create_dataset(
                    "labels", shape=(0,), maxshape=(None,), dtype="int32", chunks=True
                )

        # Synchronize all processes before starting extraction
        if self.use_fsdp:
            dist.barrier()

        # Process batches sequentially
        # With FSDP, all ranks process but only rank 0 writes
        total_batches = (len(tokenized) + batch_size - 1) // batch_size

        for i in range(0, len(tokenized), batch_size):
            batch_idx = i // batch_size
            batch_tokenized = tokenized[i : i + batch_size]
            batch_labels = dataset.labels[i : i + batch_size]

            # Process batch
            with torch.no_grad():
                # Direct forward pass with output_hidden_states=True
                if self.use_fsdp:
                    # For FSDP, move to current device
                    device = torch.cuda.current_device()
                else:
                    # For pipeline parallelism, use self.device
                    device = self.device

                outputs = self.model(
                    input_ids=batch_tokenized.tokens.to(device),
                    attention_mask=batch_tokenized.attention_mask.to(device),
                    output_hidden_states=True,
                )

                # hidden_states is a tuple of (num_layers + 1) tensors
                # Each tensor has shape [batch, seq, hidden_dim]
                # Index 0 is embeddings, index 1 is layer 0, etc.
                all_hidden_states = outputs.hidden_states

                # Extract activations at detection mask positions for each layer
                detection_mask = batch_tokenized.detection_mask
                assert detection_mask is not None, "Detection mask is required"

                # Collect batch activations and labels
                batch_activations = {layer_idx: [] for layer_idx in layer_indices}
                batch_labels_collected = []

                for j in range(len(batch_tokenized)):
                    mask_positions = detection_mask[j]
                    label = 1 if batch_labels[j].value == "deceptive" else 0
                    num_tokens = mask_positions.sum().item()

                    # Extract for each layer
                    for layer_idx in layer_indices:
                        hidden_states = all_hidden_states[
                            layer_idx + 1
                        ].cpu()  # +1 because index 0 is embeddings
                        dialogue_acts = hidden_states[j][
                            mask_positions
                        ]  # [num_detect_tokens, hidden_dim]
                        batch_activations[layer_idx].append(dialogue_acts)

                    # Labels are same for all layers
                    batch_labels_collected.extend([label] * num_tokens)

            # Write batch results to HDF5 file (only rank 0 for FSDP)
            if should_write:
                with h5py.File(cache_file, "a") as f:
                    for layer_idx in layer_indices:
                        # Concatenate batch activations for this layer
                        layer_acts = torch.cat(batch_activations[layer_idx], dim=0).numpy()

                        # Get current dataset
                        dset = f[f"layer_{layer_idx}"]
                        current_size = dset.shape[0]
                        new_size = current_size + layer_acts.shape[0]

                        # Resize and append
                        dset.resize(new_size, axis=0)
                        dset[current_size:new_size] = layer_acts

                    # Append labels (only once since shared)
                    labels_dset = f["labels"]
                    current_size = labels_dset.shape[0]
                    new_size = current_size + len(batch_labels_collected)
                    labels_dset.resize(new_size, axis=0)
                    labels_dset[current_size:new_size] = batch_labels_collected

                if self.rank == 0 or not self.use_fsdp:
                    logger.info(f"Processed batch {batch_idx + 1}/{total_batches}")

            # Synchronize after each batch for FSDP
            if self.use_fsdp:
                dist.barrier()

        # Stop GPU monitoring
        gpu_monitor.stop()

    def load_activations_from_cache(
        self,
        cache_file: str,
        layer_idx: int,
    ) -> ActivationCache:
        """Load activations for a specific layer from HDF5 cache.

        Args:
            cache_file: Path to HDF5 cache file
            layer_idx: Layer index to load

        Returns:
            ActivationCache with activations and labels for the specified layer
        """
        with h5py.File(cache_file, "r") as f:
            activations = torch.tensor(f[f"layer_{layer_idx}"][:], dtype=torch.float32)
            labels = torch.tensor(f["labels"][:], dtype=torch.long)

        return ActivationCache(
            activations=activations,
            labels=labels,
            layer_idx=layer_idx,
            hook_name=f"layer_{layer_idx}",
        )


def train_probe(
    activations: torch.Tensor,
    labels: torch.Tensor,
    train_indices: List[int],
    val_indices: List[int],
    max_iter: int = 1000,
    break_probe: bool = False,
) -> Tuple[LogisticRegression, StandardScaler, Dict[str, float]]:
    if break_probe:
        labels = torch.randint(0, 2, size=labels.shape)

    X_train = activations[train_indices].cpu().numpy()
    y_train = labels[train_indices].cpu().numpy()
    X_val = activations[val_indices].cpu().numpy()
    y_val = labels[val_indices].cpu().numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    class_weights = {
        0: len(y_train) / (2 * (y_train == 0).sum()),
        1: len(y_train) / (2 * (y_train == 1).sum()),
    }

    probe = LogisticRegression(
        max_iter=max_iter, random_state=42, solver="lbfgs", class_weight=class_weights
    )

    logger.info("Training sklearn LogisticRegression...")
    probe.fit(X_train_scaled, y_train)

    y_val_pred = probe.predict(X_val_scaled)
    y_val_proba = probe.predict_proba(X_val_scaled)[:, 1]

    accuracy = (y_val_pred == y_val).mean()
    tp = ((y_val_pred == 1) & (y_val == 1)).sum()
    fp = ((y_val_pred == 1) & (y_val == 0)).sum()
    fn = ((y_val_pred == 0) & (y_val == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    auroc = roc_auc_score(y_val, y_val_proba)
    confidence = y_val_proba.mean()
    deceptive_ratio = y_val_pred.sum() / len(y_val_pred)

    val_metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "confidence": float(confidence),
        "num_samples": len(y_val),
        "deceptive_ratio": float(deceptive_ratio),
    }

    logger.info(
        f"Training complete: "
        f"Val Accuracy={accuracy:.3f}, "
        f"Precision={precision:.3f}, "
        f"Recall={recall:.3f}, "
        f"F1={f1:.3f}, "
        f"AUROC={auroc:.3f}"
    )

    return probe, scaler, val_metrics


def test_probe(
    probe: LogisticRegression,
    scaler: StandardScaler,
    activations: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    X_test = activations.cpu().numpy()
    y_test = labels.cpu().numpy()

    X_test_scaled = scaler.transform(X_test)

    y_pred = probe.predict(X_test_scaled)
    y_proba = probe.predict_proba(X_test_scaled)[:, 1]

    accuracy = (y_pred == y_test).mean()
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    auroc = roc_auc_score(y_test, y_proba)
    confidence = y_proba.mean()
    deceptive_ratio = y_pred.sum() / len(y_pred)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "confidence": float(confidence),
        "num_samples": len(y_test),
        "deceptive_ratio": float(deceptive_ratio),
    }


def load_dataset(dataset_name: str = "roleplaying") -> DialogueDataset:
    """Load a DialogueDataset directly, reusing their data loading code."""
    repo = DatasetRepository()

    dataset_mapping = {
        "repe_honesty": "repe_honesty__you_are_fact_sys",
        "roleplaying": "roleplaying__offpolicy_train",
        "roleplaying_plain": "roleplaying__plain",
        "ai_audit": "ai_audit__no_reasoning",
        "werewolf": "werewolf__paired",
        "ai_liar": "ai_liar__original_without_answers",
        "ai_liar_with": "ai_liar__original_with_answers",
    }

    if dataset_name not in dataset_mapping:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. Options: {list(dataset_mapping.keys())}"
        )

    dataset_id = dataset_mapping[dataset_name]
    model_name = "prewritten"
    dataset = repo.get(dataset_id, model=model_name)
    logger.info(f"Successfully loaded dataset '{dataset_id}'.")
    logger.info(f"Dataset size: {len(dataset.dialogues)} dialogues")

    return dataset


def combine_datasets(
    datasets: List[DialogueDataset],
    dataset_names: List[str],
    target_samples_per_dataset: Optional[int] = None,
) -> DialogueDataset:
    """Combine multiple datasets, optionally balancing them."""
    import random
    from copy import deepcopy

    random.seed(42)
    np.random.seed(42)

    all_dialogues = []
    all_labels = []

    for dataset, name in zip(datasets, dataset_names):
        dialogues = dataset.dialogues
        labels = dataset.labels

        if target_samples_per_dataset is None:
            # Use all data
            all_dialogues.extend(dialogues)
            all_labels.extend(labels)
            logger.info(f"{name}: using all {len(dialogues)} samples")
        elif len(dialogues) >= target_samples_per_dataset:
            # Downsample
            indices = random.sample(range(len(dialogues)), target_samples_per_dataset)
            all_dialogues.extend([dialogues[i] for i in indices])
            all_labels.extend([labels[i] for i in indices])
            logger.info(
                f"{name}: downsampled from {len(dialogues)} to {target_samples_per_dataset} samples"
            )
        else:
            # Oversample
            all_dialogues.extend(dialogues)
            all_labels.extend(labels)
            remaining = target_samples_per_dataset - len(dialogues)
            for _ in range(remaining):
                idx = random.randint(0, len(dialogues) - 1)
                all_dialogues.append(dialogues[idx])
                all_labels.append(labels[idx])
            logger.info(
                f"{name}: oversampled from {len(dialogues)} to {target_samples_per_dataset} samples"
            )

    # Create a combined dataset using the first dataset as template
    combined = deepcopy(datasets[0])
    combined.dialogues = all_dialogues
    combined.labels = all_labels

    return combined


def get_cache_filename(
    model_name: str,
    dataset_names: List[str],
    split: str,
    max_length: int,
    target_samples: Optional[int] = None,
) -> str:
    """Generate a unique cache filename based on parameters."""
    # Create a hash of the key parameters
    cache_key = f"{model_name}_{'+'.join(sorted(dataset_names))}_{split}_{max_length}_{target_samples}"
    hash_suffix = hashlib.md5(cache_key.encode()).hexdigest()[:8]

    cache_dir = "./activation_cache"
    os.makedirs(cache_dir, exist_ok=True)

    return os.path.join(cache_dir, f"activations_{hash_suffix}.h5")


def save_results_to_csv(results: List[Dict], filename: str = "results.csv"):
    if not results:
        logger.warning("No results to save")
        return

    logger.info(f"Saving results to {filename}...")
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    logger.info(f"Results saved to {filename} with {len(results)} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Train and test deception detection probes on language models (direct extraction)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name from HuggingFace (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--train-datasets",
        type=str,
        nargs="+",
        default=["repe_honesty"],
        help="Training dataset names (space-separated)",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default="roleplaying",
        help="Test dataset name (default: roleplaying)",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=100,
        help="Target samples per dataset for balancing (default: 100)",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=4,
        help="Maximum number of layers to probe (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for activation extraction (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_direct.csv",
        help="Output CSV file path (default: results_direct.csv)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization (default: 512)",
    )
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        default=None,
        help="Number of batches to process in parallel (default: auto-detect based on GPU count)",
    )
    parser.add_argument(
        "--reset-cache",
        type=bool,
        default=True,
        help="Whether to reset activation cache (default: True)",
    )
    parser.add_argument(
        "--use-fsdp",
        action="store_true",
        help="Use FSDP for multi-GPU distribution (requires no quantization)",
    )

    args = parser.parse_args()

    logger.info(
        "SPAR - Training on Multiple Datasets, Testing on Roleplaying (Direct Extraction)"
    )
    logger.info(f"Model: {args.model}")
    logger.info(f"Train datasets: {args.train_datasets}")
    logger.info(f"Test dataset: {args.test_dataset}")

    all_results = []

    model_name = args.model
    train_dataset_names = args.train_datasets
    test_dataset_name = args.test_dataset
    target_samples_per_dataset = args.target_samples

    try:
        logger.info("--- Initializing direct extraction model ---")
        logger.info("Detecting available GPUs...")

        # Get GPU info before model loading
        initial_gpu_info = get_available_gpus()
        log_gpu_info(initial_gpu_info)

        # Initialize extractor (will load model across GPUs)
        extractor = DirectActivationExtractor(
            model_name=model_name, max_layers=args.max_layers, use_fsdp=args.use_fsdp
        )

        # Calculate optimal batch size based on GPU count
        optimal_batch_size = calculate_optimal_batch_size(
            extractor.gpu_info, base_batch_size=args.batch_size
        )

        # Override batch size if auto-calculated
        if len(extractor.gpu_info) > 1:
            logger.info(
                f"Multi-GPU detected: adjusting batch processing for pipeline parallelism"
            )
            # We keep the base batch size but increase pipeline depth
            actual_batch_size = args.batch_size
        else:
            actual_batch_size = args.batch_size

        logger.info(f"--- Loading training datasets: {train_dataset_names} ---")

        # Load all datasets
        train_datasets = []
        for dataset_name in train_dataset_names:
            dataset = load_dataset(dataset_name)
            train_datasets.append(dataset)

        logger.info("--- Combining/balancing datasets ---")
        combined_dataset = combine_datasets(
            train_datasets,
            train_dataset_names,
            target_samples_per_dataset if len(train_dataset_names) > 1 else None,
        )

        logger.info(f"Total training dialogues: {len(combined_dataset.dialogues)}")
        deceptive_count = sum(
            1 for l in combined_dataset.labels if l.value == "deceptive"
        )
        logger.info(
            f"Label distribution: {deceptive_count} deceptive, {len(combined_dataset.labels) - deceptive_count} honest"
        )

        # Split dataset at dialogue level
        logger.info("--- Creating train/validation splits ---")
        from copy import deepcopy

        num_dialogues = len(combined_dataset.dialogues)
        num_val = int(num_dialogues * 0.2)

        # Sequential split (matching the original implementation)
        train_dataset = deepcopy(combined_dataset)
        train_dataset.dialogues = (
            combined_dataset.dialogues[:-num_val]
            if num_val > 0
            else combined_dataset.dialogues
        )
        train_dataset.labels = (
            combined_dataset.labels[:-num_val]
            if num_val > 0
            else combined_dataset.labels
        )

        val_dataset = deepcopy(combined_dataset)
        val_dataset.dialogues = (
            combined_dataset.dialogues[-num_val:] if num_val > 0 else []
        )
        val_dataset.labels = combined_dataset.labels[-num_val:] if num_val > 0 else []

        logger.info(
            f"Dataset split: {len(train_dataset.dialogues)} train, {len(val_dataset.dialogues)} validation dialogues"
        )

        logger.info("TRAINING PROBES ON COMBINED DATASETS")

        layer_indices = extractor.select_layers()
        logger.info(f"Selected layers: {layer_indices}")

        # Generate cache filenames
        train_cache_file = get_cache_filename(
            model_name,
            train_dataset_names,
            "train",
            args.max_length,
            target_samples_per_dataset,
        )
        val_cache_file = get_cache_filename(
            model_name,
            train_dataset_names,
            "val",
            args.max_length,
            target_samples_per_dataset,
        )

        if os.path.exists(train_cache_file) and args.reset_cache:
            os.remove(train_cache_file)
        if os.path.exists(val_cache_file) and args.reset_cache:
            os.remove(val_cache_file)
        # Extract and cache activations if not already cached
        if not os.path.exists(train_cache_file):
            logger.info(
                "Extracting train activations for all layers (this will be cached)..."
            )
            logger.info(f"GPU status before extraction:")
            log_gpu_info(get_available_gpus())

            extractor.extract_activations_all_layers(
                train_dataset,
                layer_indices,
                train_cache_file,
                batch_size=actual_batch_size,
                max_length=args.max_length,
                pipeline_depth=args.pipeline_depth,
            )

            logger.info(f"GPU status after extraction:")
            log_gpu_info(get_available_gpus())
            logger.info(f"Train activations cached to: {train_cache_file}")
        else:
            logger.info(f"Using cached train activations from: {train_cache_file}")

        if not os.path.exists(val_cache_file):
            logger.info(
                "Extracting validation activations for all layers (this will be cached)..."
            )
            logger.info(f"GPU status before extraction:")
            log_gpu_info(get_available_gpus())

            extractor.extract_activations_all_layers(
                val_dataset,
                layer_indices,
                val_cache_file,
                batch_size=actual_batch_size,
                max_length=args.max_length,
                pipeline_depth=args.pipeline_depth,
            )

            logger.info(f"GPU status after extraction:")
            log_gpu_info(get_available_gpus())
            logger.info(f"Validation activations cached to: {val_cache_file}")
        else:
            logger.info(f"Using cached validation activations from: {val_cache_file}")

        all_probe_results = {}
        save_dir = "./probe_checkpoints_direct"
        os.makedirs(save_dir, exist_ok=True)

        for layer_idx in layer_indices:
            logger.info(f"\n{'='*40}")
            logger.info(f"TRAINING PROBE FOR LAYER {layer_idx}")
            logger.info(f"{'='*40}")

            logger.info(f"Loading activations from cache for layer {layer_idx}...")
            # Load from cache
            train_cache = extractor.load_activations_from_cache(
                train_cache_file, layer_idx
            )
            val_cache = extractor.load_activations_from_cache(val_cache_file, layer_idx)

            logger.info(
                f"Train tokens: {len(train_cache.activations)}, Val tokens: {len(val_cache.activations)}"
            )

            # Combine and create indices for the probe training function
            all_acts = torch.cat(
                [train_cache.activations, val_cache.activations], dim=0
            )
            all_labels = torch.cat([train_cache.labels, val_cache.labels], dim=0)
            train_indices = list(range(len(train_cache.activations)))
            val_indices = list(range(len(train_cache.activations), len(all_acts)))

            logger.info("Training probe...")
            probe, scaler, val_metrics = train_probe(
                all_acts,
                all_labels,
                train_indices,
                val_indices,
                max_iter=1000,
            )

            probe_path = os.path.join(save_dir, f"probe_layer_{layer_idx}.pkl")
            with open(probe_path, "wb") as f:
                pickle.dump({"probe": probe, "scaler": scaler}, f)

            all_probe_results[layer_idx] = {
                "probe": probe,
                "scaler": scaler,
                "val_metrics": val_metrics,
            }

            all_results.append(
                {
                    "train_datasets": "+".join(train_dataset_names),
                    "test_dataset": "validation",
                    "layer": layer_idx,
                    "probe_type": "logistic",
                    "accuracy": val_metrics["accuracy"],
                    "precision": val_metrics["precision"],
                    "recall": val_metrics["recall"],
                    "f1": val_metrics["f1"],
                    "auroc": val_metrics["auroc"],
                    "confidence": val_metrics["confidence"],
                    "num_samples": val_metrics["num_samples"],
                }
            )

        logger.info(f"\n--- Loading {test_dataset_name} dataset for testing ---")
        test_dataset = load_dataset(test_dataset_name)

        logger.info(f"TESTING PROBES ON {test_dataset_name.upper()} DATASET")

        # Generate cache filename for test set
        test_cache_file = get_cache_filename(
            model_name, [test_dataset_name], "test", args.max_length, None
        )

        # Extract and cache test activations if not already cached
        if not os.path.exists(test_cache_file):
            logger.info(
                "Extracting test activations for all layers (this will be cached)..."
            )
            logger.info(f"GPU status before extraction:")
            log_gpu_info(get_available_gpus())

            extractor.extract_activations_all_layers(
                test_dataset,
                layer_indices,
                test_cache_file,
                batch_size=actual_batch_size,
                max_length=args.max_length,
                pipeline_depth=args.pipeline_depth,
            )

            logger.info(f"GPU status after extraction:")
            log_gpu_info(get_available_gpus())
            logger.info(f"Test activations cached to: {test_cache_file}")
        else:
            logger.info(f"Using cached test activations from: {test_cache_file}")

        for layer_idx in layer_indices:
            logger.info(
                f"\nLoading test activations from cache for layer {layer_idx}..."
            )
            test_cache = extractor.load_activations_from_cache(
                test_cache_file, layer_idx
            )

            probe = all_probe_results[layer_idx]["probe"]
            scaler = all_probe_results[layer_idx]["scaler"]
            test_metrics = test_probe(
                probe, scaler, test_cache.activations, test_cache.labels
            )

            logger.info(
                f"Layer {layer_idx} Test Results: "
                f"Accuracy={test_metrics['accuracy']:.3f}, "
                f"F1={test_metrics['f1']:.3f}, "
                f"AUROC={test_metrics['auroc']:.3f}, "
                f"Deception Ratio={test_metrics['deceptive_ratio']:.3f}"
            )

            all_results.append(
                {
                    "train_datasets": "+".join(train_dataset_names),
                    "test_dataset": test_dataset_name,
                    "layer": layer_idx,
                    "probe_type": "logistic",
                    "accuracy": test_metrics["accuracy"],
                    "precision": test_metrics["precision"],
                    "recall": test_metrics["recall"],
                    "f1": test_metrics["f1"],
                    "auroc": test_metrics["auroc"],
                    "confidence": test_metrics["confidence"],
                    "num_samples": test_metrics["num_samples"],
                }
            )

        save_results_to_csv(all_results, args.output)

        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 60)
        logger.info(
            f"Training datasets: {', '.join(train_dataset_names)} ({len(train_dataset.dialogues)} train + {len(val_dataset.dialogues)} val dialogues)"
        )
        logger.info(
            f"Test dataset: {test_dataset_name} ({len(test_dataset.dialogues)} dialogues)"
        )
        logger.info(f"Results saved to: {args.output} ({len(all_results)} rows)")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(
            f"An unexpected error occurred in the main process: {e}", exc_info=True
        )
    finally:
        # Cleanup distributed if using FSDP
        if args.use_fsdp:
            cleanup_distributed()


if __name__ == "__main__":
    main()
