#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import Accelerator, PartialState, DeepSpeedPlugin
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

from data.deception_detection.deception_detection.repository import DatasetRepository
from data.deception_detection.deception_detection.utils import preprocess_dialogue
from data.deception_detection.deception_detection.tokenized_data import TokenizedDataset
from data.deception_detection.deception_detection.data import DialogueDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        use_deepspeed: bool = True,
        deepspeed_config: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.use_deepspeed = use_deepspeed

        # Initialize accelerator with DeepSpeed if requested
        if use_deepspeed:
            # Create default DeepSpeed config if not provided
            if deepspeed_config is None:
                default_ds_config = {
                    "train_micro_batch_size_per_gpu": 1,
                    "gradient_accumulation_steps": 1,
                    "zero_optimization": {
                        "stage": 3,
                        "offload_param": {"device": "cpu"},
                        "offload_optimizer": {"device": "cpu"},
                    },
                    "fp16": {"enabled": torch.cuda.is_available()},
                    "bf16": {"enabled": False},
                }
                ds_plugin = DeepSpeedPlugin(hf_ds_config=default_ds_config)
                self.accelerator = Accelerator(deepspeed_plugin=ds_plugin)
            else:
                # Load config from file
                import json
                with open(deepspeed_config, 'r') as f:
                    ds_config = json.load(f)
                ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)
                self.accelerator = Accelerator(deepspeed_plugin=ds_plugin)

            logger.info(f"DeepSpeed + Accelerate initialized")
            logger.info(f"Device: {self.accelerator.device}, Num processes: {self.accelerator.num_processes}")
            logger.info(f"Distributed type: {self.accelerator.distributed_type}")
        else:
            self.accelerator = Accelerator()
            logger.info(f"Accelerate initialized (no DeepSpeed)")
            logger.info(f"Device: {self.accelerator.device}")

        use_8bit = torch.cuda.is_available() and not use_deepspeed  # Don't use 8bit with DeepSpeed
        logger.info(f"Loading model {model_name} (8bit={use_8bit}, deepspeed={use_deepspeed})...")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if not use_deepspeed else None,  # DeepSpeed handles device placement
            load_in_8bit=use_8bit,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # Prepare model with accelerator (handles DeepSpeed integration)
        if use_deepspeed:
            self.model = self.accelerator.prepare(self.model)

        self.model.eval()

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
    ) -> None:
        """Extract activations from all specified layers in a single pass and save incrementally to HDF5.

        Args:
            dataset: The dataset to extract activations from
            layer_indices: List of layer indices to extract
            cache_file: Path to HDF5 cache file
            batch_size: Batch size for processing
            max_length: Max sequence length
        """
        # Use their TokenizedDataset to get proper detection masks
        tokenized = TokenizedDataset.from_dataset(
            dataset,
            self.tokenizer,
            max_length=max_length,
        )

        # Initialize HDF5 file with datasets for each layer
        with h5py.File(cache_file, 'w') as f:
            # Store metadata
            f.attrs['num_layers'] = len(layer_indices)
            f.attrs['layer_indices'] = layer_indices
            f.attrs['hidden_size'] = self.hidden_size

            # Create extensible datasets for each layer
            for layer_idx in layer_indices:
                f.create_dataset(
                    f'layer_{layer_idx}',
                    shape=(0, self.hidden_size),
                    maxshape=(None, self.hidden_size),
                    dtype='float32',
                    chunks=True,
                    compression='gzip',
                    compression_opts=4
                )

            # Create dataset for labels (shared across all layers)
            f.create_dataset(
                'labels',
                shape=(0,),
                maxshape=(None,),
                dtype='int32',
                chunks=True
            )

        # Process batches and incrementally append to HDF5
        for i in range(0, len(tokenized), batch_size):
            batch_tokenized = tokenized[i : i + batch_size]
            batch_labels = dataset.labels[i : i + batch_size]

            with torch.no_grad():
                # Direct forward pass with output_hidden_states=True
                target_device = self.accelerator.device if hasattr(self, 'accelerator') else self.device
                outputs = self.model(
                    input_ids=batch_tokenized.tokens.to(target_device),
                    attention_mask=batch_tokenized.attention_mask.to(target_device),
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
                        hidden_states = all_hidden_states[layer_idx + 1].cpu()  # +1 because index 0 is embeddings
                        dialogue_acts = hidden_states[j][mask_positions]  # [num_detect_tokens, hidden_dim]
                        batch_activations[layer_idx].append(dialogue_acts)

                    # Labels are same for all layers
                    batch_labels_collected.extend([label] * num_tokens)

                # Append to HDF5 file
                with h5py.File(cache_file, 'a') as f:
                    for layer_idx in layer_indices:
                        # Concatenate batch activations for this layer
                        layer_acts = torch.cat(batch_activations[layer_idx], dim=0).numpy()

                        # Get current dataset
                        dset = f[f'layer_{layer_idx}']
                        current_size = dset.shape[0]
                        new_size = current_size + layer_acts.shape[0]

                        # Resize and append
                        dset.resize(new_size, axis=0)
                        dset[current_size:new_size] = layer_acts

                    # Append labels (only once since shared)
                    labels_dset = f['labels']
                    current_size = labels_dset.shape[0]
                    new_size = current_size + len(batch_labels_collected)
                    labels_dset.resize(new_size, axis=0)
                    labels_dset[current_size:new_size] = batch_labels_collected

            logger.info(f"Processed batch {i//batch_size + 1}/{(len(tokenized) + batch_size - 1)//batch_size}")

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
        with h5py.File(cache_file, 'r') as f:
            activations = torch.tensor(f[f'layer_{layer_idx}'][:], dtype=torch.float32)
            labels = torch.tensor(f['labels'][:], dtype=torch.long)

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
        "--no-deepspeed",
        action="store_true",
        help="Disable DeepSpeed (default: use DeepSpeed)",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON file (optional, uses default ZeRO-3 config if not provided)",
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
        extractor = DirectActivationExtractor(
            model_name=model_name,
            max_layers=args.max_layers,
            use_deepspeed=not args.no_deepspeed,
            deepspeed_config=args.deepspeed_config,
        )

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
            target_samples_per_dataset if len(train_dataset_names) > 1 else None
        )

        logger.info(f"Total training dialogues: {len(combined_dataset.dialogues)}")
        deceptive_count = sum(1 for l in combined_dataset.labels if l.value == "deceptive")
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
        train_dataset.dialogues = combined_dataset.dialogues[:-num_val] if num_val > 0 else combined_dataset.dialogues
        train_dataset.labels = combined_dataset.labels[:-num_val] if num_val > 0 else combined_dataset.labels

        val_dataset = deepcopy(combined_dataset)
        val_dataset.dialogues = combined_dataset.dialogues[-num_val:] if num_val > 0 else []
        val_dataset.labels = combined_dataset.labels[-num_val:] if num_val > 0 else []

        logger.info(
            f"Dataset split: {len(train_dataset.dialogues)} train, {len(val_dataset.dialogues)} validation dialogues"
        )

        logger.info("TRAINING PROBES ON COMBINED DATASETS")

        layer_indices = extractor.select_layers()
        logger.info(f"Selected layers: {layer_indices}")

        # Generate cache filenames
        train_cache_file = get_cache_filename(
            model_name, train_dataset_names, "train", args.max_length, target_samples_per_dataset
        )
        val_cache_file = get_cache_filename(
            model_name, train_dataset_names, "val", args.max_length, target_samples_per_dataset
        )

        # Extract and cache activations if not already cached
        if not os.path.exists(train_cache_file):
            logger.info("Extracting train activations for all layers (this will be cached)...")
            extractor.extract_activations_all_layers(
                train_dataset,
                layer_indices,
                train_cache_file,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            logger.info(f"Train activations cached to: {train_cache_file}")
        else:
            logger.info(f"Using cached train activations from: {train_cache_file}")

        if not os.path.exists(val_cache_file):
            logger.info("Extracting validation activations for all layers (this will be cached)...")
            extractor.extract_activations_all_layers(
                val_dataset,
                layer_indices,
                val_cache_file,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
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
            train_cache = extractor.load_activations_from_cache(train_cache_file, layer_idx)
            val_cache = extractor.load_activations_from_cache(val_cache_file, layer_idx)

            logger.info(f"Train tokens: {len(train_cache.activations)}, Val tokens: {len(val_cache.activations)}")

            # Combine and create indices for the probe training function
            all_acts = torch.cat([train_cache.activations, val_cache.activations], dim=0)
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
            logger.info("Extracting test activations for all layers (this will be cached)...")
            extractor.extract_activations_all_layers(
                test_dataset,
                layer_indices,
                test_cache_file,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            logger.info(f"Test activations cached to: {test_cache_file}")
        else:
            logger.info(f"Using cached test activations from: {test_cache_file}")

        for layer_idx in layer_indices:
            logger.info(f"\nLoading test activations from cache for layer {layer_idx}...")
            test_cache = extractor.load_activations_from_cache(test_cache_file, layer_idx)

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
        logger.info(f"Test dataset: {test_dataset_name} ({len(test_dataset.dialogues)} dialogues)")
        logger.info(f"Results saved to: {args.output} ({len(all_results)} rows)")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(
            f"An unexpected error occurred in the main process: {e}", exc_info=True
        )


if __name__ == "__main__":
    main()
