#!/usr/bin/env python3
"""
Simple script to load a dataset using the deception-detection repository,
and then use probity to extract activations and train truthfulness probes.
"""

import torch
from transformers import AutoTokenizer
from probity.datasets.base import ProbingExample, ProbingDataset
from probity.datasets.tokenized import TokenizedProbingDataset, TokenPositions
from probity.collection.collectors import (
    TransformerLensCollector,
    TransformerLensConfig,
)
from probity.probes import (
    LinearProbe,
    LinearProbeConfig,
    LogisticProbe,
    LogisticProbeConfig,
    KMeansProbe,
    KMeansProbeConfig,
    PCAProbe,
    PCAProbeConfig,
    MeanDifferenceProbe,
    MeanDiffProbeConfig,
)
from probity.training.trainer import (
    SupervisedProbeTrainer,
    SupervisedTrainerConfig,
    DirectionalProbeTrainer,
    DirectionalTrainerConfig,
)
from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig
import os
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import numpy as np
import pandas as pd
import csv
from transformers import AutoConfig
from sklearn.metrics import roc_auc_score

# Import the new dataset repository
from data.deception_detection.deception_detection.repository import DatasetRepository
from data.deception_detection.deception_detection.types import dialogue_to_string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def validate_probe_on_split(
    probe, activation_store, split_indices: List[int], position_key: str = "last"
) -> Dict[str, float]:
    """Evaluate probe on a specific data split with detailed metrics."""
    X, y = activation_store.get_probe_data(position_key)

    # Select only the specified indices
    X_split = X[split_indices].to(probe.config.device)
    y_split = y[split_indices].to(probe.config.device)

    with torch.no_grad():
        predictions = probe(X_split)

        # Convert predictions to binary for accuracy calculation
        if predictions.shape[1] == 1:  # Binary classification
            binary_preds = (torch.sigmoid(predictions) > 0.5).float()
            accuracy = (binary_preds.squeeze() == y_split.float()).float().mean()

            # Calculate AUROC using predicted probabilities
            probabilities = torch.sigmoid(predictions).squeeze()
            auroc = roc_auc_score(y_split.cpu().numpy(), probabilities.cpu().numpy())

            # Calculate precision, recall, F1
            tp = ((binary_preds.squeeze() == 1) & (y_split.float() == 1)).sum().float()
            fp = ((binary_preds.squeeze() == 1) & (y_split.float() == 0)).sum().float()
            fn = ((binary_preds.squeeze() == 0) & (y_split.float() == 1)).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        else:  # Multi-class
            binary_preds = torch.argmax(predictions, dim=1)
            accuracy = (binary_preds == y_split.long().squeeze()).float().mean()
            precision = recall = f1 = accuracy  # Simplified for multi-class

            # For multi-class, calculate macro-averaged AUROC
            try:
                probabilities = torch.softmax(predictions, dim=1)
                auroc = roc_auc_score(
                    y_split.cpu().numpy(),
                    probabilities.cpu().numpy(),
                    multi_class='ovr',
                    average='macro'
                )
            except ValueError:
                # Fallback if there's only one class in the split
                auroc = 0.5

        # Calculate mean prediction confidence
        if predictions.shape[1] == 1:
            confidence = torch.sigmoid(predictions).mean()
        else:
            confidence = torch.softmax(predictions, dim=1).max(dim=1)[0].mean()

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "auroc": auroc if isinstance(auroc, float) else auroc.item(),
        "confidence": confidence.item(),
        "num_samples": len(y_split),
    }


def create_validation_splits(
    dataset_size: int, train_ratio: float = 0.8
) -> Tuple[List[int], List[int]]:
    """Create train/validation splits."""
    indices = list(range(dataset_size))
    torch.manual_seed(42)  # For reproducible splits
    shuffled_indices = torch.randperm(dataset_size).tolist()

    split_idx = int(dataset_size * train_ratio)
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    return train_indices, val_indices


def train_logistic_probe(
    tokenized_dataset: TokenizedProbingDataset,
    hook_point: str,
    hidden_size: int,
    device: str,
    model_name: str,
    val_indices: List[int],
    save_dir: Optional[str] = None,
    hook_point_list: Optional[List[str]] = None,
) -> Tuple[LogisticProbe, Dict[str, List[float]], Dict[str, float], 'ProbePipeline']:
    """Train a logistic probe for truthfulness detection using ProbePipeline."""
    logger.info(f"\n=== Training Logistic Probe on {hook_point} ===")

    # Extract layer number from hook_point
    layer_num = int(hook_point.split(".")[1]) if "blocks." in hook_point else 0

    # Create probe config
    probe_config = LogisticProbeConfig(
        input_size=hidden_size,
        device=device,
        model_name=model_name,
        hook_point=hook_point,
        hook_layer=layer_num,
        name="truthfulness_logistic_probe",
        normalize_weights=True,
        bias=True,
    )

    # Create trainer config
    trainer_config = SupervisedTrainerConfig(
        device=device,
        batch_size=1,  # Reduced to 1 to avoid OOM
        num_epochs=25,
        learning_rate=1e-3,
        train_ratio=0.8,
        show_progress=True,
        handle_class_imbalance=True,
        standardize_activations=True,
        patience=15,
    )

    cache_name = (
        f"./cache/{model_name.replace('.', '_')}/logistic_probe_cache_{hook_point.replace('.', '_')}"
        if hook_point_list is None
        else f"./cache/{model_name.replace('.', '_')}/logistic_probe_cache_multiple_hooks"
    )
    # Create pipeline config
    pipeline_config = ProbePipelineConfig(
        dataset=tokenized_dataset,
        probe_cls=LogisticProbe,
        probe_config=probe_config,
        trainer_cls=SupervisedProbeTrainer,
        trainer_config=trainer_config,
        position_key="last",
        model_name=model_name,
        hook_points=[hook_point] if hook_point_list is None else hook_point_list,
        cache_dir=cache_name,
    )

    # Run pipeline
    pipeline = ProbePipeline(pipeline_config)
    probe, history = pipeline.run()

    # Get activation store for validation metrics
    activation_stores = pipeline.activation_stores
    activation_store = activation_stores[hook_point]

    # Evaluate on validation set
    val_metrics = validate_probe_on_split(probe, activation_store, val_indices, "last")
    logger.info(
        f"Logistic Probe Validation Results: Accuracy={val_metrics['accuracy']:.3f}, "
        f"Precision={val_metrics['precision']:.3f}, Recall={val_metrics['recall']:.3f}, "
        f"F1={val_metrics['f1']:.3f}, AUROC={val_metrics['auroc']:.3f}, Confidence={val_metrics['confidence']:.3f}"
    )

    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        probe_path = os.path.join(
            save_dir, f"logistic_probe_{hook_point.replace('.', '_')}.pt"
        )
        probe.save(probe_path)
        logger.info(f"Saved logistic probe to {probe_path}")

    return probe, history, val_metrics, pipeline


def test_probes_on_dataset(
    probe_results: Dict[str, Dict],
    test_tokenized_dataset: TokenizedProbingDataset,
    model_name: str,
    dataset_name: str = "test",
) -> List[Dict]:
    """Test trained probes on a dataset and return metrics."""
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING PROBES ON {dataset_name.upper()} DATASET")
    logger.info(f"{'='*60}")

    test_results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for hook_point, results in probe_results.items():
        layer_num = hook_point.split(".")[1] if "blocks." in hook_point else "unknown"

        # Create a minimal pipeline config just to use _load_or_collect_activations
        # This will reuse the model and handle caching properly
        dummy_probe_config = LogisticProbeConfig(
            input_size=1,  # Dummy value, not used for collection
            device=device,
            model_name=model_name,
            hook_point=hook_point,
            hook_layer=int(layer_num) if layer_num.isdigit() else 0,
            name="test_probe",
        )

        dummy_trainer_config = SupervisedTrainerConfig(
            device=device,
            batch_size=1,
        )

        cache_name = f"./cache/{model_name.replace('.', '_').replace('/', '_')}/test_cache_{hook_point.replace('.', '_')}"

        pipeline_config = ProbePipelineConfig(
            dataset=test_tokenized_dataset,
            probe_cls=LogisticProbe,
            probe_config=dummy_probe_config,
            trainer_cls=SupervisedProbeTrainer,
            trainer_config=dummy_trainer_config,
            position_key="last",
            model_name=model_name,
            hook_points=[hook_point],
            cache_dir=cache_name,
        )

        # Create pipeline and get activations (with caching)
        pipeline = ProbePipeline(pipeline_config)
        activation_stores = pipeline._load_or_collect_activations()
        activation_store = activation_stores[hook_point]

        for probe_type, (probe, history, _) in results.items():
            # Get all data indices for testing
            dataset_size = len(activation_store.example_indices)
            test_indices = list(range(dataset_size))

            # Evaluate probe on test dataset
            test_metrics = validate_probe_on_split(
                probe, activation_store, test_indices, "last"
            )

            # Move probe to CPU and clear cache to free GPU memory
            probe.to('cpu')
            torch.cuda.empty_cache()

            result = {
                "dataset": dataset_name,
                "layer": layer_num,
                "hook_point": hook_point,
                "probe_type": probe_type,
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "auroc": test_metrics["auroc"],
                "confidence": test_metrics["confidence"],
                "num_samples": test_metrics["num_samples"],
            }
            test_results.append(result)

            logger.info(
                f"Layer {layer_num} {probe_type}: Acc={test_metrics['accuracy']:.3f}, "
                f"F1={test_metrics['f1']:.3f}, AUROC={test_metrics['auroc']:.3f}, Samples={test_metrics['num_samples']}"
            )

        # Clean up pipeline and model after processing this hook point
        if hasattr(pipeline, 'collector') and hasattr(pipeline.collector, 'model'):
            pipeline.collector.model.to('cpu')
            del pipeline.collector.model
            del pipeline.collector
        del pipeline
        del activation_stores
        del activation_store
        torch.cuda.empty_cache()
        logger.info(f"Cleaned up model after testing layer {layer_num}")

    return test_results


def compare_layer_performance(
    probe_results: Dict[str, Dict], dataset_name: str = "validation"
) -> List[Dict]:
    """Compare performance across different model layers and return results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"LAYER PERFORMANCE COMPARISON - {dataset_name.upper()}")
    logger.info(f"{'='*60}")

    # Create performance summary table
    performance_data = []
    for hook_point, results in probe_results.items():
        layer_num = hook_point.split(".")[1] if "blocks." in hook_point else "unknown"

        for probe_type, (probe, history, val_metrics) in results.items():
            performance_data.append(
                {
                    "dataset": dataset_name,
                    "layer": layer_num,
                    "hook_point": hook_point,
                    "probe_type": probe_type,
                    "accuracy": val_metrics["accuracy"],
                    "precision": val_metrics["precision"],
                    "recall": val_metrics["recall"],
                    "f1": val_metrics["f1"],
                    "auroc": val_metrics["auroc"],
                    "confidence": val_metrics["confidence"],
                    "num_samples": val_metrics["num_samples"],
                }
            )

    # Sort by layer number and probe type
    performance_data.sort(
        key=lambda x: (
            int(x["layer"]) if x["layer"].isdigit() else 999,
            x["probe_type"],
        )
    )

    # Print formatted table
    logger.info(
        f"\n{'Layer':<6} {'Hook Point':<25} {'Probe':<10} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'F1':<6} {'AUROC':<6} {'Conf':<6}"
    )
    logger.info("-" * 85)

    for data in performance_data:
        logger.info(
            f"{data['layer']:<6} {data['hook_point']:<25} {data['probe_type']:<10} "
            f"{data['accuracy']:<6.3f} {data['precision']:<6.3f} {data['recall']:<6.3f} "
            f"{data['f1']:<6.3f} {data['auroc']:<6.3f} {data['confidence']:<6.3f}"
        )

    # Find best performing layer/probe combinations
    best_accuracy = max(performance_data, key=lambda x: x["accuracy"])
    best_f1 = max(performance_data, key=lambda x: x["f1"])

    logger.info(f"\n{'='*60}")
    logger.info("BEST PERFORMANCE:")
    logger.info(
        f"Highest Accuracy: {best_accuracy['accuracy']:.3f} (Layer {best_accuracy['layer']}, {best_accuracy['probe_type']} probe)"
    )
    logger.info(
        f"Highest F1 Score: {best_f1['f1']:.3f} (Layer {best_f1['layer']}, {best_f1['probe_type']} probe)"
    )
    logger.info(f"{'='*60}")

    return performance_data


def load_dataset(dataset_name: str = "roleplaying"):
    """Load a dataset by name for easy switching between test datasets.

    Args:
        dataset_name: Name of the dataset to load. Options:
            - "repe_honesty": REPE honesty dataset (you_are_fact_sys variant)
            - "roleplaying": Roleplaying dataset (offpolicy_train variant)

    Returns:
        Tuple of (texts_with_labels, dataset_id)
    """
    repo = DatasetRepository()

    if dataset_name == "repe_honesty":
        dataset_id = "repe_honesty__you_are_fact_sys"
    elif dataset_name == "roleplaying":
        dataset_id = "roleplaying__offpolicy_train"
    else:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. Options: 'repe_honesty', 'roleplaying'"
        )

    model_name = "prewritten"
    dataset = repo.get(dataset_id, model=model_name)

    logger.info(f"Successfully loaded dataset '{dataset_id}'.")
    logger.info(f"Dataset size: {len(dataset.dialogues)} dialogues")

    texts_with_labels = list(zip(dataset.dialogues, dataset.labels))
    return texts_with_labels, dataset_id


def save_results_to_csv(results: List[Dict], filename: str = "results.csv"):
    """Save results to CSV file."""
    if not results:
        logger.warning("No results to save")
        return

    logger.info(f"Saving results to {filename}...")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

    logger.info(f"Results saved to {filename} with {len(results)} rows")
    logger.info(f"Columns: {', '.join(df.columns)}")


def prepare_dataset_for_testing(
    texts_with_labels: List[Tuple[str, bool]],
    tokenizer,
    config=None,
    dataset_id: str = "",
) -> TokenizedProbingDataset:
    """Prepare a dataset for testing by tokenizing (pipelines will handle activation extraction)."""
    examples = []
    for dialogue, label in texts_with_labels:
        # Convert dialogue to string using dialogue_to_string with qwen format
        text = dialogue_to_string(dialogue, "qwen")
        example = ProbingExample(
            text=text,
            label=1 if "deceptive" in label.value else 0,  # Convert boolean to int
            label_text=str(label),
        )
        examples.append(example)

    base_dataset = ProbingDataset(examples=examples, task_type="classification")

    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=base_dataset,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=128,
        truncation=True,
    )

    # Add 'last' token positions to each example manually
    for i, example in enumerate(tokenized_dataset.examples):
        if example.attention_mask:
            last_pos = sum(example.attention_mask) - 1
        else:
            last_pos = len(example.tokens) - 1

        if example.token_positions is None:
            example.token_positions = TokenPositions(positions={})

        example.token_positions.positions["last"] = last_pos

    return tokenized_dataset


def main():
    """Main function to train on repe_honesty, validate, and test on roleplaying."""
    logger.info("SPAR - Training on REPE Honesty, Testing on Roleplaying")

    all_results = []

    try:
        # --- 1. Load REPE Honesty dataset for training/validation ---
        logger.info("--- Loading REPE Honesty dataset for training/validation ---")
        train_texts, train_dataset_id = load_dataset("repe_honesty")

        # --- 2. Setup model config and tokenizer ---
        model_name = "Qwen/Qwen2.5-3B"
        model_config = AutoConfig.from_pretrained(model_name)
        num_layers = model_config.num_hidden_layers
        logger.info(f"Model {model_name} has {num_layers} layers")

        # Determine which layers to use
        if num_layers <= 10:
            layer_indices = list(range(num_layers))
        else:
            layer_indices = [0, num_layers - 1]
            if num_layers > 2:
                middle_layers = np.linspace(
                    1, num_layers - 2, min(8, num_layers - 2), dtype=int
                )
                layer_indices.extend(middle_layers.tolist())
            layer_indices = sorted(list(set(layer_indices)))

        hook_points = [f"blocks.{i}.hook_resid_post" for i in layer_indices]
        logger.info(f"Selected layers: {layer_indices}")
        logger.info(f"Hook points: {hook_points}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- 3. Prepare training dataset ---
        train_tokenized_dataset = prepare_dataset_for_testing(
            train_texts, tokenizer, None, train_dataset_id
        )

        # --- 4. Create train/validation splits ---
        logger.info("\n--- Creating train/validation splits ---")
        dataset_size = len(train_tokenized_dataset.examples)
        train_indices, val_indices = create_validation_splits(
            dataset_size, train_ratio=0.8
        )
        logger.info(
            f"Dataset split: {len(train_indices)} train, {len(val_indices)} validation examples"
        )

        logger.info("TRAINING PROBES ON REPE HONESTY DATASET")

        all_probe_results = {}
        save_dir = "./probe_checkpoints"

        hidden_size = model_config.hidden_size

        for hook_point in hook_points:
            logger.info(f"\n{'='*40}")
            logger.info(f"TRAINING PROBES FOR {hook_point}")
            logger.info(f"Hidden size: {hidden_size}, Device: {device}")
            logger.info(f"{'='*40}")

            hook_results = {}
            try:
                # --- Train Logistic Probe ---
                logistic_probe, logistic_history, logistic_val_metrics, pipeline = (
                    train_logistic_probe(
                        train_tokenized_dataset,
                        hook_point,
                        hidden_size,
                        device,
                        model_name,
                        val_indices,
                        save_dir,
                        # hook_point_list=config.hook_points,
                    )
                )

                hook_results["logistic"] = (
                    logistic_probe,
                    logistic_history,
                    logistic_val_metrics,
                )

                # Clean up the model from GPU memory after training
                if hasattr(pipeline, 'collector') and hasattr(pipeline.collector, 'model'):
                    pipeline.collector.model.to('cpu')
                    del pipeline.collector.model
                    del pipeline.collector
                del pipeline
                torch.cuda.empty_cache()
                logger.info(f"Cleaned up model for {hook_point}")
            except Exception as e:
                logger.error(
                    f"Error training probes for hook point {hook_point}: {e}",
                    exc_info=True,
                )
                continue

            all_probe_results[hook_point] = hook_results

            # Clear CUDA cache after each hook point to avoid OOM
            torch.cuda.empty_cache()

        validation_results = compare_layer_performance(all_probe_results, "validation")
        all_results.extend(validation_results)

        test_texts, test_dataset_id = load_dataset("roleplaying")

        test_tokenized_dataset = prepare_dataset_for_testing(
            test_texts, tokenizer, None, test_dataset_id
        )

        test_results = test_probes_on_dataset(
            all_probe_results, test_tokenized_dataset, model_name, "test"
        )
        all_results.extend(test_results)

        save_results_to_csv(all_results, "results.csv")

        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Training dataset: REPE Honesty ({len(train_texts)} samples)")
        logger.info(f"Test dataset: Roleplaying ({len(test_texts)} samples)")
        logger.info(f"Results saved to: results.csv ({len(all_results)} rows)")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(
            f"An unexpected error occurred in the main process: {e}",
            exc_info=True,
        )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
