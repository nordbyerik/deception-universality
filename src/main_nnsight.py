#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoConfig
from nnsight import LanguageModel
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

from data.deception_detection.deception_detection.repository import DatasetRepository
from data.deception_detection.deception_detection.types import dialogue_to_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ActivationCache:
    activations: torch.Tensor
    labels: torch.Tensor
    layer_idx: int
    hook_name: str


class NNsightActivationExtractor:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "cuda" if torch.cuda.is_available() else "cuda",
        max_layers: Optional[int] = None,
    ):
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading model {model_name} with nnsight...")
        self.model = LanguageModel(model_name, device_map=device, load_in_8bit=True)
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

    def extract_activations(
        self,
        texts: List[str],
        labels: List[int],
        layer_idx: int,
        batch_size: int = 4,
        max_length: int = 128,
    ) -> ActivationCache:
        all_activations = []
        all_labels = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                with self.model.trace(input_ids):
                    hidden_states = self.model.model.layers[layer_idx].output.save()

                activations = hidden_states

                last_token_activations = []
                for j, mask in enumerate(attention_mask):
                    last_pos = mask.sum().item() - 1
                    last_token_activations.append(activations[j, last_pos, :])

                batch_activations = torch.stack(last_token_activations)
                all_activations.append(batch_activations.cpu())
                all_labels.extend(batch_labels)

        return ActivationCache(
            activations=torch.cat(all_activations, dim=0),
            labels=torch.tensor(all_labels),
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


def load_dataset(dataset_name: str = "roleplaying") -> Tuple[List[str], List[int], str]:
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
    dataset = repo.get(
        dataset_id, model=model_name, trim_reasoning=False
    )  # TODO: Check trim reasoning
    logger.info(f"Successfully loaded dataset '{dataset_id}'.")
    logger.info(f"Dataset size: {len(dataset.dialogues)} dialogues")

    texts = []
    labels = []
    for dialogue, label in zip(dataset.dialogues, dataset.labels):
        text = dialogue_to_string(dialogue, "qwen")
        texts.append(text)
        labels.append(1 if "deceptive" in label.value else 0)
    print("Dataset:", dataset_name, "Texts:", texts[0])
    return texts, labels, dataset_id


def balance_dataset_oversample(
    texts: List[str],
    labels: List[int],
    sources: List[str],
    target_samples_per_dataset: Optional[int] = None,
) -> Tuple[List[str], List[int], List[str]]:
    if target_samples_per_dataset is None:
        return texts, labels, sources

    from collections import defaultdict
    import random

    random.seed(42)
    np.random.seed(42)

    dataset_indices = defaultdict(list)
    for i, source in enumerate(sources):
        dataset_indices[source].append(i)

    balanced_texts = []
    balanced_labels = []
    balanced_sources = []

    for dataset_name, indices in dataset_indices.items():
        dataset_texts = [texts[i] for i in indices]
        dataset_labels = [labels[i] for i in indices]

        if len(indices) >= target_samples_per_dataset:
            sampled = random.sample(indices, target_samples_per_dataset)
            balanced_texts.extend([texts[i] for i in sampled])
            balanced_labels.extend([labels[i] for i in sampled])
            balanced_sources.extend([dataset_name] * target_samples_per_dataset)
            logger.info(
                f"{dataset_name}: downsampled from {len(indices)} to {target_samples_per_dataset} samples"
            )
        else:
            logger.info(
                f"{dataset_name}: oversampling from {len(indices)} to {target_samples_per_dataset} samples"
            )

            balanced_texts.extend(dataset_texts)
            balanced_labels.extend(dataset_labels)
            balanced_sources.extend([dataset_name] * len(dataset_texts))

            remaining = target_samples_per_dataset - len(indices)
            for _ in range(remaining):
                idx = random.randint(0, len(dataset_texts) - 1)
                balanced_texts.append(dataset_texts[idx])
                balanced_labels.append(dataset_labels[idx])
                balanced_sources.append(dataset_name)

            logger.info(f"{dataset_name}: added {remaining} duplicate samples")

    return balanced_texts, balanced_labels, balanced_sources


def create_validation_splits(
    dataset_size: int, train_ratio: float = 0.8
) -> Tuple[List[int], List[int]]:
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(dataset_size).tolist()

    split_idx = int(dataset_size * train_ratio)
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    return train_indices, val_indices


def save_results_to_csv(results: List[Dict], filename: str = "results.csv"):
    if not results:
        logger.warning("No results to save")
        return

    logger.info(f"Saving results to {filename}...")
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    logger.info(f"Results saved to {filename} with {len(results)} rows")


def main():
    logger.info(
        "SPAR - Training on Multiple Datasets, Testing on Roleplaying (NNsight)"
    )

    all_results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset_names = [
        "repe_honesty",
        "ai_audit",
        "werewolf",
        "ai_liar",
        "ai_liar_with",
    ]
    test_dataset_name = "roleplaying"
    target_samples_per_dataset = 100

    try:
        logger.info(f"--- Loading training datasets: {train_dataset_names} ---")

        all_train_texts = []
        all_train_labels = []
        dataset_sources = []

        for dataset_name in train_dataset_names:
            texts, labels, dataset_id = load_dataset(dataset_name)
            all_train_texts.extend(texts)
            all_train_labels.extend(labels)
            dataset_sources.extend([dataset_name] * len(texts))
            logger.info(f"Added {len(texts)} samples from {dataset_name}")

        logger.info(f"Total training samples before balancing: {len(all_train_texts)}")

        logger.info("--- Initializing NNsight model ---")
        extractor = NNsightActivationExtractor(device=device, max_layers=4)

        logger.info("--- Balancing datasets with plain oversampling ---")
        all_train_texts, all_train_labels, dataset_sources = balance_dataset_oversample(
            all_train_texts,
            all_train_labels,
            dataset_sources,
            target_samples_per_dataset,
        )

        logger.info(f"Total training samples after balancing: {len(all_train_texts)}")
        logger.info(
            f"Label distribution: {sum(all_train_labels)} deceptive, {len(all_train_labels) - sum(all_train_labels)} honest"
        )

        logger.info("--- Creating train/validation splits ---")
        dataset_size = len(all_train_texts)
        train_indices, val_indices = create_validation_splits(
            dataset_size, train_ratio=0.8
        )
        logger.info(
            f"Dataset split: {len(train_indices)} train, {len(val_indices)} validation examples"
        )

        logger.info("TRAINING PROBES ON COMBINED DATASETS")

        layer_indices = extractor.select_layers()
        logger.info(f"Selected layers: {layer_indices}")

        all_probe_results = {}
        save_dir = "./probe_checkpoints_nnsight"
        os.makedirs(save_dir, exist_ok=True)

        for layer_idx in layer_indices:
            logger.info(f"\n{'='*40}")
            logger.info(f"TRAINING PROBE FOR LAYER {layer_idx}")
            logger.info(f"{'='*40}")

            logger.info(f"Extracting activations from layer {layer_idx}...")
            cache = extractor.extract_activations(
                all_train_texts,
                all_train_labels,
                layer_idx,
                batch_size=4,
                max_length=128,
            )

            logger.info("Training probe...")
            probe, scaler, val_metrics = train_probe(
                cache.activations,
                cache.labels,
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
                "cache": cache,
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
        test_texts, test_labels, test_dataset_id = load_dataset(test_dataset_name)

        logger.info(f"TESTING PROBES ON {test_dataset_name.upper()} DATASET")

        for layer_idx in layer_indices:
            logger.info(f"\nExtracting test activations from layer {layer_idx}...")
            test_cache = extractor.extract_activations(
                test_texts, test_labels, layer_idx, batch_size=4, max_length=128
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

        save_results_to_csv(all_results, "results_nnsight.csv")

        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 60)
        logger.info(
            f"Training datasets: {', '.join(train_dataset_names)} ({len(all_train_texts)} samples)"
        )
        logger.info(f"Test dataset: {test_dataset_name} ({len(test_texts)} samples)")
        logger.info(f"Results saved to: results_nnsight.csv ({len(all_results)} rows)")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(
            f"An unexpected error occurred in the main process: {e}", exc_info=True
        )


if __name__ == "__main__":
    main()
