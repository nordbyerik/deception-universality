#!/usr/bin/env python3

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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
        model_name: str = "Qwen/Qwen2.5-3B",
        device: str = "cuda" if torch.cuda.is_available() else "cuda",
        max_layers: Optional[int] = None,
    ):
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading model {model_name} with nnsight...")
        self.model = LanguageModel(model_name, device_map=device)
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


class SimpleProbe(nn.Module):
    def __init__(self, input_size: int, device: str = "cpu"):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_probe(
    activations: torch.Tensor,
    labels: torch.Tensor,
    train_indices: List[int],
    val_indices: List[int],
    device: str,
    num_epochs: int = 25,
    learning_rate: float = 1e-3,
    batch_size: int = 2,
    break_probe: bool = False,
) -> Tuple[SimpleProbe, Dict[str, List[float]], Dict[str, float]]:
    labels = labels if not break_probe else np.random.randint(0, 2)
    X_train = activations[train_indices].to(device)
    y_train = labels[train_indices].float().to(device)
    X_val = activations[val_indices].to(device)
    y_val = labels[val_indices].float().to(device)

    scaler = StandardScaler()
    X_train_np = X_train.cpu().numpy()
    X_val_np = X_val.cpu().numpy()
    X_train_scaled = torch.tensor(
        scaler.fit_transform(X_train_np), dtype=torch.float32
    ).to(device)
    X_val_scaled = torch.tensor(scaler.transform(X_val_np), dtype=torch.float32).to(
        device
    )

    probe = SimpleProbe(activations.shape[1], device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    pos_weight = (y_train == 0).sum().float() / (y_train == 1).sum().float()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 15

    for epoch in range(num_epochs):
        probe.train()
        epoch_loss = 0.0
        num_batches = 0
        print(X_train_scaled.device)
        indices = torch.randperm(len(X_train_scaled))
        for i in range(0, len(X_train_scaled), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_X = X_train_scaled[batch_idx]
            batch_y = y_train[batch_idx]
            optimizer.zero_grad()

            outputs = probe(batch_X)
            outputs = outputs.squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches

        probe.eval()
        with torch.no_grad():
            val_outputs = probe(X_val_scaled)
            val_loss = criterion(val_outputs.squeeze(), y_val)
            val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
            val_accuracy = (val_preds.squeeze() == y_val).float().mean()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss.item())
        history["val_accuracy"].append(val_accuracy.item())

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss={avg_train_loss:.4f}, "
                f"Val Loss={val_loss.item():.4f}, "
                f"Val Acc={val_accuracy.item():.3f}"
            )

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    probe.eval()
    with torch.no_grad():
        val_outputs = probe(X_val_scaled)
        val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
        accuracy = (val_preds.squeeze() == y_val).float().mean()

        tp = ((val_preds.squeeze() == 1) & (y_val == 1)).sum().float()
        fp = ((val_preds.squeeze() == 1) & (y_val == 0)).sum().float()
        fn = ((val_preds.squeeze() == 0) & (y_val == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        probabilities = torch.sigmoid(val_outputs).squeeze()
        auroc = roc_auc_score(y_val.cpu().numpy(), probabilities.cpu().numpy())
        confidence = probabilities.mean()

        deceptive_ratio = val_preds.squeeze().sum() / len(val_preds.squeeze())
    val_metrics = {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "auroc": auroc,
        "confidence": confidence.item(),
        "num_samples": len(y_val),
        "deceptive_ratio": deceptive_ratio
    }

    probe.scaler = scaler

    return probe, history, val_metrics


def test_probe(
    probe: SimpleProbe,
    activations: torch.Tensor,
    labels: torch.Tensor,
    device: str,
) -> Dict[str, float]:
    X_test = activations.to(device)
    y_test = labels.float().to(device)

    X_test_np = X_test.cpu().numpy()
    X_test_scaled = torch.tensor(
        probe.scaler.transform(X_test_np), dtype=torch.float32
    ).to(device)

    probe.eval()
    with torch.no_grad():
        outputs = probe(X_test_scaled)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        accuracy = (preds.squeeze() == y_test).float().mean()

        tp = ((preds.squeeze() == 1) & (y_test == 1)).sum().float()
        fp = ((preds.squeeze() == 1) & (y_test == 0)).sum().float()
        fn = ((preds.squeeze() == 0) & (y_test == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        probabilities = torch.sigmoid(outputs).squeeze()
        auroc = roc_auc_score(y_test.cpu().numpy(), probabilities.cpu().numpy())
        confidence = probabilities.mean()
        deceptive_ratio = preds.squeeze().sum() / len(preds.squeeze())
    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "auroc": auroc,
        "confidence": confidence.item(),
        "num_samples": len(y_test),
        "deceptive_ratio": deceptive_ratio
    }


def load_dataset(dataset_name: str = "roleplaying") -> Tuple[List[str], List[int], str]:
    repo = DatasetRepository()

    if dataset_name == "repe_honesty":
        dataset_id = "repe_honesty__you_are_fact_sys"
    elif dataset_name == "roleplaying":
        dataset_id = "roleplaying__offpolicy_train"
    elif dataset_name == "roleplaying_plain":
        dataset_id = "roleplaying__plain"
    else:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. Options: 'repe_honesty', 'roleplaying'"
        )

    model_name = "prewritten"
    dataset = repo.get(dataset_id, model=model_name)

    logger.info(f"Successfully loaded dataset '{dataset_id}'.")
    logger.info(f"Dataset size: {len(dataset.dialogues)} dialogues")

    texts = []
    labels = []
    for dialogue, label in zip(dataset.dialogues, dataset.labels):
        text = dialogue_to_string(dialogue, "qwen")
        texts.append(text)
        labels.append(1 if "deceptive" in label.value else 0)

    return texts, labels, dataset_id


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
    logger.info("SPAR - Training on REPE Honesty, Testing on Roleplaying (NNsight)")

    all_results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        logger.info("--- Loading REPE Honesty dataset for training/validation ---")
        train_texts, train_labels, train_dataset_id = load_dataset("repe_honesty")

        logger.info("--- Initializing NNsight model ---")
        extractor = NNsightActivationExtractor(device=device, max_layers=4)

        logger.info("--- Creating train/validation splits ---")
        dataset_size = len(train_texts)
        train_indices, val_indices = create_validation_splits(
            dataset_size, train_ratio=0.8
        )
        logger.info(
            f"Dataset split: {len(train_indices)} train, {len(val_indices)} validation examples"
        )

        logger.info("TRAINING PROBES ON REPE HONESTY DATASET")

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
                train_texts, train_labels, layer_idx, batch_size=4, max_length=128
            )

            logger.info("Training probe...")
            probe, history, val_metrics = train_probe(
                cache.activations,
                cache.labels,
                train_indices,
                val_indices,
                device,
                num_epochs=25,
                learning_rate=1e-3,
                batch_size=2,
            )

            logger.info(
                f"Layer {layer_idx} Validation Results: "
                f"Accuracy={val_metrics['accuracy']:.3f}, "
                f"Precision={val_metrics['precision']:.3f}, "
                f"Recall={val_metrics['recall']:.3f}, "
                f"F1={val_metrics['f1']:.3f}, "
                f"AUROC={val_metrics['auroc']:.3f}"
                f"Deceptive Ratio={val_metrics['deceptive_ratio']:.3f}"
            )

            probe_path = os.path.join(save_dir, f"probe_layer_{layer_idx}.pt")
            torch.save(
                {"state_dict": probe.state_dict(), "scaler": probe.scaler}, probe_path
            )

            all_probe_results[layer_idx] = {
                "probe": probe,
                "history": history,
                "val_metrics": val_metrics,
                "cache": cache,
            }

            all_results.append(
                {
                    "dataset": "validation",
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

        logger.info("\n--- Loading Roleplaying dataset for testing ---")
        test_texts, test_labels, test_dataset_id = load_dataset("roleplaying")

        logger.info("TESTING PROBES ON ROLEPLAYING DATASET")

        for layer_idx in layer_indices:
            logger.info(f"\nExtracting test activations from layer {layer_idx}...")
            test_cache = extractor.extract_activations(
                test_texts, test_labels, layer_idx, batch_size=4, max_length=128
            )

            probe = all_probe_results[layer_idx]["probe"]
            test_metrics = test_probe(
                probe, test_cache.activations, test_cache.labels, device
            )

            logger.info(
                f"Layer {layer_idx} Test Results: "
                f"Accuracy={test_metrics['accuracy']:.3f}, "
                f"F1={test_metrics['f1']:.3f}, "
                f"AUROC={test_metrics['auroc']:.3f}"
                f"Deception Ratio={test_metrics['deceptive_ratio']:.3f}"
            )

            all_results.append(
                {
                    "dataset": "test",
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
        logger.info(f"Training dataset: REPE Honesty ({len(train_texts)} samples)")
        logger.info(f"Test dataset: Roleplaying ({len(test_texts)} samples)")
        logger.info(f"Results saved to: results_nnsight.csv ({len(all_results)} rows)")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(
            f"An unexpected error occurred in the main process: {e}", exc_info=True
        )


if __name__ == "__main__":
    main()
