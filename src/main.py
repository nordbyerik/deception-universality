#!/usr/bin/env python3
"""
Simple script to load and test Qwen 0.6B model using probity for activation extraction.
This demonstrates using probity's collection system for probe-ready activations.
"""

import torch
from transformers import AutoTokenizer
from probity.datasets.base import ProbingExample, ProbingDataset
from probity.datasets.tokenized import TokenizedProbingDataset
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
import os
from typing import Dict, List, Tuple, Optional


def create_simple_dataset(texts_with_labels):
    """Create a simple probing dataset from text-label pairs."""
    examples = []
    for text, label in texts_with_labels:
        example = ProbingExample(
            text=text,
            label=1 if label else 0,  # Convert boolean to int
            label_text=str(label),
        )
        examples.append(example)

    return ProbingDataset(examples=examples, task_type="classification")


def setup_collector(model_name="Qwen/Qwen2.5-0.5B"):
    """Setup the probity collector for activation extraction."""
    # Define which layers we want to extract activations from
    # For Qwen models, we'll extract from multiple residual stream positions
    hook_points = [
        "blocks.12.hook_resid_post",  # Middle layer
        "blocks.23.hook_resid_post",  # Final layer (Qwen2.5-0.5B has 24 layers)
    ]

    config = TransformerLensConfig(
        model_name=model_name,
        hook_points=hook_points,
        batch_size=4,  # Small batch size for demo
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"Initializing collector for {model_name}...")
    collector = TransformerLensCollector(config)

    return collector, config


def extract_activations_with_probity(collector, dataset):
    """Extract activations using probity's collection system."""
    print("\nExtracting activations using probity...")

    # Collect activations for all hook points
    activation_stores = collector.collect(dataset)

    # Print info about collected activations
    for hook_point, store in activation_stores.items():
        print(f"\nHook point: {hook_point}")
        print(f"Activation shape: {store.raw_activations.shape}")
        print(f"Number of examples: {len(store.example_indices)}")
        print(f"Hidden size: {store.hidden_size}")

    return activation_stores


def evaluate_probe_accuracy(
    probe, activation_store, position_key: str = "last"
) -> Dict[str, float]:
    """Evaluate probe accuracy and return metrics."""
    X, y = activation_store.get_probe_data(position_key)

    # Move to probe device
    X = X.to(probe.config.device)
    y = y.to(probe.config.device)

    with torch.no_grad():
        predictions = probe(X)

        # Convert predictions to binary for accuracy calculation
        if predictions.shape[1] == 1:  # Binary classification
            binary_preds = (torch.sigmoid(predictions) > 0.5).float()
            accuracy = (binary_preds.squeeze() == y.float()).float().mean()
        else:  # Multi-class
            binary_preds = torch.argmax(predictions, dim=1)
            accuracy = (binary_preds == y.long().squeeze()).float().mean()

        # Calculate mean prediction confidence
        if predictions.shape[1] == 1:
            confidence = torch.sigmoid(predictions).mean()
        else:
            confidence = torch.softmax(predictions, dim=1).max(dim=1)[0].mean()

    return {
        "accuracy": accuracy.item(),
        "confidence": confidence.item(),
        "num_samples": len(y),
    }


def train_linear_probe(
    activation_stores: Dict[str, any],
    hook_point: str,
    hidden_size: int,
    device: str,
    save_dir: Optional[str] = None,
) -> Tuple[LinearProbe, Dict[str, List[float]]]:
    """Train a linear probe for truthfulness detection."""
    print(f"\n=== Training Linear Probe on {hook_point} ===")

    # Debug: Check activation store
    activation_store = activation_stores[hook_point]
    print(f"Activation store labels shape: {activation_store.labels.shape}")
    print(f"Raw activations shape: {activation_store.raw_activations.shape}")
    print(f"Example indices: {activation_store.example_indices}")

    # Test get_probe_data directly
    try:
        X, y = activation_store.get_probe_data("last")
        print(f"Got probe data: X shape {X.shape}, y shape {y.shape}")
    except Exception as e:
        print(f"Error getting probe data: {e}")
        import traceback

        traceback.print_exc()
        raise

    # Create probe config
    config = LinearProbeConfig(
        input_size=hidden_size,
        device=device,
        model_name="Qwen/Qwen2.5-0.5B",
        hook_point=hook_point,
        name="truthfulness_linear_probe",
        loss_type="mse",
        normalize_weights=True,
        bias=True,
    )

    # Create probe and trainer
    probe = LinearProbe(config)
    trainer_config = SupervisedTrainerConfig(
        device=device,
        batch_size=2,  # Small batch for demo
        num_epochs=20,
        learning_rate=1e-3,
        train_ratio=0.8,
        show_progress=True,
        standardize_activations=True,
    )
    trainer = SupervisedProbeTrainer(trainer_config)

    # Prepare data
    train_loader, val_loader = trainer.prepare_supervised_data(activation_store, "last")
    print(f"Created train loader with {len(train_loader)} batches")

    # Train
    history = trainer.train(probe, train_loader, val_loader)

    # Evaluate
    metrics = evaluate_probe_accuracy(probe, activation_store, "last")
    print(
        f"Linear Probe Results: Accuracy={metrics['accuracy']:.3f}, Confidence={metrics['confidence']:.3f}"
    )

    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        probe_path = os.path.join(
            save_dir, f"linear_probe_{hook_point.replace('.', '_')}.pt"
        )
        probe.save(probe_path)
        print(f"Saved linear probe to {probe_path}")

    return probe, history


def train_logistic_probe(
    activation_stores: Dict[str, any],
    hook_point: str,
    hidden_size: int,
    device: str,
    save_dir: Optional[str] = None,
) -> Tuple[LogisticProbe, Dict[str, List[float]]]:
    """Train a logistic probe for truthfulness detection."""
    print(f"\n=== Training Logistic Probe on {hook_point} ===")

    # Create probe config
    config = LogisticProbeConfig(
        input_size=hidden_size,
        device=device,
        model_name="Qwen/Qwen2.5-0.5B",
        hook_point=hook_point,
        name="truthfulness_logistic_probe",
        normalize_weights=True,
        bias=True,
    )

    # Create probe and trainer
    probe = LogisticProbe(config)
    trainer_config = SupervisedTrainerConfig(
        device=device,
        batch_size=2,  # Small batch for demo
        num_epochs=25,
        learning_rate=1e-3,
        train_ratio=0.8,
        show_progress=True,
        handle_class_imbalance=True,
        standardize_activations=True,
    )
    trainer = SupervisedProbeTrainer(trainer_config)

    # Prepare data
    activation_store = activation_stores[hook_point]
    train_loader, val_loader = trainer.prepare_supervised_data(activation_store, "last")

    # Train
    history = trainer.train(probe, train_loader, val_loader)

    # Evaluate
    metrics = evaluate_probe_accuracy(probe, activation_store, "last")
    print(
        f"Logistic Probe Results: Accuracy={metrics['accuracy']:.3f}, Confidence={metrics['confidence']:.3f}"
    )

    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        probe_path = os.path.join(
            save_dir, f"logistic_probe_{hook_point.replace('.', '_')}.pt"
        )
        probe.save(probe_path)
        print(f"Saved logistic probe to {probe_path}")

    return probe, history


def train_directional_probes(
    activation_stores: Dict[str, any],
    hook_point: str,
    hidden_size: int,
    device: str,
    save_dir: Optional[str] = None,
) -> Dict[str, Tuple[any, Dict[str, List[float]]]]:
    """Train directional probes (KMeans, PCA, MeanDiff) for truthfulness detection."""
    print(f"\n=== Training Directional Probes on {hook_point} ===")

    results = {}
    activation_store = activation_stores[hook_point]

    # KMeans Probe
    print("\n--- Training KMeans Probe ---")
    kmeans_config = KMeansProbeConfig(
        input_size=hidden_size,
        device=device,
        model_name="Qwen/Qwen2.5-0.5B",
        hook_point=hook_point,
        name="truthfulness_kmeans_probe",
        n_clusters=2,
        normalize_weights=True,
    )
    kmeans_probe = KMeansProbe(kmeans_config)

    trainer_config = DirectionalTrainerConfig(
        device=device,
        standardize_activations=True,
    )
    trainer = DirectionalProbeTrainer(trainer_config)

    train_loader, _ = trainer.prepare_supervised_data(activation_store, "last")
    kmeans_history = trainer.train(kmeans_probe, train_loader)
    kmeans_metrics = evaluate_probe_accuracy(kmeans_probe, activation_store, "last")
    print(f"KMeans Probe Results: Accuracy={kmeans_metrics['accuracy']:.3f}")
    results["kmeans"] = (kmeans_probe, kmeans_history)

    # PCA Probe
    print("\n--- Training PCA Probe ---")
    pca_config = PCAProbeConfig(
        input_size=hidden_size,
        device=device,
        model_name="Qwen/Qwen2.5-0.5B",
        hook_point=hook_point,
        name="truthfulness_pca_probe",
        n_components=1,
        normalize_weights=True,
    )
    pca_probe = PCAProbe(pca_config)

    pca_history = trainer.train(pca_probe, train_loader)
    pca_metrics = evaluate_probe_accuracy(pca_probe, activation_store, "last")
    print(f"PCA Probe Results: Accuracy={pca_metrics['accuracy']:.3f}")
    results["pca"] = (pca_probe, pca_history)

    # Mean Difference Probe
    print("\n--- Training Mean Difference Probe ---")
    meandiff_config = MeanDiffProbeConfig(
        input_size=hidden_size,
        device=device,
        model_name="Qwen/Qwen2.5-0.5B",
        hook_point=hook_point,
        name="truthfulness_meandiff_probe",
        normalize_weights=True,
    )
    meandiff_probe = MeanDifferenceProbe(meandiff_config)

    meandiff_history = trainer.train(meandiff_probe, train_loader)
    meandiff_metrics = evaluate_probe_accuracy(meandiff_probe, activation_store, "last")
    print(f"Mean Difference Probe Results: Accuracy={meandiff_metrics['accuracy']:.3f}")
    results["meandiff"] = (meandiff_probe, meandiff_history)

    # Save probes if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for probe_type, (probe, _) in results.items():
            probe_path = os.path.join(
                save_dir, f"{probe_type}_probe_{hook_point.replace('.', '_')}.pt"
            )
            probe.save(probe_path)
            print(f"Saved {probe_type} probe to {probe_path}")

    return results


def main():
    """Main function to demonstrate Qwen usage with probity."""
    print("SPAR - Qwen 0.6B with Probity Collection")
    print("=" * 50)

    try:
        # Create sample dataset for demonstration
        sample_texts = [
            ("I am telling the truth about this matter.", True),
            ("The weather today is quite pleasant.", True),
            ("This statement is completely false and misleading.", False),
            ("The cat is sleeping on the windowsill.", True),
        ]

        base_dataset = create_simple_dataset(sample_texts)

        # Setup collector
        collector, config = setup_collector()

        # Create tokenized dataset
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
            dataset=base_dataset, tokenizer=tokenizer
        )

        # Add 'last' token positions to each example manually
        print("Adding 'last' token positions to examples...")
        for i, example in enumerate(tokenized_dataset.examples):
            if example.attention_mask:
                # Find the last non-padding token
                last_pos = sum(example.attention_mask) - 1
            else:
                # If no attention mask, assume last token is last position
                last_pos = len(example.tokens) - 1

            # Create token_positions if it doesn't exist
            if example.token_positions is None:
                from probity.datasets.tokenized import TokenPositions

                example.token_positions = TokenPositions(positions={})

            example.token_positions.positions["last"] = last_pos
            print(
                f"Example {i}: Set last position to {last_pos} (seq_len: {len(example.tokens)})"
            )

        # Extract activations using probity
        activation_stores = extract_activations_with_probity(
            collector, tokenized_dataset
        )

        # Print activation info
        for hook_point, store in activation_stores.items():
            first_seq_len = store.sequence_lengths[0].item()
            first_example_activations = store.raw_activations[
                0, first_seq_len - 1, :
            ]  # Last token
            print(
                f"  First example last token shape: {first_example_activations.shape}"
            )
            print(f"  Sequence length: {first_seq_len}")
            print(f"  Mean activation: {first_example_activations.mean():.4f}")
            print(f"  Std activation: {first_example_activations.std():.4f}")

        print("\n" + "=" * 60)
        print("TRAINING PROBES FOR TRUTHFULNESS DETECTION")
        print("=" * 60)

        # Get device and hidden size from config
        device = config.device

        # Train probes for each hook point
        all_probe_results = {}
        save_dir = "./probe_checkpoints"  # Directory to save trained probes

        for hook_point, store in activation_stores.items():
            hidden_size = store.hidden_size
            print(f"\n{'='*40}")
            print(f"TRAINING PROBES FOR {hook_point}")
            print(f"Hidden size: {hidden_size}, Device: {device}")
            print(f"{'='*40}")

            hook_results = {}

            # Train Linear Probe
            try:
                linear_probe, linear_history = train_linear_probe(
                    activation_stores, hook_point, hidden_size, device, save_dir
                )
                hook_results["linear"] = (linear_probe, linear_history)
            except Exception as e:
                print(f"Error training linear probe: {e}")
                import traceback

                traceback.print_exc()

            # Train Logistic Probe
            try:
                logistic_probe, logistic_history = train_logistic_probe(
                    activation_stores, hook_point, hidden_size, device, save_dir
                )
                hook_results["logistic"] = (logistic_probe, logistic_history)
            except Exception as e:
                print(f"Error training logistic probe: {e}")
                import traceback

                traceback.print_exc()

        # Print final summary
        print("\n" + "=" * 60)
        print("PROBE TRAINING SUMMARY")
        print("=" * 60)

        for hook_point, hook_results in all_probe_results.items():
            print(f"\n{hook_point}:")
            for probe_type, (probe, history) in hook_results.items():
                if history and "train_loss" in history and history["train_loss"]:
                    final_loss = history["train_loss"][-1]
                    print(f"  {probe_type:12} - Final Loss: {final_loss:.4f}")
                else:
                    print(f"  {probe_type:12} - No training history available")

        print(f"\nAll probes saved to: {save_dir}")
        print("\nProbe training complete! You can now use these probes for inference.")
        print("Try loading a probe with: probe = LinearProbe.load('path/to/probe.pt')")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have probity and dependencies installed:")
        print("pip install -e /path/to/probity")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
