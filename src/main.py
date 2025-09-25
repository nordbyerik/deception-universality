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

        # Extract activations using probity
        activation_stores = extract_activations_with_probity(
            collector, tokenized_dataset
        )

        for _, store in activation_stores.items():

            # Get activations for first example at last token position
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

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have probity and dependencies installed:")
        print("pip install -e /path/to/probity")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
