# Requirements — Activation Probing Toolkit

This document lists functional and non-functional requirements for the activation probing toolkit.

1. Functional requirements

- FR1: Activation extraction
  - Provide a programmatic API to extract activations from transformer models during forward passes.
  - Support Hugging Face Transformers and PyTorch models. Keep backend abstraction to allow adding JAX/Flax later.

- FR2: Layer and token selection
  - Allow selecting which layer(s) to probe by index or name.
  - Support token aggregation strategies: `last`, `first`, `mean`, `max`, `pooler`, `cls`, and `custom` (user-provided function).
  - Provide easy CLI/Programmatic switch for strategies.

- FR3: Dataset runner
  - Provide tools to run batched inference over datasets and save activations.
  - Support streaming large datasets (iterators) and checkpointing.

- FR4: Serialization
  - Save activations in a compact, versioned format (preferably HDF5, NPZ, or Arrow). Include metadata about model, tokenizer, layer, strategy, input ids, and timestamps.

- FR5: Extensibility
  - Design modular interfaces for model backends, extraction strategies, and dataset connectors.
  - Minimal coupling between I/O (datasets/storage) and extraction logic.

- FR6: Testing
  - Core modules must be covered by unit tests.
  - Provide example integration tests that use small synthetic models/datasets.

2. Non-functional requirements

- NFR1: Reproducibility
  - Record exact model name/version, random seeds, and tokenizer configuration in metadata.

- NFR2: Performance
  - Efficient memory usage when storing activations; avoid keeping activations for entire dataset in RAM.

- NFR3: Collaboration
  - Clear contribution guidelines and tests for PRs.

3. Constraints and assumptions

- Initial backend: PyTorch + Hugging Face Transformers.
- Python>=3.10. Use typed code (PEP 484) and prefer mypy-compatible annotations.
- Minimal external dependencies: `transformers`, `torch`, `pytest`, `numpy`, `h5py` (optional), `tqdm`.

4. Success criteria

- A core `ActivationExtractor` class with unit tests.
- A `DatasetRunner` that batches and stores activations.
- Documentation (this repo) describing how to add new strategies and backends.
