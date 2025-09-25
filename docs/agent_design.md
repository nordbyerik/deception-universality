# Agent & Design Doc — Activation Probing Toolkit

This document defines the core classes, interfaces, and workflows for the activation probing toolkit. It is written to guide initial implementation and testing.

1. Overview

We will create small, testable components:

- `ModelBackend` — abstraction over a deep learning framework (start with PyTorch/HF).
- `ActivationExtractor` — registers hooks and extracts activations; supports layer selection and aggregation strategies.
- `DatasetRunner` — runs inference on datasets and writes activations to disk in a streaming fashion.
- `Storage` — pluggable storage interface (local NPZ/HDF5, cloud) for saved activations and metadata.

2. Component interfaces (sketch)

- `ModelBackend` (protocol)
  - load_model(model_name: str, device: str = 'cpu') -> BackendModel
  - tokenizer for pre/post-processing

- `ActivationExtractor`
  - init(model: BackendModel, layers: Sequence[LayerSelector], strategy: ExtractionStrategy)
  - extract(inputs: Batch) -> ActivationBatch
  - context manager to register/unregister hooks

- `DatasetRunner`
  - run(dataset: Iterable[Example], batch_size: int, extractor: ActivationExtractor, storage: Storage)

- `Storage`
  - open_run(run_id) -> RunWriter
  - RunWriter.append(batch_id, activations, metadata)

3. Extraction strategies

ExtractionStrategy is a small abstraction. Built-ins:

- `LastTokenStrategy` — select last non-padding token representation
- `MeanStrategy` — mean across token dimension
- `MaxStrategy` — max across token dimension
- `CLSStrategy` — select CLS token / `pooler_output` if available
- `CustomStrategy` — user-provided function `np.ndarray -> np.ndarray`

4. Hooking into models

For PyTorch/HF models, use `register_forward_hook` on `nn.Module` layers. The `ActivationExtractor` must:

- Accept layers by index, module name, or direct module reference.
- Store activations in a buffer keyed by example indices. For batch extraction, store batched activations.
- Support multiple layers in one run.

5. Dataflow for a run

1. Load model and tokenizer via `ModelBackend`.
2. Create `ActivationExtractor` with target layers and strategy.
3. Initialize `DatasetRunner` with dataset, extractor, and storage.
4. `DatasetRunner` batches inputs, calls `extractor.extract(batch)` and writes results into `storage`.

6. Testing plan (TDD)

Unit tests first. Example tests:

- `tests/test_activation_extractor.py`:
  - Create a tiny synthetic PyTorch model (2-layer transformer-like MLP) and run `ActivationExtractor` to confirm activations are captured for `last`, `mean`, `max` strategies.
  - Test layer selection by index and name.

- `tests/test_dataset_runner.py`:
  - Use a synthetic dataset and model; run the `DatasetRunner` and verify the storage contains activation arrays with correct shapes and metadata.

- `tests/test_storage.py`:
  - Test NPZ/HDF5 writing and reading round-trip for small activation arrays.

7. Edge cases and failure modes

- Empty dataset — runner should no-op and create an empty run record.
- Model without pooler/CLS — `CLSStrategy` should fallback to first token or raise a clear error.
- Mixed-length sequences — ensure token aggregation respects attention masks.
- GPU/CPU device mismatch — detect and raise helpful error.

8. Collaboration and contribution

- Use `CONTRIBUTING.md` to specify coding standards, branch naming, test requirements, and PR checks (unit tests pass, type checks).
- Prefer small, focused PRs: one core tool or strategy per PR.

9. Short example (pseudo-code)

```
from activations import ModelBackend, ActivationExtractor, DatasetRunner, NPZStorage, MeanStrategy

backend = ModelBackend()
model = backend.load_model('distilbert-base-uncased', device='cuda:0')
extractor = ActivationExtractor(model, layers=[-1], strategy=MeanStrategy())
runner = DatasetRunner(dataset, batch_size=32, extractor=extractor, storage=NPZStorage('out/'))
runner.run()
```

10. Notes on future work

- Add JAX/Flax backend.
- Add visualization tools and interactive notebooks.
- Add experiments registry and reproducibility tooling (MLFlow or simple JSON manifests).
