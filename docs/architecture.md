# Architecture — Activation Probing Toolkit

This file summarizes the project architecture and recommended file layout.

Proposed layout:

```
SPAR/
├── README.md
├── requirements.txt
├── CONTRIBUTING.md
├── docs/
│   ├── requirements.md
│   ├── agent_design.md
│   └── architecture.md
├── src/
│   └── activations/
│       ├── __init__.py
│       ├── backend.py
│       ├── extractor.py
│       ├── strategies.py
│       ├── runner.py
│       └── storage.py
└── tests/
    ├── test_activation_extractor.py
    └── test_dataset_runner.py
```

Core modules description:

- `backend.py` — model and tokenizer loading abstractions.
- `extractor.py` — hook management, activation collection, and extraction API.
- `strategies.py` — token aggregation strategies implementations.
- `runner.py` — dataset iteration, batching, GPU device management, and coordination with storage.
- `storage.py` — read/write activation storage backends.

Roadmap (initial):

1. Implement `backend.py` for HF/PyTorch.
2. Implement `strategies.py` and unit tests.
3. Implement minimal `extractor.py` that uses `register_forward_hook` and tests with synthetic model.
4. Implement `runner.py` and `storage.py` (NPZ writer), integration tests.
5. Add CI, formatters, type checks, and docs website.
