# SPAR Usage Guide

## Getting Started

### 1. Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Simple Qwen Test

```bash
source venv/bin/activate
python qwen_simple.py
```

## What the Script Does

The `qwen_simple.py` script demonstrates:

1. **Model Loading**: Loads Qwen2.5-0.5B model and tokenizer
2. **Basic Inference**: Tests text generation with a simple prompt
3. **Hidden State Extraction**: Extracts activations from all 25 model layers
   - Shape: (896,) per layer (hidden dimension)
   - Foundation for probe training

## Key Features

- Uses Qwen2.5-0.5B (closest to planned 0.6B)
- Extracts last token representations from all layers
- CPU/GPU compatible with device_map="auto"
- Foundation for deception probe research

## Next Steps

This script provides the foundation for:
- Training linear probes on deception datasets
- Comparing representations across model layers
- Expanding to multiple model sizes (1.7B, 4B, 8B, etc.)
- Building the full deception detection pipeline

## Output

The script successfully:
- ✓ Loads model and tokenizer
- ✓ Generates text samples
- ✓ Extracts hidden states from 25 layers
- ✓ Returns 896-dimensional representations per layer