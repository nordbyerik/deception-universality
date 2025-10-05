# Running Experiments

## Quick Start

### 1. Test the Setup
```bash
./test_run.sh
```
Runs a minimal test experiment to verify everything works.

### 2. Run Basic Experiments
```bash
./run_experiments.sh
```
Runs experiments across 3 models with the full training dataset.

### 3. Run Advanced Experiments
```bash
./run_experiments_advanced.sh
```
Runs comprehensive experiments with multiple dataset combinations.

## Manual Experiments

### Single Experiment
```bash
uv run python src/main_nnsight.py \
    --model "Qwen/Qwen3-0.6B" \
    --train-datasets repe_honesty ai_audit werewolf \
    --test-dataset roleplaying \
    --output results/experiment1.csv
```

### Multiple Datasets
```bash
# Train on AI liar datasets, test on werewolf
uv run python src/main_nnsight.py \
    --model "meta-llama/Llama-3.2-1B" \
    --train-datasets ai_liar ai_liar_with \
    --test-dataset werewolf \
    --target-samples 150 \
    --output results/ai_liar_to_werewolf.csv
```

### Different Models
```bash
# Supported models (examples):
uv run python src/main_nnsight.py --model "Qwen/Qwen3-0.6B" ...
uv run python src/main_nnsight.py --model "meta-llama/Llama-3.2-1B" ...
uv run python src/main_nnsight.py --model "google/gemma-2-2b" ...
```

## Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | str | `Qwen/Qwen3-0.6B` | HuggingFace model name |
| `--train-datasets` | list | Multiple | Space-separated training datasets |
| `--test-dataset` | str | `roleplaying` | Test dataset name |
| `--target-samples` | int | 100 | Samples per dataset for balancing |
| `--max-layers` | int | 4 | Number of layers to probe |
| `--batch-size` | int | 4 | Batch size for extraction |
| `--output` | str | `results_nnsight.csv` | Output file path |

## Available Datasets

- `repe_honesty` - REPE honesty dataset
- `ai_audit` - AI audit dataset  
- `werewolf` - Werewolf game dataset
- `ai_liar` - AI liar dataset (without answers)
- `ai_liar_with` - AI liar dataset (with answers)
- `roleplaying` - Roleplaying dataset (off-policy)
- `roleplaying_plain` - Roleplaying dataset (plain)

## Output Files

Results are saved as CSV files with the following columns:
- `train_datasets` - Training datasets used (joined with +)
- `test_dataset` - Test dataset name
- `layer` - Layer index
- `probe_type` - Type of probe (logistic)
- `accuracy` - Classification accuracy
- `precision` - Precision score
- `recall` - Recall score
- `f1` - F1 score
- `auroc` - Area under ROC curve
- `confidence` - Average prediction confidence
- `num_samples` - Number of test samples

## Automated Scripts

### `run_experiments.sh`
- Installs uv if needed
- Syncs dependencies
- Runs 3 models with same dataset config
- Saves timestamped results

### `run_experiments_advanced.sh`
- Runs multiple dataset configurations
- Tests cross-dataset generalization
- Generates combined summary CSV

### `test_run.sh`
- Quick validation test
- Uses minimal resources
- Good for debugging

## Tips

1. **GPU Memory**: Adjust `--batch-size` if you run out of memory
2. **Speed**: Reduce `--max-layers` for faster experiments
3. **Coverage**: Increase `--target-samples` for better training
4. **Debugging**: Use `test_run.sh` first to validate setup
5. **Results**: Check `experiment_results/` for all outputs

## Troubleshooting

### CUDA Errors
If you see CUDA errors, the script will fall back to CPU automatically.

### Memory Issues
Reduce batch size:
```bash
uv run python src/main_nnsight.py --batch-size 2 ...
```

### Missing Dependencies
```bash
uv sync
```

### Model Not Found
Ensure the model name is correct on HuggingFace Hub.
