#!/bin/bash

set -e

echo "=================================="
echo "SPAR Experiment Runner"
echo "=================================="

if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "uv is already installed"
fi

echo "Installing project dependencies..."
uv sync

echo "=================================="
echo "Starting experiments..."
echo "=================================="

MODELS=(
    "Qwen/Qwen3-0.6B"
    "meta-llama/Llama-3.2-1B"
    "google/gemma-2-2b"
)

TRAIN_DATASETS="repe_honesty ai_audit werewolf ai_liar ai_liar_with"
TEST_DATASET="roleplaying"

RESULTS_DIR="./experiment_results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for MODEL in "${MODELS[@]}"; do
    MODEL_SAFE=$(echo "$MODEL" | sed 's/\//_/g')
    OUTPUT_FILE="${RESULTS_DIR}/results_${MODEL_SAFE}_${TIMESTAMP}.csv"
    
    echo ""
    echo "=================================="
    echo "Running experiment with model: $MODEL"
    echo "Output: $OUTPUT_FILE"
    echo "=================================="
    
    uv run python src/main_nnsight.py \
        --model "$MODEL" \
        --train-datasets $TRAIN_DATASETS \
        --test-dataset "$TEST_DATASET" \
        --target-samples 100 \
        --max-layers 4 \
        --batch-size 4 \
        --output "$OUTPUT_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✓ Experiment completed successfully for $MODEL"
    else
        echo "✗ Experiment failed for $MODEL"
    fi
done

echo ""
echo "=================================="
echo "All experiments complete!"
echo "Results saved in: $RESULTS_DIR"
echo "=================================="
