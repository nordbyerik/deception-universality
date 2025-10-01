# Dataset Summary

This document summarizes the datasets available in this repository and explains how to load them.

## Available Datasets

The following datasets are available. Each dataset may have multiple "variants" that affect how the data is processed.

*   **ai_audit**: 
    *   variants: `a1_2`, `a1_2_reasoning`, `no_reasoning`, `reasoning`
*   **ai_liar**: 
    *   variants: `original_with_answers`, `original_without_answers`, `mc`, `normal_instruction`, `deceive_instruction`
*   **alpaca**: 
    *   variants: `plain`
*   **insider_trading**: 
    *   variants: `prewritten`, `onpolicy`, `upscale`
*   **insider_trading_doubledown**: 
    *   variants: `onpolicy`, `upscale`
*   **repe**: 
    *   variants: `plain`
*   **roleplaying**: 
    *   variants: `plain`
*   **sandbagging_v2**: 
    *   variants: `wmdp_mmlu`
*   **werewolf**: 
    *   variants: `simple`, `paired`, `open_response`
*   **geometry_of_truth**: A collection of datasets testing knowledge of geometry and truth.
    *   variants: `cities`, `cities_cities_conj`, `cities_cities_disj`, `common_claim_true_false`, `companies_true_false`, `counterfact_true_false`, `larger_than`, `neg_cities`, `neg_sp_en_trans`, `smaller_than`, `sp_en_trans`
*   **internal_state**: Datasets from internal state introspection.
    *   variants: `animals_true_false`, `cities_true_false`, `companies_true_false`, `elements_true_false`, `facts_true_false`, `inventions_true_false`
*   **train**: Training datasets.
    *   variants: `descriptive_qa`

## Loading Datasets

To load a dataset, use the `deception_detection.repository.DatasetRepository` class.

Here is an example of how to load the `ai_liar` dataset with the `original_with_answers` variant:

```python
from deception_detection.repository import DatasetRepository

# Instantiate the repository
repo = DatasetRepository()

# Get the dataset
# The first argument is a "partial_id" of the form "<base_name>__<variant>"
# The model parameter defaults to "prewritten" for datasets that are not generated from model rollouts.
dataset = repo.get("ai_liar__original_with_answers", model="prewritten")

# You can now access the dialogues and labels
print(f"Number of dialogues: {len(dataset.dialogues)}")
print(f"Number of labels: {len(dataset.labels)}")
```

### Loading Rollout Datasets

Some datasets are generated from model rollouts. To load these, you need to specify the model name in the `model` parameter. For example:

```python
from deception_detection.repository import DatasetRepository

repo = DatasetRepository()

# Load a rollout dataset
dataset = repo.get("insider_trading__upscale", model="llama-70b-3.3")
```

The available rollout datasets and their corresponding models can be found in `dataset_paths.json`.
