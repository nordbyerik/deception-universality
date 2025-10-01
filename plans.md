# SPAR Project Plans

## Overview

SPAR (Transformer Activation Probing Toolkit) serves as both a toolkit for probing transformer activations and a research platform for investigating the universality of deception probes across model families and sizes.

## Toolkit Goals

- Provide small, well-tested core tools for extracting activations from transformer models (PyTorch/Hugging Face compatible)
- Make it easy to select extraction strategies (last token, max, mean, pooled, custom hooks)
- Provide dataset runner utilities to batch inputs, run models, and store activations
- Use Test Driven Development for core components and provide clear contribution guidelines

## Research Project: Universality of Deception Probes

### Executive Summary

This project systematically evaluates the efficacy of linear probes for detecting deception across various model families and sizes. While linear probes have shown success across many tasks (domain detection, truthfulness, etc.), there has not been a systematic exploration of whether deception is universally represented linearly across all model families and sizes.

### Primary Objectives

1. **Universal Detection**: Determine if deception is represented linearly across all models and can be detected with probes
2. **Cross-Model Comparison**: Systematically compare the efficacy of linear probes across model families and types

### Secondary Objectives

- Compare deception representation across model sizes
- Analyze where in models deception is initially represented (early vs late layers)
- Compare efficacy across different model architectures (Instruct, Reasoning, MoE)
- Evaluate per-token vs general classification performance
- Compare linear vs non-linear representation universality

## Implementation Plan

### MVP (Minimum Viable Product)

**Dataset**: Single dataset proven useful in previous research (e.g., Sandbagging by Benton et al.)

**Models**: Qwen 3 family - 4 models from: 0.6B, 1.7B, 4B, 8B, 14B, 32B

**Methodology**:
- Use probity to train probes for deceptive vs non-deceptive classification
- Test across subset of layers for each model
- Compare metrics across:
  - Variously sized dataset subsets (learning curves)
  - Model sizes
  - Various layers to identify when deception representations emerge

### Future Expansion

**Datasets**:
- All datasets from "Black-to-White Performance Boosts" paper
- Automated pipeline creation for additional datasets
- Flexible environments (e.g., Concordia by DeepMind)

**Models**:
- Llama (3.1, 3.2, 3.3 comparison)
- Gemma variants (ShieldGemma, MedGemma, VaultGemma)
- Deepseek (R1, R1-Zero, V3)
- Deepseek Distilled Models (1.5B, 7B, 14B, 32B)
- GPT OSS (20B, 120B)
- Phi-4 variants (reasoning, reasoning plus, default)

**Advanced Methodology**:
- Multi-dataset training with held-out evaluation
- Cross-model probe training and evaluation
- Multi-class/hierarchical probes for deception types
- Token-level analysis
- Comparison with other interpretability techniques (SAEs, causal tracing)

### Metrics & Evaluation

- AU-ROC
- TPR@1% FPR
- Accuracy
- F1 Score/Precision/Recall
- Calibration Metrics (Brier score)

## Timeline & Milestones

| Task | Start Date | Target Completion |
|------|------------|-------------------|
| Stand Up Initial Probity Script for Single Dataset | Sept 22 | Sept 28 |
| Review Literature | Sept 22 | Oct 4 |
| Expand Script to Additional Datasets | Sept 28 | Oct 11 |
| Add Support for Additional Models | Sept 28 | Oct 11 |
| Expand to Novel Datasets | Oct 11 | Oct 25 |
| Evaluate Progress & Decide Experimental Directions | Oct 25 | End of SPAR |
| Venue Shopping & Communication | Oct 11 | End of SPAR |

## Risk Assessment

| Failure Mode | Likelihood | Mitigation Strategy |
|--------------|------------|-------------------|
| Too Long to Publish (research overlap) | Unknown | Adjust scope to address related work |
| Poor Dataset Curation | ~30% | Robust & diverse deceptive scenario datasets |
| Loss of Time/Motivation | ~10% | Clear communication about constraints |

## Early Stopping Conditions

1. **Probes incapable of detecting deception**: Full group meeting to decide next steps
2. **Dataset curation too complex**: Use previously curated datasets from other papers
3. **Another paper addresses this question**: Reevaluate goals and next steps

## Path to Impact

- Provide solid pipeline for future project iterations
- Publish pre-print on arXiv
- Post results to Alignment Forum
- Pivot to successful project components if applicable

## Key References

### Deception Probes
- Detecting Strategic Deception Using Linear Probes (Apollo)
- Toward universal steering and monitoring of AI models
- Benchmarking Deception Probes via Black-to-White Performance Boosts

### General Probing Background
- Understanding intermediate layers using linear classifier probes
- The Internal State of an LLM Knows when It's Lying
- The geometry of truth: Emergent linear structure in large language model representations

### Deception & LLMs
- Large Language Models can Strategically Deceive their Users when Put Under Pressure
- How to Catch an AI Liar: Lie Detection in Black-Box LLMs by Asking Unrelated Questions
- AI Deception: A Survey of Examples, Risks, and Potential Solutions

## Next Steps (Toolkit Development)

1. Implement `src/activations` core modules and tests
2. Add CI to run tests and linting
3. Add example experiments and notebooks

*See `docs/` for detailed requirements and design notes.*