# lfrl-comparison
Label-Free Intrinsic Rewards for Reasoning in Small Language Models
# Label-Free Intrinsic Rewards for Reasoning in Small Language Models

> Comparison of Self-Certainty, Entropy Minimization, and Majority Voting as label-free RLVR signals for small LLMs (≤1.5B parameters).

## Overview

This repository contains the full experimental code for the paper:
**"Label-Free Intrinsic Rewards for Reasoning in Small Language Models: A Cross-Domain Comparison"**
— Rhonald Jose Torres-Diaz, Universidad Nacional de Colombia, Sede La Paz.

## Requirements

- Python 3.11
- PyTorch 2.6.0 + CUDA 12.4
- Unsloth 2026.5.6
- trl 0.24.0

```bash
pip install unsloth trl bitsandbytes
```

## Usage

### Training
```bash
# Single run
python train.py --signal self_certainty --domain math --max_samples 500

# All 9 experiments
run_experiments.bat
```

### Evaluation
```bash
python evaluate.py --model_path runs/<run_name>/final_model --domain math --max_samples 50 --k 5
python evaluate.py --table
```

## Signals
| Signal | Description |
|---|---|
| `self_certainty` | KL divergence between output distribution and uniform |
| `entropy` | Negative mean token-level Shannon entropy |
| `majority_voting` | Fraction of N=8 samples that agree |

## Results
| Signal | Domain | Pass@1 | Pass@5 |
|---|---|---|---|
| Baseline | Math | 40.0% | 40.0% |
| Self-certainty | Math | 20.0% | 48.0% |
| Entropy | Code | **50.0%** | 74.0% |
| Majority voting | Logic | **76.0%** | 80.0% |

## Citation
```bibtex
@article{torres2026labelfree,
  title={Label-Free Intrinsic Rewards for Reasoning in Small Language Models},
  author={Torres-Diaz, Rhonald Jose},
  year={2026}
}
```
