# LoRA Fine-Tuning on MNIST

A from-scratch implementation of **Low-Rank Adaptation (LoRA)** applied to a neural network for image classification. This notebook demonstrates how to efficiently fine-tune a pre-trained model on a new task while keeping the vast majority of original weights frozen — achieving strong performance with a tiny fraction of trainable parameters.

---

## Overview

LoRA is a parameter-efficient fine-tuning (PEFT) technique that injects trainable low-rank decomposition matrices into existing weight layers instead of updating all model parameters. This notebook walks through the complete workflow:

1. Train a baseline neural network on the full MNIST digit classification task
2. Inject LoRA parametrization into the network's linear layers
3. Freeze all original weights
4. Fine-tune **only the LoRA parameters** on a single digit (digit `9`)
5. Verify that original weights remain unchanged and compare performance with LoRA enabled vs. disabled

---

## How LoRA Works

For a weight matrix **W**, LoRA introduces two low-rank matrices **A** and **B** such that the effective weight becomes:

```
W_effective = W + (B @ A) * scale
```

where `scale = alpha / rank`. Since `rank << min(features_in, features_out)`, the number of new trainable parameters is dramatically smaller than the original weight matrix.

The original weights are never modified — LoRA adapters can be toggled on or off at inference time.

---

## Model Architecture

A three-layer fully connected network (`RichNeuralNet`) designed to make the parameter savings from LoRA clearly visible:

| Layer   | Input Dim | Output Dim |
|---------|-----------|------------|
| Linear 1 | 784 (28×28) | 1,000 |
| Linear 2 | 1,000 | 2,000 |
| Linear 3 | 2,000 | 10 |

Activation: ReLU between hidden layers.

---

## Requirements

```
torch
torchvision
tqdm
matplotlib
```

Install with:

```bash
pip install torch torchvision tqdm matplotlib
```

A CUDA-capable GPU is optional but recommended. The code automatically selects `cuda` if available, otherwise falls back to `cpu`.

---

## Usage

Run the notebook cells in order:

### 1. Train the base model
The network is trained on all 10 MNIST digit classes for 1 epoch using the Adam optimizer with cross-entropy loss.

### 2. Evaluate baseline performance
The `test()` function reports overall accuracy and per-digit wrong counts.

### 3. Inject LoRA parametrization
LoRA matrices are registered on each linear layer using `torch.nn.utils.parametrize`:

```python
parametrize.register_parametrization(
    net.linear1, 'weight', linear_layer_parameterization(net.linear1, device)
)
```

### 4. Fine-tune on digit 9 only
Original weights are frozen (`requires_grad = False`). Only the LoRA `A` and `B` matrices are trained on a filtered dataset containing only digit `9` examples.

### 5. Compare LoRA on vs. off
```python
enable_disable_lora(enabled=True)   # Uses W + LoRA adaptation
enable_disable_lora(enabled=False)  # Uses original W only
```

---

## Parameter Efficiency

With `rank=1`, the LoRA adaptation introduces a small number of additional parameters relative to the original model:

| | Parameters |
|---|---|
| Original model | ~2.8M |
| LoRA parameters added | ~6,000 |
| Parameter overhead | ~0.2% |

The original weights remain completely intact and verifiable via assertions after fine-tuning.

---

## Key Implementation Details

- **`LoRAParametrization`** — Custom `nn.Module` implementing the `W + (B @ A) * scale` forward pass, with an `enabled` flag for toggling
- **`linear_layer_parameterization`** — Factory function that reads layer dimensions and returns a correctly sized `LoRAParametrization`
- **`enable_disable_lora()`** — Utility to switch LoRA on/off across all adapted layers simultaneously
- **Determinism** — `torch.manual_seed(0)` ensures reproducible results
- **Weight integrity checks** — Assertions verify that original weights are unchanged after LoRA fine-tuning

---

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [PyTorch Parametrizations Tutorial](https://pytorch.org/tutorials/intermediate/parametrizations.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
