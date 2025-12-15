# NVIDIA Model Optimizer (submodule)

This directory vendors NVIDIA's **Model Optimizer** (**ModelOpt**) repository as a git submodule.

## What it is

**ModelOpt** is a unified library of state-of-the-art model optimization techniques for **faster / cheaper inference**, including:

- **Post-training quantization (PTQ)** and **quantization-aware training (QAT)**
- **Pruning**
- **Distillation**
- **Speculative decoding**
- **Sparsity**

It supports **Hugging Face**, **PyTorch**, and **ONNX** model inputs, and can export optimized checkpoints for deployment stacks like **TensorRT-LLM**, **TensorRT**, **vLLM**, and **SGLang**.

## Why it’s in this repo

This repo is a grab-bag of “real” ML-systems codebases (training + inference). ModelOpt is included here as a reference implementation of **inference-time optimization workflows**, complementary to training-centric repos like Megatron-LM and Miles.

## Quick start

### Initialize / update the submodule

```bash
git submodule update --init --recursive examples/model-optimizer/Model-Optimizer
```

### Install (pip)

```bash
python -m pip install -U "nvidia-modelopt[all]"
```

### Browse upstream docs / examples

- Docs: `https://nvidia.github.io/Model-Optimizer`
- Examples: `examples/model-optimizer/Model-Optimizer/examples/`
- Upstream repo: `https://github.com/NVIDIA/Model-Optimizer`

---

- Upstream path: `examples/model-optimizer/Model-Optimizer`

