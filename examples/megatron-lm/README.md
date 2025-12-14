# Megatron-LM (submodule)

This directory vendors NVIDIA's **Megatron-LM** repository as a git submodule.

## What it is

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) is NVIDIA’s GPU-optimized codebase for training transformer models at scale. It includes:

- **Megatron Core**: a composable library of optimized building blocks (parallelism, kernels, transformer components) under `megatron/core`.
- **Reference training implementations + scripts**: end-to-end examples and utilities for large-scale training runs (under `examples/`, plus top-level `pretrain_*.py` entrypoints).

## Why it's in this repo

This repo includes Megatron-LM so example systems (like Miles) can use it as a **training backend**. In particular, when Miles is configured to use the Megatron backend, Megatron is where training-side parallelism is implemented (e.g., **TP/PP/CP/EP**), while rollout/inference is typically handled by SGLang.

### Note: FSDP (AG/RS) is usually a training optimization, not an inference optimization

It’s easy to get confused by phrasing like “AG/RS vs NCCL”. **All-Gather (AG)** and **Reduce-Scatter (RS)** are *collective communication patterns* that are typically implemented by **NCCL** on NVIDIA GPUs.

- **Training (where FSDP helps)**: FSDP/ZeRO-style sharding saves memory by sharding **parameters / gradients / optimizer state**, then using **AG** (to materialize weights) and **RS** (to shard-reduce gradients). With backward compute, some of this communication can be overlapped.
- **Inference (where FSDP usually hurts)**: there is no backward pass, and per-token compute is smaller, so FSDP-style sharding often turns into frequent **weight all-gathers** that are hard to hide and can increase latency. Inference stacks (e.g., vLLM) usually prefer **TP/PP with weights kept resident** (plus KV-cache optimizations) rather than FSDP.

### Primer: Megatron-FSDP vs “regular” PyTorch FSDP2 (in this repo)

Both approaches implement the same *idea*: **ZeRO/FSDP-style sharding** across a data-parallel sharding group. Conceptually:

```text
Forward:  (need weights)  -> All-Gather (AG) parameter shards -> compute -> (optional) reshard/free
Backward: (need weights)  -> (maybe AG again) -> compute -> Reduce-Scatter (RS) gradient shards
Optimizer: update local shards of parameters / optimizer state
```

Where they differ is mostly **integration/API + performance engineering**:

- **PyTorch FSDP2** (`--use-torch-fsdp2`)
  - **What it is**: Torch’s native “FSDP2 / DTensor” path (`torch.distributed.fsdp`).
  - **What you do**: typically call `fully_shard(...)` on each block (and the root module) or rely on a wrapper to do so.
  - **Caveat in Megatron-LM**: in this vendored tree, Megatron-LM explicitly notes FSDP2 “has not been tested with pipeline parallelism, and may contain bugs” (so treat it as more experimental here).

- **Megatron-FSDP** (`--use-megatron-fsdp` + `--data-parallel-sharding-strategy {no_shard,optim,optim_grads,optim_grads_params}`)
  - **What it is**: NVIDIA’s Megatron-integrated FSDP implementation (packaged as `megatron-fsdp`) designed to compose cleanly with Megatron Core + Transformer Engine.
  - **Performance “why”**: focuses on reducing overhead and improving overlap via things like **dtype-aware bucketing**, **zero-copy-ish param/grad buffer layouts**, better **AG/RS overlap**, and optional comm offload/SM-reduction knobs (e.g., SHARP / NCCL user-buffer registration) when available.
  - **How you pick sharding level**: the `--data-parallel-sharding-strategy` flag is essentially “how ZeRO-ish do you want DP to be” (from `no_shard` ~ DDP-like up to `optim_grads_params` ~ ZeRO-3).

Rule of thumb for *this* repo:

- If you’re running Megatron-LM training and want “FSDP-style sharding”, start with **Megatron-FSDP** (`--use-megatron-fsdp`) unless you have a specific reason to test Torch FSDP2.
- For inference/serving (e.g., vLLM), prefer **TP/PP with resident weights** rather than FSDP-style sharding.

- Upstream: `https://github.com/NVIDIA/Megatron-LM`
- Path: `examples/megatron-lm/megatron-lm`

## Megatron Core `tensor_parallel` (TP) (what it is)

Megatron Core’s **Tensor Parallelism (TP)** shards *individual transformer layers* across a small group of GPUs. Instead of replicating a layer’s biggest weight matrices on every GPU, TP **splits those tensors by dimension** so each TP rank owns a slice, does the corresponding slice of compute, and then uses fast collectives to assemble the result (or to assemble gradients).

- **What TP is (conceptually)**: *intra-layer* model parallelism. You can think of it as “split the GEMMs” (and the embedding/softmax) so one layer spans multiple GPUs, while the model’s layer stack still looks like one logical network.
- **Why it’s used**: enables training when a single GPU can’t fit the model (weight/optimizer/activation memory), and often improves throughput by increasing total math bandwidth—at the cost of **extra communication inside each layer**.
- **Column-parallel vs row-parallel linears (the core pattern)**:
  - **Column-parallel linear**: split the weight by output features. Each rank produces a partial output; optionally **all-gather** to materialize the full hidden state, or keep it sharded for subsequent ops.
  - **Row-parallel linear**: split the weight by input features. Each rank produces a partial contribution to the same output features; forward typically needs an **all-reduce** (sum) to combine contributions.
  Together, these two patterns let you shard MLPs and attention projections while keeping the math identical to a non-sharded layer.
- **Vocab-parallel embedding + loss**: the embedding table (and output logits / softmax) can be sharded over the vocabulary so ranks only store/compute their vocab slice. Cross-entropy is computed via **numerically-stable global reductions** (e.g., global max and sum-exp across TP ranks) without ever needing a full, replicated vocab on one GPU.
- **Communication is “by design”**: TP works by carefully placing **all-reduce / reduce-scatter / all-gather** at the boundaries where sharded tensors must be combined (either to form activations for the next op, or to aggregate gradients).
- **How it composes**: TP typically combines with **Data Parallel (DP)** (replicate the TP group across data shards) and often with **Pipeline Parallel (PP)** (split layers into stages, with TP inside each stage). It can also combine with other modes like context/sequence parallelism and expert parallelism in MoE setups.

Reference: Megatron Core tensor-parallel API docs: `https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/tensor_parallel.html`

## Megatron Core `datasets` package (what it does)

Megatron Core’s `datasets` package is the **data pipeline layer** that connects **tokenized data on disk** to **distributed PyTorch datasets/data loaders** used by training runs:

- **Preprocessing format**: defines Megatron’s low-level on-disk dataset (`IndexedDataset`) and how to build it (`IndexedDatasetBuilder`). Data lives in a `.bin` (tokens) + `.idx` (metadata) pair.
- **Distributed dataset construction**: provides configs/builders (`BlendedMegatronDatasetConfig`, `BlendedMegatronDatasetBuilder`) that all ranks use to deterministically build the same train/valid/test datasets without deadlocks/hangs.
- **Training-ready datasets**: implements higher-level dataset abstractions (`MegatronDataset` and concrete ones like `GPTDataset`) that turn documents/sequences into fixed-length training samples via cached index mappings (document/sample/shuffle indices).
- **Dataset blending**: `BlendedDataset` mixes multiple datasets according to weights so a split can be a controlled blend of sources.

### Visual guide (what it looks like)

```text
┌──────────────────────────┐
│ Raw text / JSONL / etc.  │
└─────────────┬────────────┘
              │ tokenize (your code)
              v
┌──────────────────────────┐
│ token ids per document   │
└─────────────┬────────────┘
              │ build
              v
┌───────────────────────────────────────────────────────────┐
│ IndexedDatasetBuilder                                     │
└─────────────┬─────────────────────────────────────────────┘
              │ writes (on disk)
              v
      ┌─────────────────┐        ┌──────────────────────────┐
      │ dataset.bin      │        │ dataset.idx              │
      │ token payload    │        │ metadata (offsets,       │
      │ (contiguous)     │        │ lengths, doc boundaries) │
      └─────────────────┘        └──────────────────────────┘


                    (runtime / training)
┌───────────────────────────────────────────────────────────┐
│ All ranks: BlendedMegatronDatasetBuilder + Config          │
└─────────────┬─────────────────────────────────────────────┘
              │ builds splits + (optional) blend weights
              v
     ┌───────────────────────┐        ┌───────────────────────┐
     │ MegatronDataset (base) │  ...   │ MegatronDataset (base) │
     └──────────┬────────────┘        └──────────┬────────────┘
                │                                  │
                v                                  v
        ┌────────────────┐                 ┌────────────────┐
        │ GPTDataset      │                 │ GPTDataset      │
        │ (or BERT/T5...) │                 │ (or BERT/T5...) │
        └───────┬────────┘                 └───────┬────────┘
                │                                  │
                └──────────────┬───────────────────┘
                               v
                      ┌───────────────────┐
                      │ BlendedDataset     │
                      │ (mix by weights)   │
                      └─────────┬─────────┘
                                v
                      ┌───────────────────┐
                      │ PyTorch DataLoader │
                      └─────────┬─────────┘
                                v
                      ┌───────────────────┐
                      │ training loop      │
                      └───────────────────┘


                (inside GPTDataset: how one sample is formed)
┌───────────────────────────────────────────────────────────┐
│ IndexedDataset: doc0 doc1 doc2 ...                         │
└───────────────┬───────────────────────────────────────────┘
                │ builds/caches index mappings
                v
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │ document index    │   │ sample index      │   │ shuffle index     │
   │ Do_idx            │   │ Sa_idx            │   │ Sh_idx            │
   │ docs repeated+    │   │ (doc_i, offset)   │   │ permutes samples  │
   │ shuffled by seed  │   │ bounds per sample │   │ each epoch/seed   │
   └─────────┬────────┘   └─────────┬────────┘   └─────────┬────────┘
             │                      │                      │
             └──────────────┬───────┴──────────────┬───────┘
                            v                      v
                     select doc spans      pack into fixed-length
                     across boundaries     sequence S tokens
```

Reference: Megatron Core datasets API docs: `https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/datasets.html`

## Megatron Core MoE package (Mixture of Experts) (what it does)

Megatron Core’s **Mixture of Experts (MoE)** package provides production-oriented MoE building blocks that integrate **Expert Parallelism (EP)** with Megatron’s other parallel modes (**TP/DP/PP**, plus **CP** for long context). It supports common MoE architectures (e.g., Mixtral-style) and includes:

- **Parallelism + mappings**: EP can be combined with TP/DP/PP/CP; when using EP+TP, **sequence parallelism is required**. It also supports **MoE parallel folding** to decouple the MoE parallel groups from dense/attention groups, enabling more flexible sharding (including setting a MoE-specific TP via `--expert-tensor-parallel-size`).
- **Routing + load balancing**: Top-\(k\) MLP router with multiple load-balancing strategies (aux-loss, Sinkhorn, or none), plus CUDA-fused routing/load-balancing kernels.
- **Token dispatch**: Supports dropless (no drop) and capacity-based token dropping/padding, with dispatcher choices such as **allgather** (often good without EP) and **alltoall** (recommended with EP).
- **Performance optimizations**: GroupedGEMM for multiple local experts, token permutation/unpermutation fusion, communication overlap knobs (e.g., TP comm overlap), and (experimental) **DeepEP** for more efficient cross-node token dispatch via the flex dispatcher.
- **Checkpointing**: MoE supports Megatron Core’s **distributed checkpointing** (`--ckpt-format torch_dist`, optional `--auto-detect-ckpt-format`) for flexible save/load across parallel mappings.

Reference: Megatron Core MoE API docs: `https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html`

## Setup

If you cloned this repo without submodules:

```bash
git submodule update --init --recursive
```


