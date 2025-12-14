# MILES Examples

**MILES** is an enterprise-facing reinforcement learning framework for large-scale MoE (Mixture of Experts) post-training and production workloads. Forked from [slime](https://github.com/THUDM/slime), MILES focuses on new hardware support (e.g., GB300), stable RL for large MoE models, and production-grade features.

## Q&A (Core Rollout / “Phase 1” knobs)

### What do `--rollout-batch-size` and `--n-samples-per-prompt` mean?

- **`--rollout-batch-size`**: Defines the **number of Prompts** for each round of sampling (one rollout step).
- **`--n-samples-per-prompt`**: Defines the **number of responses generated for each Prompt** (used for GRPO-like algorithms).

In Miles, it helps to separate **prompt** vs **sample**:

- **Prompt**: one input item pulled from your prompt dataset (one “question” / “conversation” entry).
- **Sample**: one generated response trajectory for a prompt (tokens + rewards + logprobs + metadata).

So in one rollout step:

- **Total generated samples (trajectories)**:
  `num_samples_per_rollout = rollout_batch_size * n_samples_per_prompt`
- **How it’s used by GRPO-style algorithms**: for each prompt, Miles collects a *group* of `n_samples_per_prompt` responses, which enables per-prompt/group statistics (e.g., relative advantage within the group). Setting this to 1 effectively disables “grouped” behavior.

Finally, Miles enforces that rollout output matches training consumption per rollout step:

`rollout_batch_size * n_samples_per_prompt = global_batch_size * num_steps_per_rollout`

This is why these two flags are “core to phase 1”: they define how much data the rollout subsystem produces each iteration, and everything else (training update cadence, batching, throughput) is constrained around that.

### What does `--context-parallel-size` (CP) mean? (visual intuition)

**CP = “split the token positions of each sequence across multiple GPUs”.** In other words, instead of every GPU holding activations for the full context length, a **context-parallel group** of size `CP` shares the sequence length so each GPU only holds a fraction of tokens.

This is different from other parallelism axes:

- **DP** splits *samples* (different examples on different ranks)
- **TP** splits *tensor/hidden dims* (weight shards)
- **PP** splits *layers*
- **EP** splits *experts* (MoE)
- **CP** splits *token positions within the same sample*

#### What it looks like in Miles (important detail)

When CP is enabled, Miles uses a **“two-chunk mirrored” layout** per sequence:

- The full sequence of length `T` is divided into **`2 * CP` chunks** (equal-ish, with padding).
- Each CP rank gets **two chunks**:
  - one chunk from the front
  - one chunk from the back (mirrored)

So with **CP4**, there are **8 chunks** total:

```
original token positions (0..T-1), split into 8 chunks:
chunk0  chunk1  chunk2  chunk3  chunk4  chunk5  chunk6  chunk7

CP rank 0 holds: chunk0 + chunk7
CP rank 1 holds: chunk1 + chunk6
CP rank 2 holds: chunk2 + chunk5
CP rank 3 holds: chunk3 + chunk4
```

Each rank stores its two chunks **concatenated** (front chunk then back chunk). This means the CP-local token stream is generally **not contiguous in the original order**; it’s a layout optimized for how Miles packs sequences and reconstructs full ordering when needed.

#### Why this helps (and what you pay)

- **Benefit**: long-context training becomes feasible because per-GPU activation footprint scales roughly with **~1/CP** (each GPU holds fewer token positions).
- **Cost**: some operations require CP-group communication (e.g., temporarily reconstructing “full sequence” tensors for certain losses/metrics, and attention implementations that need whole-sequence ordering).

#### Interaction with `--use-dynamic-batch-size` / `--max-tokens-per-gpu`

When dynamic batching is enabled, Miles budgets microbatches using a token budget proportional to CP. A useful rule of thumb is:

- effective per-step token budget scales like **`max_tokens_per_gpu * CP`** (because the CP group collectively covers more token positions).

#### Note: CP (Megatron training) is not SGLang “dp attention”

You may also see rollout-side flags like `--sglang-enable-dp-attention` / `--sglang-dp-size ...`. Those are **rollout/inference-side** sharding knobs in SGLang; **`--context-parallel-size`** is a **Megatron training-side** parallelism axis used by the Miles training backend.

## Mathematical view: the core Miles RL loop

### Top-level summary

Miles is an online RL system organized as a repeated **(rollout → update → sync)** loop: an inference/rollout policy generates trajectories, rewards are computed, a training backend updates the policy parameters, and the updated parameters are synchronized back to the rollout engines for the next sampling round.

### Notation

- Prompt dataset: draw a prompt `x` from a dataset `D`
- Current policy parameters (actor): `theta_t`
- Rollout policy (served by SGLang): `pi_{theta_t}(y | x)`
- Group size: `K = --n-samples-per-prompt`
- Prompts per rollout step: `B = --rollout-batch-size`
- Samples per rollout step: `N = B * K`
- Reward function / reward model: `r(x, y)` (may include post-processing, scaling, grouping, etc.)

### Phase 1 (Rollout / sampling)

At rollout step `t`, Miles samples `B` prompts `{x_i}_{i=1..B}`. For each prompt `x_i`, it generates a **group** of `K` responses:

`y_{i,1..K} ~ pi_{theta_t}(. | x_i)` and `r_{i,k} = r(x_i, y_{i,k})`.

This yields `N = B * K` total trajectories `{(x_i, y_{i,k}, r_{i,k})}` for the update phase.

Why “groups” matter: GRPO-like objectives use *within-prompt* statistics (e.g., centered/normalized rewards within the group) to build advantages; `K` is therefore a core algorithmic knob, not just a throughput knob.

### Phase 2 (Training / update)

The training backend (Megatron or FSDP) consumes those `N` samples and performs `S = --num-steps-per-rollout` optimizer updates using a training batch size of `G = --global-batch-size` samples per update.

Miles enforces the conservation constraint:

`B * K = G * S`

so “data produced per rollout” matches “data consumed per rollout” in the default on-policy configuration (and the framework can validate/auto-fill `G` when `S` is provided).

At a high level, the update computes an advantage signal `A` (e.g., GRPO, PPO, Reinforce variants) from the rollout data, and applies a policy-gradient-style update:

`theta_{t+1} <- theta_t - eta * grad_theta E[ L(theta; x, y, A, ...) ]`

optionally with KL regularization / reference model terms, entropy terms, clipping (PPO), etc., depending on `--advantage-estimator` and related flags.

### Weight sync (closing the loop)

After each training update (or every `--update-weights-interval` rollout steps), Miles synchronizes the updated `theta_{t+1}` from the training engine back to the rollout engines so the next rollout step is sampled from the latest policy.

### System view: the Miles runtime “stack” (call flow)

In practice, the loop above is implemented as a small number of long-lived system components created once at startup, plus a repeated per-rollout execution cycle.

**Visual guide (runtime stack at a glance)**

```
You run:  python3 miles/train.py  (or train_async.py)
   |
   v
Parse + normalize args
   |
   v
Ray allocates GPU resources (placement groups)
   |
   +------------------------------+
   |                              |
   v                              v
Rollout subsystem                 Training subsystem
(RolloutManager)                  (RayTrainGroup of N ranks)
SGLang engines + router            Megatron or FSDP backend actors
   |                              |
   +--------------+---------------+
                  |
                  v
Repeated cycle (per rollout_id):
  [1] sync weights (train -> rollout)
  [2] rollout generate + reward + package batch
  [3] train update (logprobs/advantages/optimizer step)
  [4] sync weights again (or every update-weights-interval)
```

**Visual guide (colocated vs decoupled)**

```
Colocated (--colocate): ONE GPU pool, time-sliced
------------------------------------------------
   Same physical GPUs are shared by both subsystems.

   [Megatron/FSDP training ranks]  <-->  [SGLang rollout engines]
        (wake/train/offload)               (onload/generate/offload)

   Main pain point: memory handoff + OOM margins + sync timing.


Decoupled (default if not colocated): TWO GPU pools, bridged by sync
-------------------------------------------------------------------
   Training GPUs and rollout GPUs are disjoint.

   TRAIN GPU POOL                          ROLLOUT GPU POOL
   [training ranks]  --(update_weights)--> [SGLang engines]

   Main pain point: weight-sync bandwidth/latency + networking/NCCL/ports.
```

**Entrypoint + configuration**

- You launch `miles/train.py` (sync loop) or `miles/train_async.py` (pipelined loop).
- CLI args are parsed and normalized in `miles/utils/arguments.py` (this is also where debug/colocate/offload modes can rewrite the effective config).

**Cluster scheduling (Ray)**

- Miles reserves GPUs via Ray placement groups in `miles/ray/placement_group.py`.
- It then creates two “subsystems”:
  - **Rollout subsystem**: a `RolloutManager` (`miles/ray/rollout.py`) which owns SGLang engine processes (and optionally a router) and knows how to run your rollout function.
  - **Training subsystem**: an `actor_model` (and optional `critic_model`) which are `RayTrainGroup`s (`miles/ray/actor_group.py`): a group of per-rank Ray actors that execute the training backend (Megatron or FSDP).

**Per-rollout cycle (the repeating call stack)**

- **(Sync “train → rollout”)** Ensure rollout engines are using the latest actor weights. This is what closes the on-policy loop (or keeps staleness bounded if you intentionally sync less often).
- **(Rollout generation)** `RolloutManager.generate(rollout_id)` calls the configured rollout function (typically through the SGLang router) to produce `Sample`s, computes rewards, and packages the result into a training-consumable batch.
- **(Training update)** `actor_model.async_train(rollout_id, rollout_data_ref)` runs one update step (or multiple micro-steps) inside the training backend, recomputes or consumes logprobs (depending on mismatch/correction settings), computes advantages/returns, and performs optimizer updates.
- **(Periodic side effects)** save checkpoints, run eval, update ref model (if enabled), and emit metrics/traces.

**Colocated vs decoupled (what changes in the stack)**

- **Colocated (`--colocate`)**: training and rollout share the same physical GPU pool; Miles alternates which subsystem is “active” and may offload/onload weights and KV cache to stay within memory limits. The high-level cycle is still “sync → rollout → train → sync”, but the system also has to manage memory handoff between the two subsystems.
- **Decoupled (not colocated)**: training GPUs and rollout GPUs are disjoint; rollouts can proceed without contending for the training pool, and weight sync becomes the main coupling point between the two subsystems.

**Where to start reading code for the stack**

- Entrypoint loops: `miles/train.py`, `miles/train_async.py`
- Arg parsing + mode rewrites: `miles/utils/arguments.py`
- Ray orchestration: `miles/ray/placement_group.py`, `miles/ray/actor_group.py`
- Rollout manager + SGLang engine lifecycle: `miles/ray/rollout.py`
- Backend implementations (the “train step”): `miles/backends/megatron_utils/*` or `miles/backends/fsdp_utils/*`

### Visual guide: “first training step” logprob/KL alignment (Megatron mental model)

This is the specific flow you usually debug when you suspect precision / weight-sync / kernel nondeterminism problems.

```
Goal at rollout_id=0 (common debugging heuristic):
  rollout/log_probs  == rollout/ref_log_probs    (=> KL ~ 0 at step 0)

Data path:
  (Rollout)  SGLang generates (x, y) and rewards
      |
      v
  RolloutManager packages tokens + masks + rewards into a training batch
      |
      v
  (Train) Megatron computes:
      - old logprobs:  log p_old(y | x)   (from actor weights at start of step)
      - ref logprobs:  log p_ref(y | x)   (from reference weights, if enabled)
      - new logprobs:  log p_new(y | x)   (after/before update depending on loss path)
      |
      v
  KL signals (conceptually):
      - “rollout stats” view: compare log_probs vs ref_log_probs at rollout 0
      - “PPO step” view:      ppo_kl ≈ old_logprobs - new_logprobs  (should be 0 at first step)

Common failure buckets:
  - weights not actually identical (load/update path)
  - kernels / precision / determinism mismatch (even with same weights)
  - CP/TP/PP slicing or masking bugs (logprob computed on different token sets)
```

## Primer: training–inference mismatch (and rollout correction)

Miles splits the RL loop into two subsystems:

- **Rollout / inference** (typically SGLang): generates tokens and can compute rollout logprobs.
- **Training** (Megatron or FSDP): computes gradients and updates weights, and often recomputes logprobs inside the loss.

This separation is great for throughput, but it introduces an important failure mode: **the “policy that generated the data” and the “policy/logprobs used to train on that data” may not be exactly consistent**. When that happens, the system becomes *quietly off-policy* even if you think you’re doing “on-policy RL”.

### What exactly is “mismatch”?

Fix a prompt `x` and a generated token sequence `y = (y1..yT)` produced during rollout.

- **Rollout logprob**: `logp_rollout = sum_t log p_rollout(yt | x, y< t)`
- **Training logprob**: `logp_train   = sum_t log p_train(yt | x, y< t)` (computed by the training engine)

Mismatch means these are not equal, even for the same `(x, y)`, i.e.:

- token-level differences: `log p_rollout(yt | ...) != log p_train(yt | ...)`
- sequence-level differences: `logp_rollout != logp_train`

Miles tracks this explicitly; a key “am I really on-policy?” signal is the token-level diff metric (see the true on-policy example for the exact metric name and expected value).

### Why mismatch happens (common root causes)

There are two broad categories:

1. **Numerical / implementation mismatch**: for the same prompt `x` and generated tokens `y`, the log-probabilities computed by SGLang and by the training engine may differ (different kernels, precision, determinism settings, fused ops, etc.).
2. **Staleness mismatch**: even if implementations matched, rollouts might be generated with weights that are slightly behind the training engine due to weight sync cadence / asynchrony.

Concrete examples you’ll see in practice:

- **Different attention/GEMM kernels** between rollout and training (even when “both are flash attention”, different versions/backends can change numerics).
- **Mixed precision / quantization differences** (bf16 vs fp16 vs fp8; different accumulation rules).
- **Nondeterminism** (atomic reductions, kernel heuristics, different compilation paths).
- **Weight sync lag** (rollout samples were generated with `theta_{t-Δ}`, but training treats them as if they came from `theta_t`).

### Why this matters (RL math intuition, plain English)

PPO/GRPO-style updates are extremely sensitive to **which policy produced the actions** and **which logprobs you use inside the loss**.

In RL terms:

- **Behavior policy**: the system that actually generated `y` during rollout (often SGLang).
- **“Old/proximal” policy**: the logprobs you treat as the baseline for ratios/clipping/KL (often recomputed by training, unless you bypass).
- **Current/target policy**: the model you’re updating.

If you generate samples with behavior policy A (rollout) but compute “old” logprobs using policy B (training), you can accidentally:

- **Bias the importance ratios** used for clipping / advantage weighting
- **Mis-measure KL** (or other stability signals) that you rely on for tuning
- **Trigger instability/collapse** in long runs, because your update is more off-policy than you think

### Two ways Miles addresses the problem

- **Infrastructure-level fix (preferred when possible)**: make training and inference forward passes match, i.e. “true on-policy” (see `miles/examples/true_on_policy/README.md`). This aims to drive mismatch to (near) zero by aligning kernels/precision/determinism end-to-end.
- **Algorithmic fix (when mismatch can’t be eliminated)**: explicitly treat the system as off-policy and correct it via importance sampling / truncation / rejection sampling, or restructure PPO’s roles so behavior vs proximal vs target policies are handled correctly.

### How to diagnose quickly (before changing algorithms)

- **Start with metrics-only**: enable `--get-mismatch-metrics` so Miles reports mismatch-related metrics without changing the loss.
- **If available, watch the token-level diff** between rollout and training logprobs (this is the “ground truth” indicator of numerical mismatch).
- **Also watch KL / perplexity-style mismatch metrics** to see whether drift is growing over training.

### What the linked doc contains (how to use it)

Read `miles/examples/train_infer_mismatch_helper/README.md` as a cookbook for the **algorithmic** approach. It includes:

- **Three correction modes**:
  - **Baseline PPO (no correction)**: what goes wrong when rollout vs training don’t match
  - **Bypass mode**: use rollout-provided logprobs directly in the loss (skips recomputing “old logprobs” on the training engine)
  - **Decoupled 3-policy PPO**: separates “behavior” (rollout), “proximal/old” (training-recomputed), and “target” (current) policies and applies importance sampling between them
- **The key flags** to turn things on:
  - `--use-rollout-logprobs` (bypass recomputation)
  - `--use-tis` and `--custom-config-path` (configure truncated importance sampling / rejection sampling)
  - `--get-mismatch-metrics` (monitor mismatch without changing training)
- **Mismatch metrics** you can watch to quantify drift (e.g. rollout-vs-train KL / logprob diffs) and IS/RS diagnostics when enabled.

If you’re new to the mismatch topic, a good progression is:

1. Turn on **metrics-only** (`--get-mismatch-metrics`) and confirm whether mismatch is actually present and whether it grows over time.
2. If you can afford it and your use case needs strict correctness/debuggability, try the **infrastructure route** (true on-policy) to eliminate the mismatch source.
3. If you can’t eliminate mismatch (or you’re running intentionally async / partially stale), use the **algorithmic route**:
   - simplest: `--use-rollout-logprobs` (treat rollout logprobs as the “old” policy term)
   - more principled: enable correction (`--use-tis` + config) and/or use a decoupled PPO mode as described in the linked doc

## Questions These Examples Should Answer

1. **How do model architecture scripts work?** What is the purpose of `scripts/models/` and how are model configuration scripts (e.g., `deepseek-v3.sh`) structured and used in training workflows?

2. **How do you set up and configure a MILES training run?** What are the key configuration parameters, environment setup steps, and how do you initialize a training job for MoE models?

3. **How does true on-policy training work in MILES?** What infrastructure-level mechanisms ensure zero mismatch between training and inference, and how is this implemented with SGLang + FSDP?

4. **How do you implement speculative training?** How does MILES perform online SFT on draft models during RL, and what are the performance benefits and implementation details?

5. **How do you handle memory optimization for large MoE models?** What strategies and techniques does MILES use to manage GPU memory efficiently, handle OOMs gracefully, and optimize memory usage for production workloads?

## Answers / Where to Look

### 1) Model architecture scripts (`miles/scripts/models/*.sh`)

**Purpose:** Megatron (and some conversion flows) require the model *architecture hyperparameters* to be provided explicitly as CLI flags; they are not reliably inferred from checkpoints in the way HF inference stacks typically are. Miles keeps those “shape-defining” flags in **bash model scripts** under `miles/scripts/models/`.

**How they’re structured:** Each script defines a bash array called `MODEL_ARGS=( ... )` containing Megatron-compatible flags (e.g. `--num-layers`, `--hidden-size`, plus MoE routing/topk/dispatcher knobs for MoE models).

- Example: `miles/scripts/models/deepseek-v3.sh` dynamically constructs `--moe-layer-freq` based on `MODEL_ARGS_NUM_LAYERS` and then populates `MODEL_ARGS` with attention + MoE settings.

**How they’re used in workflows:**

- **Weight conversion (HF → Megatron torch_dist)**: `docs/en/get_started/quick_start.md` shows sourcing a model script, then running `tools/convert_hf_to_torch_dist.py` with `${MODEL_ARGS[@]}`.
- **Training launch**: the `miles/scripts/run-*.sh` launchers typically do `source "${SCRIPT_DIR}/models/<model>.sh"` and pass `${MODEL_ARGS[@]}` into `python3 train.py`.

**How you customize:** source the script, then append/override flags in-place, e.g.:

- `MODEL_ARGS+=(--rotary-base 10000)`

### 2) Setting up + configuring a Miles training run

**Recommended setup path:** follow the official quick start in `miles/docs/en/get_started/quick_start.md` (Docker-first, because Miles may include patches for SGLang/Megatron).

**High-level flow (Megatron backend):**

- **Environment**: run inside Miles Docker (or follow `miles/build_conda.sh` if you must).
- **Install**: `pip install -e .` inside `miles/`.
- **Download**: model weights + datasets (examples use `dapo-math-17k` for training and `aime-2024` for eval).
- **Convert**: HF → Megatron `torch_dist` (`tools/convert_hf_to_torch_dist.py`) using `${MODEL_ARGS[@]}` from `miles/scripts/models/...`.
- **Launch**: run one of the example launchers, e.g. `miles/scripts/run-qwen3-4B.sh`, which submits `python3 train.py` as a Ray job and also launches SGLang engines for rollout.

**How a run is configured (the canonical “run script” pattern):**

- **MODEL_ARGS**: architecture flags (from `miles/scripts/models/*.sh`)
- **CKPT_ARGS**: `--hf-checkpoint`, `--ref-load`, `--load`, `--save`, `--save-interval`
- **ROLLOUT_ARGS**: `--prompt-data`, `--rollout-batch-size`, `--n-samples-per-prompt`, sampling params, reward model config
- **TRAIN/ALGO args**: e.g. `--advantage-estimator grpo`, KL settings, optimizer/lr
- **PERF args**: TP/PP/CP/EP sizes + recompute/checkpointing + dynamic batch sizing
- **SGLANG args**: forwarded with `--sglang-...` (plus `--rollout-num-gpus-per-engine`)
- **Ray cluster args**: `--actor-num-nodes`, `--actor-num-gpus-per-node`, and often `--colocate`

**The “make batch sizes consistent” rule:** Miles enforces:
`rollout_batch_size * n_samples_per_prompt = global_batch_size * num_steps_per_rollout`
See the explanation in `miles/docs/en/get_started/quick_start.md` and the validation logic in `miles/miles/utils/arguments.py`.

**Multi-node:** start a Ray cluster (`ray start --head ...` on node0, `ray start --address ...` on others) then `ray job submit ... python3 train.py ...` (also documented in `miles/docs/en/get_started/quick_start.md`).

### 3) True on-policy training (zero train/infer mismatch)

**What it means in Miles:** the token-level logprobs produced by the rollout inference engine (SGLang) are **bitwise / numerically aligned** with those produced by the training engine, so the tracked metric `train/train_rollout_logprob_abs_diff` should be **exactly 0**.

**Where the example is:** `miles/examples/true_on_policy/README.md`.

**How to enable it (FSDP path):** the unified runner `miles/scripts/run_mcore_fsdp.py` exposes `--true-on-policy` and translates it into:

- `--true-on-policy-mode`
- deterministic SGLang + deterministic training settings
- aligned attention backends (FlashAttention-3 both sides) and specific determinism env vars

**How it works (infra-level alignment):** see `miles/examples/true_on_policy/README.md` for the “what is aligned” list (FlashAttention-3, DeepGEMM, batch-invariant kernels, `torch.compile`, and matched numeric details between training + inference).

### 4) Speculative training (online SFT for the draft during RL)

Miles’ speculative acceleration is built on **SGLang speculative decoding** (EAGLE) plus **online training of the draft (MTP) layers** during RL to avoid drift.

**Where the doc is:** `miles/docs/en/advanced/speculative-decoding.md`.

**How to turn on speculative decoding (rollout speedup):**

- `--sglang-speculative-algorithm EAGLE`
- `--sglang-speculative-num-steps ...`
- `--sglang-speculative-num-draft-tokens ...`
- (often) `--sglang-enable-draft-weights-cpu-backup`

See a working end-to-end launcher in `miles/scripts/run-mimo-7B-rl-eagle.sh`.

**How “speculative training” is implemented (online SFT of draft):**

- Enable training of MTP layers during RL:
  - `--mtp-num-layers 1` (or more)
  - `--enable-mtp-training`
  - `--mtp-loss-scaling-factor 0.2`

This keeps the draft distribution close to the target as the target updates, improving acceptance rate and preventing speculative decoding from becoming net-negative late in training.

### 5) Memory optimization for large MoE runs

Miles’ memory story is a combination of *scheduling*, *offload*, *recompute/checkpointing*, and *precision* controls. The most commonly-used knobs are:

- **Colocate + offload (train ↔ rollout sharing GPUs)**: `--colocate` automatically enables `--offload-train` and `--offload-rollout` (see `miles/miles/utils/arguments.py`). The training loop in `miles/train.py` explicitly onloads/offloads rollout weights/KV cache around each step.
- **Cap rollout GPU memory**: use `--sglang-mem-fraction-static <ratio>` (especially important in colocate mode; see `miles/docs/en/get_started/quick_start.md`).
- **Dynamic microbatching by token budget**: `--use-dynamic-batch-size --max-tokens-per-gpu N` packs variable-length samples to reduce padding waste (explained in `miles/docs/en/get_started/quick_start.md`).
- **Activation recompute / checkpointing**:
  - Megatron: `--recompute-granularity full --recompute-method uniform --recompute-num-layers ...` (used in many `miles/scripts/run-*.sh`)
  - FSDP: `--gradient-checkpointing` (see `miles/scripts/run_mcore_fsdp.py`)
- **Chunked weight sync (important for MoE)**: `--update-weight-buffer-size` controls the staging buffer for sending weights to rollout engines (defined in `miles/miles/utils/arguments.py`; also set in `miles/scripts/run_mcore_fsdp.py`).
- **FP8 for lower memory + higher throughput**: see `miles/examples/low_precision/README.md` and the conversion helpers under `miles/tools/` (e.g. `convert_hf_to_fp8.py`).
- **Memory safety margins / host memory controls**:
  - `--train-memory-margin-bytes` reserves headroom to avoid NCCL-related OOM edge cases
  - `--disable-weights-backuper` can reduce host peak memory (both in `miles/miles/utils/arguments.py`)
- **Fault tolerance to survive “benign” failures**: `--use-fault-tolerance` enables rollout health checks and engine restart (see `miles/docs/en/advanced/fault-tolerance.md`).
