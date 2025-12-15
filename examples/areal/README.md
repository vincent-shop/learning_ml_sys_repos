# AReaL (submodule)

This folder vendors **AReaL** as a git submodule:

- **Upstream**: `https://github.com/inclusionAI/AReaL`
- **Local path**: `examples/areal/AReaL`

## What it is

An open-source **fully asynchronous** RL training system for large **reasoning and
agentic** models, built on ReaLHF, with strong support for **SGLang** inference and
both **Megatron** + **PyTorch FSDP** training backends.

## Start here (inside the submodule)

- `AReaL/README.md` (project overview + links)
- `AReaL/examples/` (end-to-end example tasks: math, multi-turn, search-agent, RLHF, etc.)
- `AReaL/docs/` (docs source; the rendered site is linked from the top-level README)

## Quickstart (from upstream README)

```bash
python3 -m areal.launcher.local \
  examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml
```


