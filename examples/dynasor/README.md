# Dynasor (submodule)

This folder vendors **Dynasor** as a git submodule:

- **Upstream**: `https://github.com/hao-ai-lab/Dynasor`
- **Local path**: `examples/dynasor/Dynasor`

## What it is

Dynasor🦖 is a tool that helps you **speed up LLM reasoning models without training**.
It uses dynamic execution and early stopping techniques to improve inference efficiency
for chain-of-thought reasoning. Built as an extension on vLLM.

Integrated into [Snowflake Arctic Inference](https://github.com/snowflakedb/ArcticInference) and
[NVIDIA TRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).

## Start here (inside the submodule)

- `Dynasor/README.md` — full docs + quick start
- `Dynasor/docs/local.md` — run Dynasor locally

## Quickstart (from upstream)

```bash
# Install
git clone https://github.com/hao-ai-lab/Dynasor.git
cd Dynasor && pip install . && cd -

# Start chat with an endpoint
dynasor-chat --base-url http://localhost:8000/v1
```

## Tools provided

- `dynasor-chat` — CLI chat interface
- `dynasor-openai` — OpenAI-compatible server
- `dynasor-vllm` — vLLM-native server

