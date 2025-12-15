# SGLang (submodule)

This folder vendors **SGLang** as a git submodule:

- **Upstream**: `https://github.com/sgl-project/sglang`
- **Local path**: `examples/sglang/sglang`

## What it is

SGLang is a **high-performance serving framework** for large language models and
vision-language models, designed for low-latency, high-throughput inference from
a single GPU to large distributed clusters.

**Core features:**
- RadixAttention for prefix caching, zero-overhead CPU scheduler
- Prefill-decode disaggregation, speculative decoding, continuous batching
- Tensor/pipeline/expert/data parallelism
- FP4/FP8/INT4/AWQ/GPTQ quantization, multi-LoRA batching
- Supports NVIDIA, AMD, Intel, TPU, Ascend hardware

**Industry adoption:** Deployed on 400,000+ GPUs worldwide, powering xAI, AMD, NVIDIA,
LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, and more.

## Start here (inside the submodule)

- `sglang/README.md` — project overview
- [Documentation](https://docs.sglang.io/) — full docs
- [Install Guide](https://docs.sglang.io/get_started/install.html)
- [Quick Start](https://docs.sglang.io/basic_usage/send_request.html)

## Links

- [Blog](https://lmsys.org/blog/)
- [Roadmap](https://roadmap.sglang.io/)
- [Slack](https://slack.sglang.io/)

