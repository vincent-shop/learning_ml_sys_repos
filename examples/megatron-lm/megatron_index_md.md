# Megatron-LM Markdown Index

This file is an index of Markdown docs under `examples/megatron-lm/megatron-lm/` (vendored Megatron-LM).

- Upstream links: `https://github.com/NVIDIA/Megatron-LM/blob/main/<path>`
- Files indexed: **79**
- Descriptions: first *meaningful* line (skips common wrappers like YAML `---`, HTML `<div>`, and code-fence/include directives).

## Index

### `./`
- **Changelog** — [`CHANGELOG.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/CHANGELOG.md)
- **Contributing to Megatron-LM** — [`CONTRIBUTING.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/CONTRIBUTING.md)
- **Megatron-LM & Megatron Core** — [`README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/README.md)

### `.github/`
- **name: Bug report** — [`.github/ISSUE_TEMPLATE/bug_report.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/.github/ISSUE_TEMPLATE/bug_report.md)
- **name: Feature request** — [`.github/ISSUE_TEMPLATE/feature_request.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/.github/ISSUE_TEMPLATE/feature_request.md)
- **name: QUESTION** — [`.github/ISSUE_TEMPLATE/question.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/.github/ISSUE_TEMPLATE/question.md)
- **name: REGRESSION** — [`.github/ISSUE_TEMPLATE/regression.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/.github/ISSUE_TEMPLATE/regression.md)
- **What does this PR do ?** — [`.github/pull_request_template.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/.github/pull_request_template.md)

### `docs/`
- **API Backward Compatibility Checking** — [`docs/api-backwards-compatibility-check.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-backwards-compatibility-check.md)
- **context_parallel package** — [`docs/api-guide/context_parallel.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/context_parallel.md)
- **NOTE: In M-Core 0.14, the custom FSDP refactored its checkpoint implementation to use DTensor-based torch distributed checkpointing. The custom FSDP was also renamed Megatron FSDP. The relevant sections of this document are no longer applicable.** — [`docs/api-guide/custom_fsdp.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/custom_fsdp.md)
- **datasets package** — [`docs/api-guide/datasets.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/datasets.md)
- **(no description)** — [`docs/api-guide/datasets_readme.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/datasets_readme.md)
- **dist_checkpointing package** — [`docs/api-guide/dist_checkpointing.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/dist_checkpointing.md)
- **dist_checkpointing.strategies package** — [`docs/api-guide/dist_checkpointing.strategies.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/dist_checkpointing.strategies.md)
- **Distributed Optimizer** — [`docs/api-guide/dist_optimizer.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/dist_optimizer.md)
- **distributed package** — [`docs/api-guide/distributed.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/distributed.md)
- **fusions package** — [`docs/api-guide/fusions.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/fusions.md)
- **API Guide** — [`docs/api-guide/index.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/index.md)
- **models.bert package** — [`docs/api-guide/models.bert.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/models.bert.md)
- **models.gpt package** — [`docs/api-guide/models.gpt.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/models.gpt.md)
- **models package** — [`docs/api-guide/models.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/models.md)
- **models.t5 package** — [`docs/api-guide/models.t5.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/models.t5.md)
- **Mixture of Experts package** — [`docs/api-guide/moe.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/moe.md)
- **Multi-Latent Attention** — [`docs/api-guide/multi_latent_attention.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/multi_latent_attention.md)
- **Multi-Token Prediction (MTP)** — [`docs/api-guide/multi_token_prediction.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/multi_token_prediction.md)
- **Microbatches Calculator** — [`docs/api-guide/num_microbatches_calculator.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/num_microbatches_calculator.md)
- **Optimizer CPU offload package** — [`docs/api-guide/optimizer_cpu_offload.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/optimizer_cpu_offload.md)
- **Optimizer Parameters Scheduler** — [`docs/api-guide/optimizer_param_scheduler.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/optimizer_param_scheduler.md)
- **pipeline_parallel package** — [`docs/api-guide/pipeline_parallel.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/pipeline_parallel.md)
- **Custom Pipeline Model Parallel Layout** — [`docs/api-guide/pipeline_parallel_layout.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/pipeline_parallel_layout.md)
- **tensor_parallel package** — [`docs/api-guide/tensor_parallel.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/tensor_parallel.md)
- **New Tokenizer System** — [`docs/api-guide/tokenizers.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/tokenizers.md)
- **transformer package** — [`docs/api-guide/transformer.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/api-guide/transformer.md)
- **Megatron User Guide** — [`docs/index.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/index.md)
- **Llama, Mistral and other Llama-like model support in Megatron-LM** — [`docs/llama_mistral.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/llama_mistral.md)
- **User Guide** — [`docs/user-guide/index.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/index.md)
- **(no description)** — [`docs/user-guide/msc_integration.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/msc_integration.md)
- **(no description)** — [`docs/user-guide/quickstart.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/quickstart.md)

### `examples/`
- **SGEAT: Detoxify Larger-scale Language Models** — [`examples/academic_paper_scripts/detxoify_lm/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/academic_paper_scripts/detxoify_lm/README.md)
- **Multi-Stage Prompting for Knowledgeable Dialogue Generation** — [`examples/academic_paper_scripts/msdp/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/academic_paper_scripts/msdp/README.md)
- **Reproducing Figures in SC21 Paper** — [`examples/academic_paper_scripts/sc21/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/academic_paper_scripts/sc21/README.md)
- **BERT MODEL** — [`examples/bert/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/bert/README.md)
- **Megatron Core Export** — [`examples/export/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/export/README.md)
- **Megatron Core To TRTLLM Export Documentation** — [`examples/export/trtllm_export/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/export/trtllm_export/README.md)
- **GPT3 MODEL** — [`examples/gpt3/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/gpt3/README.md)
- **Megatron Core Inference Documentation** — [`examples/inference/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/inference/README.md)
- **Llama Models** — [`examples/llama/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/llama/README.md)
- **Mamba-based Language Models** — [`examples/mamba/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/mamba/README.md)
- **Mixtral 8x7B Model Inference and Finetuning** — [`examples/mixtral/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/mixtral/README.md)
- **Multimodal Example** — [`examples/multimodal/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/multimodal/README.md)
- **Llama-3.1-Nemotron-Nano-VL-8B-V1** — [`examples/multimodal/llama_3p1_nemotron_nano_vl_8b_v1/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/multimodal/llama_3p1_nemotron_nano_vl_8b_v1/README.md)
- **NVLM** — [`examples/multimodal/nvlm/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/multimodal/nvlm/README.md)
- **Advanced Usage** — [`examples/post_training/modelopt/ADVANCED.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/ADVANCED.md)
- **Model Optimizer Integrated Examples** — [`examples/post_training/modelopt/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/README.md)
- **Speculative Decoding** — [`examples/post_training/modelopt/speculative.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/post_training/modelopt/speculative.md)
- **RETRO MODEL** — [`examples/retro/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/retro/README.md)
- **Reinforcement Learning in megatron** — [`examples/rl/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/rl/README.md)
- **Countdown Agentic Environment** — [`examples/rl/environments/countdown/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/rl/environments/countdown/README.md)
- **T5 MODEL** — [`examples/t5/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/t5/README.md)

### `megatron/`
- **Multi-Storage Client (MSC) Integration** — [`megatron/core/MSC_Integration.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/MSC_Integration.md)
- **Quick Start** — [`megatron/core/QuickStart.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/QuickStart.md)
- **Megatron Core** — [`megatron/core/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/README.md)
- **StragglerDetector for a TP Group** — [`megatron/core/README_STRAGGLER.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/README_STRAGGLER.md)
- **Data Pipeline** — [`megatron/core/datasets/readme.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/readme.md)
- **How to use pytorch FSDP2?** — [`megatron/core/distributed/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/README.md)
- **🚀 Megatron-FSDP** — [`megatron/core/distributed/fsdp/src/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/fsdp/src/README.md)
- **About** — [`megatron/core/extensions/TransformerEngineMixedPrecision.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/extensions/TransformerEngineMixedPrecision.md)
- **MIMO: Multimodal In/Out Model** — [`megatron/core/models/mimo/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/mimo/README.md)
- **How to use ?** — [`megatron/core/optimizer/cpu_offloading/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/cpu_offloading/README.md)
- **Megatron Core MoE** — [`megatron/core/transformer/moe/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)
- **Megatron-LM ModelOpt Distillation Integration** — [`megatron/post_training/docs/distillation.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/post_training/docs/distillation.md)
- **Megatron-RL** — [`megatron/rl/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/rl/README.md)
- **Data Pipeline** — [`megatron/training/datasets/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/datasets/README.md)

### `tests/`
- **Gradient tests** — [`tests/functional_tests/test_cases/gpt/gpt3_mcore_reruns_resume_check_grads/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/tests/functional_tests/test_cases/gpt/gpt3_mcore_reruns_resume_check_grads/README.md)
- **Gold standard prompts** — [`tests/functional_tests/test_cases/gpt/gpt_static_inference_tp1_pp1_16b_multiprompt_tokensmatch/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/tests/functional_tests/test_cases/gpt/gpt_static_inference_tp1_pp1_16b_multiprompt_tokensmatch/README.md)

### `tools/`
- **Retro and InstructRetro** — [`tools/retro/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/retro/README.md)
- **This directory contains a collection of tools for building the retrieval database and pretraining neighbors for Retro. This preprocessing pipeline is broken into 3 main stages:** — [`tools/retro/build_db.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/retro/build_db.md)
- **Note** — [`tools/retro/sft/README.md`](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/retro/sft/README.md)
