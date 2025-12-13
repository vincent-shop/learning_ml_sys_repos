# MILES Examples

**MILES** is an enterprise-facing reinforcement learning framework for large-scale MoE (Mixture of Experts) post-training and production workloads. Forked from [slime](https://github.com/THUDM/slime), MILES focuses on new hardware support (e.g., GB300), stable RL for large MoE models, and production-grade features.

## What These Examples Cover

These examples demonstrate practical MILES workflows for RL training:

- **Training Setup**: Configuration and initialization for MoE RL training
- **True On-Policy Training**: Infrastructure-level on-policy support for SGLang + FSDP
- **Speculative Training**: Online SFT on draft models during RL for faster rollouts
- **Memory Optimization**: Handling large MoE models with efficient memory management
- **Production Workflows**: Enterprise-ready training pipelines and best practices

Each example includes setup instructions, configuration files, and explanations of key concepts.

