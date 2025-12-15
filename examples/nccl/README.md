# NCCL Examples

This folder vendors **NCCL** as a git submodule and provides a quick map for learning + hacking on it.

- **Submodule**: `examples/nccl/nccl/`
- **Upstream**: [NVIDIA/nccl](https://github.com/NVIDIA/nccl)

## What NCCL is

**NCCL** (pronounced “Nickel”) is NVIDIA’s library of optimized collective communication primitives for multi-GPU and multi-node training. It implements **all-reduce**, **all-gather**, **reduce**, **broadcast**, **reduce-scatter**, and **send/recv**, tuned for PCIe, NVLink/NVSwitch, and networking (InfiniBand verbs or TCP/IP sockets).

Official docs: [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)

## Where to start in this repo

- **High-level README / build**: `examples/nccl/nccl/README.md`
- **Examples (best first read)**: `examples/nccl/nccl/examples/`
  - `01_communicators/`: communicator setup patterns (single process / pthread / MPI)
  - `03_collectives/01_allreduce/`: minimal all-reduce example
  - `02_point_to_point/`: send/recv ring patterns
  - `06_device_api/`: device-side collectives examples
- **Core implementation**: `examples/nccl/nccl/src/`
  - `collectives.cc`: top-level collective entrypoints
  - `transport/`: IB/TCP and other transport backends
  - `graph/`: topology discovery + ring/tree construction

## Build (source)

From the submodule root:

```bash
cd examples/nccl/nccl
make -j src.build
```

If CUDA isn’t under `/usr/local/cuda`, pass `CUDA_HOME`:

```bash
make -j src.build CUDA_HOME=/path/to/cuda
```

## Tests / benchmarks

NCCL’s common perf tests are in a separate repo:

- [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests)

Typical flow:

```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```


