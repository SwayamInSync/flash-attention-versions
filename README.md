# Flash Attention - Separated Implementations

A reorganized workspace containing **Flash Attention 2, 3, and 4** as standalone, self-contained modules — extracted from the [official Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) repository for easier study, development, and benchmarking of individual versions.

## Repository Structure

```
flash-attention/
├── flash_attention_2/      # Flash Attention 2 — CUDA (Ampere / SM80)
├── flash_attention_3/      # Flash Attention 3 — CUDA (Hopper SM90 + Blackwell SM100)
├── flash_attention_4/      # Flash Attention 4 — CuTeDSL / JIT (SM90 / SM100 / SM110)
├── flash-attention-upstream/  # Original upstream repo (git submodule)
└── README.md
```

## Versions at a Glance

| Version | Language | Target GPUs | Key Feature |
|---------|----------|-------------|-------------|
| **Flash Attention 2** | CUDA C++ | Ampere (SM80) | IO-aware exact attention with tiling |
| **Flash Attention 3** | CUDA C++ | Ampere, Hopper, Blackwell (SM80/90/100) | TMA + WGMMA, paged KV cache |
| **Flash Attention 4** | Python (CuTeDSL) | Hopper, Blackwell (SM90/100/110) | JIT-compiled kernels, block sparsity |

## Flash Attention 2

The foundational CUDA implementation targeting Ampere GPUs.

- FP16 / BF16 support
- Head dimensions: 32, 64, 96, 128, 192, 256
- Causal masking, dropout, ALiBi, SplitKV
- Requires: CUDA 11.8+, PyTorch 2.0+

See [flash_attention_2/README.md](flash_attention_2/README.md) for build instructions.

## Flash Attention 3

Next-generation implementation leveraging Hopper and Blackwell hardware features.

- TMA (Tensor Memory Accelerator) and WGMMA instructions
- Paged KV cache, Pack GQA, sliding window attention
- Softcap, varlen attention, deterministic backward
- Requires: CUDA 12.0+, PyTorch 2.4+

See [flash_attention_3/README.md](flash_attention_3/README.md) for build instructions.

## Flash Attention 4

Latest generation using NVIDIA CuTeDSL for runtime JIT compilation.

- No ahead-of-time CUDA compilation needed
- Block sparsity, score/mask modifiers, 2CTA (SM100)
- Requires: CUDA 12.0+, PyTorch 2.4+, Python 3.10+, `nvidia-cutlass-dsl >= 4.4.1`

See [flash_attention_4/README.md](flash_attention_4/README.md) for build instructions.

## Upstream

The full original [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) repository is included as a **git submodule** under `flash-attention-upstream/` for reference.

```bash
# After cloning, initialize the submodule:
git submodule update --init --recursive
```

## Getting Started

```bash
# Clone with submodule
git clone --recursive https://github.com/SwayamInSync/flash-attention.git
cd flash-attention

# Or, if already cloned:
git submodule update --init --recursive
```

Then navigate into the version you want to build — each has its own `setup.py` / `pyproject.toml` and build instructions.

## Credits

All core attention algorithms are by **Tri Dao** and collaborators at [Dao-AILab](https://github.com/Dao-AILab).

**Papers:**
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://tridao.me/publications/flash2/flash2.pdf) — Tri Dao
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://tridao.me/publications/flash3/flash3.pdf) — Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao
- [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451) — Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, Tri Dao

## License

The individual modules retain their original licenses from the upstream repository. See the `LICENSE` file in each subdirectory and in `flash-attention-upstream/` for details.
