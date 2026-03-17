# Flash Attention 3

Flash Attention 3 is the next-generation fast, memory-efficient exact attention algorithm with support for Hopper (SM90) and Blackwell (SM100) GPUs. It features improved performance through TMA (Tensor Memory Accelerator) and WGMMA (Warp Group Matrix Multiply Accumulate) instructions.

## Features

- Fast exact attention with O(N d) complexity
- SM80, SM90, SM100 GPU support (Ampere, Hopper, Blackwell)
- FP16 and BF16 precision support
- Head dimensions: 32, 64, 96, 128, 192, 256
- SplitKV support for long sequences
- Paged KV Cache support
- Pack GQA support
- Softcap for attention scores
- Sliding window / local attention
- Varlen (variable length) attention
- Deterministic backward pass option

## Requirements

- CUDA Toolkit 12.0+
- PyTorch 2.4+ (for SM90/SM100 support)
- Python 3.8+
- cutlass (included as submodule at `third_party/cutlass`)

## Building from Source

### 1. Clone and prepare dependencies

Ensure you have the repository with submodules:

```bash
# If cloning fresh:
git clone --recursive <repo-url>

# If already cloned, init submodules:
git submodule update --init --recursive
```

### 2. Build

```bash
cd flash_attention_3

# Standard install
pip install .

# Or editable install for development
pip install -e .
```

### 3. Build options

The build compiles for SM80, SM90, and SM100 by default. To specify GPU architectures:

```bash
# Build with custom CUDA archs
export FLASH_ATTN_CUDA_ARCHS="90;100"
pip install .
```

You can also disable certain features at build time:

```bash
# Disable specific features
export FLASH_ATTENTION_DISABLE_SPLIT=1       # Disable SplitKV
export FLASH_ATTENTION_DISABLE_PAGEDKV=1     # Disable Paged KV Cache
export FLASH_ATTENTION_DISABLE_SOFTCAP=1     # Disable softcap
export FLASH_ATTENTION_DISABLE_PACKGQA=1     # Disable Pack GQA
pip install .
```

## File Structure

```
flash_attention_3/
├── setup.py                    # Build configuration
├── flash_api.cpp              # Python binding entry point
├── flash_api_stable.cpp       # Stable API entry point (PyTorch 2.9+)
├── flash_attn_interface.py    # Python interface
├── flash_fwd_combine.cu       # SplitKV combine kernel
├── flash_prepare_scheduler.cu # Scheduler preparation
├── generate_kernels.py        # Kernel generation script
├── cuda_check.h               # CUDA capability checks
├── block.h                    # Block utilities
├── heuristics.h               # Kernel heuristics
├── mask.h                     # Masking logic
├── paged_kv.h                 # Paged KV Cache
├── pack_gqa.h                 # Pack GQA
├── rotary.h                   # RoPE rotary embeddings
├── softmax.h                  # Softmax implementation
├── tile_scheduler.hpp         # Tile scheduling
└── src/
    ├── flash_fwd_*.cu            # Forward kernels (SM80/SM90/SM100)
    ├── flash_bwd_*.cu            # Backward kernels (SM80/SM90)
    └── *.hpp                     # Kernel headers
```

## Usage

```python
import torch
from flash_attn import flash_attn_func

# Standard attention
q = torch.randn(2, 4, 16, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 4, 16, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 4, 16, 64, device='cuda', dtype=torch.float16)

out = flash_attn_func(q, k, v)

# With causal masking
out = flash_attn_func(q, k, v, causal=True)

# With sliding window
out = flash_attn_func(q, k, v, window_size_left=512, window_size_right=0)
```

## Performance Notes

- FA3 is optimized for Hopper (SM90) and Blackwell (SM100) GPUs
- On Ampere (SM80) GPUs, consider using Flash Attention 2 for better performance
- Use FP16 for best performance, BF16 for better numerical stability
- Enable SplitKV for very long sequences (>4K)
- Use Paged KV Cache for inference with long context

## License

See root LICENSE file.
