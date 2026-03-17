# Flash Attention 2

Flash Attention 2 is a fast, memory-efficient exact attention algorithm written in CUDA. This is the original Flash Attention implementation targeting Ampere (SM80) GPUs.

## Features

- Fast exact attention with O(N d) complexity
- SM80 GPU support (Ampere architecture)
- FP16 and BF16 precision support
- Head dimensions: 32, 64, 96, 128, 192, 256
- SplitKV support for long sequences
- Causal masking support
- Dropout support

## Requirements

- CUDA Toolkit 11.8+
- PyTorch 2.0+
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
cd flash_attention_2

# Standard install
pip install .

# Or editable install for development
pip install -e .
```

### 3. Build options

The build uses PyTorch's CUDA extension builder. By default it compiles for SM80 (Ampere). To specify GPU architectures:

```bash
# Build with custom CUDA archs
export FLASH_ATTN_CUDA_ARCHS="80"
pip install .
```

## File Structure

```
flash_attention_2/
├── setup.py              # Build configuration
├── flash_api.cpp        # Python binding entry point
├── generate_kernels.py  # Kernel generation script
├── philox.cuh           # Philox random number generator
├── philox_unpack.cuh   # Philox helper
└── src/
    ├── flash_fwd_*.cu       # Forward kernels
    ├── flash_bwd_*.cu       # Backward kernels
    ├── flash_fwd_split_*.cu # SplitKV forward kernels
    └── *.h                  # Headers
```

## Usage

```python
import torch
from flash_attn import flash_attn_func

q = torch.randn(2, 4, 16, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 4, 16, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 4, 16, 64, device='cuda', dtype=torch.float16)

out = flash_attn_func(q, k, v)
```

## License

See root LICENSE file.
