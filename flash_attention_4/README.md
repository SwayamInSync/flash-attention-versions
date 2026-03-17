# Flash Attention 4

Flash Attention 4 is the latest generation of fast, memory-efficient exact attention, implemented using CuTeDSL (NVIDIA CUTLASS CUDA Template Engine). It features runtime JIT compilation of kernels optimized for Hopper (SM90) and Blackwell (SM100/SM110) GPUs.

## Features

- Fast exact attention with O(N d) complexity
- Hopper (SM90) and Blackwell (SM100/SM110) GPU support
- JIT-compiled kernels at runtime using CuTeDSL
- FP16 and BF16 precision support
- Head dimensions: 64, 96, 128
- SplitKV support for long sequences
- Paged KV Cache support
- Sliding window / local attention
- Block sparsity support
- Deterministic backward pass option
- 2CTA instruction support (SM100)
- Score modifier and mask modifier support

## Requirements

- CUDA Toolkit 12.0+
- PyTorch 2.4+
- Python 3.10+
- nvidia-cutlass-dsl >= 4.4.1
- apache-tvm-ffi
- quack-kernels >= 0.3.3

## Building from Source

### 1. Clone and prepare dependencies

```bash
git clone --recursive <repo-url>
cd flash-attention
```

### 2. Install dependencies

```bash
pip install nvidia-cutlass-dsl>=4.4.1
pip install apache-tvm-ffi quack-kernels torch einops
```

### 3. Build

```bash
cd flash_attention_4

# Standard install
pip install .

# Or editable install for development
pip install -e .
```

## File Structure

```
flash_attention_4/
├── pyproject.toml           # Build configuration
├── __init__.py             # Package init
├── README.md               # This file
├── interface.py            # Main Python API
├── flash_fwd.py            # Forward pass base
├── flash_fwd_sm90.py       # Hopper forward kernels
├── flash_fwd_sm100.py      # Blackwell forward kernels
├── flash_fwd_sm120.py      # Reserved for future arch
├── flash_fwd_combine.py    # SplitKV combine kernel
├── flash_bwd.py            # Backward pass base
├── flash_bwd_sm90.py       # Hopper backward kernels
├── flash_bwd_sm100.py      # Blackwell backward kernels
├── flash_bwd_preprocess.py # Backward preprocessing
├── flash_bwd_postprocess.py# Backward postprocessing
├── mask.py                 # Masking logic
├── softmax.py               # Online softmax
├── tile_scheduler.py       # Tile scheduling
├── pipeline.py             # Pipeline state management
├── paged_kv.py             # Paged KV Cache
├── pack_gqa.py             # Pack GQA support
├── block_sparsity.py       # Block sparsity
├── cache_utils.py          # JIT compilation caching
├── cute_dsl_utils.py       # CuTeDSL utilities
├── blackwell_helpers.py    # Blackwell-specific helpers
├── ampere_helpers.py       # Ampere helpers (fallback)
└── [other] *.py            # Utility modules
```

## Usage

```python
import torch
from flash_attn_4 import flash_attn_func

# Standard attention
q = torch.randn(2, 4, 16, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 4, 16, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 4, 16, 64, device='cuda', dtype=torch.float16)

out = flash_attn_func(q, k, v)

# With causal masking
out = flash_attn_func(q, k, v, causal=True)

# With sliding window
out = flash_attn_func(q, k, v, window_size_left=512, window_size_right=0)

# Variable length sequences
from flash_attn_4 import flash_attn_varlen_func

cu_seqlens_q = torch.tensor([0, 1024, 2048], device='cuda', dtype=torch.int32)
cu_seqlens_k = torch.tensor([0, 1024, 2048], device='cuda', dtype=torch.int32)

out = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k)
```

## Environment Variables

FA4 uses several environment variables for debugging and performance tuning:

```bash
# Enable disk cache for compiled kernels
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1

# Use FakeTensorMode for compilation (no GPU memory needed)
FLASH_ATTENTION_FAKE_TENSOR=1

# Dump compiled SASS/CUBIN to disk
CUTE_CUBIN_PATH=/path/to/dump

# Keep PTX for debugging
CUTE_DSL_KEEP_PTX=1
```

## Performance Notes

- FA4 is optimized for Hopper (SM90) and Blackwell (SM100) GPUs
- For Ampere (SM80) GPUs, consider using Flash Attention 2 or 3
- Kernels are JIT-compiled at runtime; first call will be slower
- Use the cache to avoid recompilation across runs
- Use FP16 for best performance, BF16 for better numerical stability

## Testing

```bash
# Run all tests
pytest tests/cute/

# Run specific test
pytest tests/cute/test_flash_attn.py -k "test_flash_attn_output"

# Fast two-pass testing (compile then run)
FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -n 64 -x tests/cute/test_flash_attn.py
FLASH_ATTENTION_FAKE_TENSOR=0 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 pytest -x tests/cute/test_flash_attn.py
```

## License

See root LICENSE file.
