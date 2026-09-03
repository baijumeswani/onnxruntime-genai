# Local reproduction evidence

The builder was exercised from a clean worktree using:

- Target:
  `qwen3.8-27b-int4-int8-kv`
- DFlash2 source:
  `qwen3.8-27b-nvfp4-int8-kv-dflash2`
- Retained ORT quantization:
  symmetric INT4, block size 32
- Qualified conversion environment:
  Python 3.13.15, ONNX 1.22.0, NumPy 2.4.2, and ONNX Runtime 1.30.0

`MODEL_BUILD_MANIFEST.json` records these versions and the SHA-256 digest of
the exact `matmul_nbits_quantizer.py` implementation used for each build.

The generated model passed `validate_model.py` and loaded successfully through
the ONNX Runtime GenAI continuous-batching Engine with the CUDA EP.

Generated drafter structure:

- 58 `MatMulNBits` nodes.
- One dynamic `/dflash2/selector/pair` `MatMul`.
- 1,332,731,904-byte compact drafter external-data file.

The prior winning experimental artifact had the same operator counts but kept
4,590,690,304 bytes because it retained unused source BF16 ranges and duplicate
shared target weights.

The generated model was compared with the previously qualified winner on:

- Technical scheduler reasoning.
- Python LRU implementation.
- C++ zero-budget correction.

Results:

- 3/3 token-exact outputs.
- 100% positional token agreement.
- The same 2/3 instruction-constraint result; the shared C++ failure is a
  base-model behavior, not a conversion difference.

This file records local evidence and is not a substitute for CI. A production
PR should add model-fixture or integration coverage in an environment where the
large source artifacts are available.
