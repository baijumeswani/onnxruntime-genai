# MAI Code 1 Flash RTX Spark benchmark

Date: 2026-09-18

## Summary

MAI Code 1 Flash loaded and completed the requested batch-size-1 benchmark at
exact 4K, 16K, and 32K prompt lengths. Each measured request generated exactly
512 tokens. Decode throughput decreased from 44.72 tok/s at 4K to 28.99 tok/s
at 32K.

The model occupied almost all dedicated GPU memory: peak sampled usage was
63,574 MiB (62.08 GiB), versus 1,497 MiB before loading the model.

## Results

| Context | Output tokens | Model load | TTFT | Prompt tok/s | Decode tok/s | Request time | Peak GPU memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 512 | 97.68 s | 6.19 s | 661.30 | 44.72 | 17.62 s | 63,540 MiB |
| 16K | 512 | reused | 26.00 s | 630.12 | 38.12 | 39.41 s | 63,544 MiB |
| 32K | 512 | reused | 55.93 s | 585.83 | 28.99 | 73.56 s | 63,550 MiB |

All three requests finished with `FinishReason.MAX_GENERATED_TOKENS`, which is
the expected result for the requested fixed 512-token measurement.

## Load and memory

| Metric | Result |
|---|---:|
| `og.Model` cold load | 97.68 s |
| Tokenizer initialization | 0.80 s |
| Engine initialization | 0.26 s |
| Total model-to-ready time | 98.74 s |
| Baseline GPU memory | 1,497 MiB |
| Peak sampled GPU memory | 63,574 MiB |
| Increment above baseline | 62,077 MiB (60.62 GiB) |
| Peak process resident memory | 17,425.58 MiB (17.02 GiB) |
| Peak system memory used | 35,444.84 MiB (34.61 GiB) |
| Minimum system memory available | 29,527.66 MiB (28.84 GiB) |

GPU memory is sampled from `nvidia-smi` and is device-wide, not isolated to the
benchmark process. The low 1,497 MiB baseline makes the approximately 60.62 GiB
increment a useful estimate of the model and runtime's dedicated-memory
footprint on this otherwise idle device.

## Methodology

- Model:
  `C:\Users\Baiju\workspace\engine-sample\mai-code-flash\mai-code-flash`
- Default graph: `mai-code-1-flash.onnx`
- KV cache: FP16
- Weight representation: W4 `MatMulNBits` and prepacked W4 `QMoE`
- Batch size: 1
- Contexts: exactly 4,096, 16,384, and 32,768 tokenizer tokens
- Output: exactly 512 greedy tokens
- CUDA graphs: enabled by the model configuration
- Prompts: source-grounded ONNX Runtime GenAI code corpus, truncated separately
  for each exact token length
- Decode throughput:
  `(output_tokens - 1) / (last_token_time - first_token_time)`
- Model loading is excluded from TTFT and request throughput.
- A short 256-token prompt/16-token output warmup ran before measurement.
- One measured request was run at each context.

## Earlier natural-EOS observations

Before the scope changed to fixed 512-token outputs, two natural-completion
requests finished successfully:

| Context | Output tokens | TTFT | Decode tok/s | Peak GPU memory | Finish |
|---:|---:|---:|---:|---:|---|
| 4K | 3,067 | 6.26 s | 42.66 | 63,628 MiB | EOS |
| 16K | 2,080 | 26.63 s | 35.57 | 63,632 MiB | EOS |

The unbounded 32K natural-EOS request was stopped at the user's direction and
is not reported as a completed measurement.

## Local qualification artifacts

The raw measurements and generated prompts/responses remain in the local
qualification workspace and are not included in this documentation-only
branch:

- Benchmark harness: `tools\benchmark_mai_code_flash_eos.py`
- Fixed-output results:
  `results\mai-code-flash-512-20260918\results.json`
- Exact prompts and generated responses:
  `results\mai-code-flash-512-20260918`
- Preserved natural-EOS results:
  `results\mai-code-flash-20260918\results.json`

