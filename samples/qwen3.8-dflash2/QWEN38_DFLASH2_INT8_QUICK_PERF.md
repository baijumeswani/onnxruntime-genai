# Qwen 3.8 27B DFlash2 INT8-KV Quick Performance

## Scope

- Machine: RTX Spark N1X, Windows ARM64
- Runtime: ONNX Runtime `1.30.0.dev20260901002`
- ORT GenAI: `0.16.0-dev1001393453`
- CUDA EP plugin: `0.2.0.dev20260901075055`
- Model: `qwen3.8-27b-nvfp4-int8-kv-dflash2`
- Batch size: 1
- Output length: 128 tokens
- DFlash2 draft width: up to 7 tokens
- Dynamic batching: `max_batch_size=1`, `num_blocks=65`
- Contexts: 4K, 8K, and 16K

The benchmark is adapted from
`samples/qwen3.8-dflash2/benchmark.py` on the
`tlwu/qwen_3.8_dflash2_example` branch. A long-answer suffix was added to the
synthetic prompt so every measured request completed the full 128-token decode
window. Both arms used the same target model, prompt, cache type, and greedy
search settings. Each arm was warmed up at the largest context before
measurement.

## Results

| Requested context | Actual prompt tokens | Arm | TTFT (s) | Prompt TPS | Decode TPS | Acceptance | Output tokens / target forward |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 4,093 | DFlash2 | 5.142 | 795.99 | 23.54 | 73.17% | 3.28 |
| 4,096 | 4,093 | Baseline | 5.002 | 818.25 | 10.16 | - | - |
| 8,192 | 8,173 | DFlash2 | 10.560 | 773.98 | 23.29 | 73.39% | 3.20 |
| 8,192 | 8,173 | Baseline | 10.276 | 795.36 | 10.21 | - | - |
| 16,384 | 16,381 | DFlash2 | 21.749 | 753.18 | 22.90 | 75.00% | 3.05 |
| 16,384 | 16,381 | Baseline | 21.154 | 774.38 | 10.09 | - | - |

## DFlash2 Speedup

| Context | Decode speedup | TTFT overhead |
| ---: | ---: | ---: |
| 4K | 2.32x | 2.8% |
| 8K | 2.28x | 2.8% |
| 16K | 2.27x | 2.8% |

DFlash2 sustains approximately 23 decode tokens per second across the tested
range, compared with approximately 10 decode tokens per second for the same
target with its drafter disabled. Acceptance remains stable at 73-75%.

These are quick, single measured runs after warmup. The results are suitable
for an initial performance check but not a substitute for the multi-repeat
qualification used for the standard INT4 models.

## Ollama MTP Comparison

Ollama `qwen3.8:27b` was measured with its built-in MTP drafter enabled, a
maximum draft width of four tokens, FP16 KV cache, and 128 generated tokens.
Independent prompt prefixes prevented reuse of Ollama's automatic prompt
cache. The Ollama model is Q4_K_M GGUF, so this comparison covers complete
serving stacks rather than identical weights.

| Context | ORT DFlash2 prompt TPS | Ollama MTP prompt TPS | ORT DFlash2 decode TPS | Ollama MTP decode TPS | Ollama decode advantage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 795.99 | 687.86 | 23.54 | 26.50 | 12.6% |
| 8K | 773.98 | 640.48 | 23.29 | 28.08 | 20.5% |
| 16K | 753.18 | 598.08 | 22.90 | 25.47 | 11.2% |

| Context | ORT DFlash2 acceptance | Ollama MTP acceptance | ORT wall time | Ollama wall time excluding load |
| ---: | ---: | ---: | ---: | ---: |
| 4K | 73.17% | 54.43% | 10.538 s | 10.800 s |
| 8K | 73.39% | 67.65% | 16.012 s | 17.475 s |
| 16K | 75.00% | 60.96% | 27.294 s | 32.778 s |

Ollama MTP decodes 11-21% faster, but ORT DFlash2 processes prompts 16-26%
faster. The prefill advantage makes ORT's complete request 2-17% faster over
this short 128-token generation window. Longer output lengths would give
Ollama's higher decode rate more opportunity to offset its slower prefill.

Ollama reports prompt evaluation duration rather than event-level TTFT, while
ORT measures submission-to-first-token time. Prompt throughput and wall-clock
results are therefore the better cross-stack indicators; the timing fields are
not identical internal measurements.

## Raw Results

- `results/dflash2-int8-4k-16k-final.json`
- `results/dflash2-int8-baseline-4k-16k.json`
- `results/ollama-mtp-4k-16k.json`
