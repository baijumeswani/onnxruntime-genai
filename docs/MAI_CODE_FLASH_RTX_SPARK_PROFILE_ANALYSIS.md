# MAI Code 1 Flash RTX Spark Profile Analysis

## Executive summary

MAI Code 1 Flash is compute-bound at short context and global-attention-bandwidth-bound at long context on the RTX Spark. CUDA graph replay is working, QMoE weights are already prepacked, and the existing CUDA EP configuration is generally appropriate. Session-option tuning alone is not sufficient to reach 50 batch-1 decode tokens/s from 4K through 128K.

The most important measured result is that the seven global-attention layers add approximately 11.7 ms per generated token between 4K and 32K, while W4 dense projections, the FP16 language-model head, and QMoE remain nearly constant. At 128K, the complete decoder takes 74.62 ms/token. The 50 tok/s target requires at most 20 ms/token.

## Test configuration

- Machine: Windows ARM64 RTX Spark, NVIDIA GB10 (SM121)
- ONNX Runtime: 1.30.0
- ONNX Runtime GenAI: 0.16.0-dev, commit `4809ff9dd2292c64ffe9388f03a14da9b865b550`
- CUDA EP plugin:
  `C:\Users\Baiju\workspace\engine-sample\.venv-latest-main-benchmark\Lib\site-packages\onnxruntime_ep_cuda\onnxruntime_providers_cuda.dll`
- Model: MAI Code 1 Flash, W4 weights with FP16 KV cache
- Batch size: 1
- Prefill chunk size: 512
- Generated tokens: 512 for unprofiled qualification
- CUDA graphs: enabled
- QMoE row tile size: 64
- Profiling: Nsight Systems 2026.3.2 with CUDA graph node tracing

Node-trace profiling adds overhead and is used only for attribution. All throughput values below are from unprofiled runs.

## Complete FP16 baseline

| Prompt context | TTFT | Prompt throughput | Decode throughput | Decode latency | Peak GPU memory |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 6.19 s | 661.30 tok/s | 44.72 tok/s | 22.36 ms/token | 63,540 MiB |
| 16,384 | 26.00 s | 630.12 tok/s | 38.12 tok/s | 26.23 ms/token | 63,544 MiB |
| 32,768 | 55.93 s | 585.83 tok/s | 28.99 tok/s | 34.49 ms/token | 63,550 MiB |
| 65,536 | 112.79 s | 581.04 tok/s | 21.34 tok/s | 46.87 ms/token | 63,711 MiB |
| 131,072 | 257.00 s | 510.00 tok/s | 13.40 tok/s | 74.62 ms/token | 63,715 MiB |

All five requests completed the requested 512 generated tokens. The 128K run used 1,029 cache blocks so the 131,072-token prompt and 512-token output fit in the paged cache.

### Gap to 50 tok/s

| Context | Current latency | Latency reduction required |
|---:|---:|---:|
| 4K | 22.36 ms/token | 2.36 ms/token (10.6%) |
| 16K | 26.23 ms/token | 6.23 ms/token (23.8%) |
| 32K | 34.49 ms/token | 14.49 ms/token (42.0%) |
| 64K | 46.87 ms/token | 26.87 ms/token (57.3%) |
| 128K | 74.62 ms/token | 54.62 ms/token (73.2%) |

## Decode profile

Each decode trace contains 255 post-first-token CUDA graph replays. The traced wall spans were 5.78 seconds at 4K and 9.25 seconds at 32K.

| GPU work across 255 decode steps | 4K | 32K | Context scaling |
|---|---:|---:|---:|
| W4 dense GEMV (`MatMulFloat4BitsKernelM1`) | 1.754 s | 1.749 s | Flat |
| FP16 LM-head GEMM | 1.445 s | 1.442 s | Flat |
| W4 QMoE GEMV and finalize | 1.021 s | 1.025 s | Flat |
| Global FlashAttention | approximately 0.34 s | approximately 3.31 s | +2.98 s |
| Local 512-token FlashAttention | approximately 0.25 s | approximately 0.26 s | Flat |
| Normalization | 0.286 s | 0.302 s | Effectively flat |
| Routing and selection | 0.073 s | 0.071 s | Flat |

The global-attention increase is approximately 11.7 ms per output token and accounts for nearly all of the 4K-to-32K decode regression. The 35 local-attention layers remain bounded by their 512-token window.

The decoder also performs one `cudaDeviceSynchronize` per generated token. The synchronization exposes the graph's completion latency to the host, but removing it cannot eliminate the underlying global-attention GPU work. It should be optimized only after preserving streaming-event and request-lifetime semantics.

### Native FP16 XQA H128 result

A native FP16 paged-XQA specialization for MAI's head size 128 and GQA group
size six was built at `sm_121a` and exercised through the same engine.

| Context | Baseline decode | XQA decode | Speedup | Matching token prefix |
|---:|---:|---:|---:|---:|
| 4K | 44.72 tok/s | 47.48 tok/s | 1.06x | 177 |
| 16K | 38.12 tok/s | 44.20 tok/s | 1.16x | 38 |
| 32K | 28.99 tok/s | 40.04 tok/s | 1.38x | 458 |
| 64K | 21.34 tok/s | 34.54 tok/s | 1.62x | 122 |
| 128K | 13.40 tok/s | 27.08 tok/s | 2.02x | 101 |

The 32K node trace contains
`H128::grp6_fp16_fp16_paged::kernel_mha`. Across the 63 profiled decode steps,
all 42 attention layers total approximately 4.61 ms/token, versus approximately
13.89 ms/token for global and local FlashAttention in the previous trace.

The rebuilt plugin with `ORT_ENABLE_XQA=0` matches the original baseline token
IDs, so the changed trajectories are specific to XQA's FP16 arithmetic order.
This is consistent with ORT's tolerance-based XQA operator tests but means the
candidate is not yet production-qualified. It needs a native FP16 H128
operator-level tolerance test and the coding-quality/long-context gate.

## Fixed-cost findings

### FP16 language-model head

The approximately 5.6 ms/token dense CUTLASS GEMM is the final FP16 language-model head:

```text
[1, 3584] x [3584, 200064]
```

It is not an attention or router projection. Quantizing or otherwise optimizing this projection is the largest single context-independent opportunity. It can help the 4K target substantially but cannot close the 64K or 128K gap by itself.

### W4 `MatMulNBits`

The graph contains 273 `MatMulNBits` nodes with 4-bit, block-size-64 weights. They have no `weight_prepacked` attribute and run `MatMulFloat4BitsKernelM1` during decode.

The installed CUDA plugin contains the `ep.cuda.fpa_intb_gemm` and `ORT_FPA_INTB_GEMM` implementations. Enabling `ep.cuda.fpa_intb_gemm=1` preserved exact output parity for the tested 4K continuation, but produced only 45.55 tok/s versus the 44.72 tok/s baseline. This is not a material improvement without repeated evidence or a profile showing a changed kernel.

### QMoE prepacking

All 21 QMoE nodes already specify:

```text
weights_prepacked=1
expert_weight_bits=4
k=8
swiglu_fusion=1
```

The configured QMoE row tile is 64. Row tiling is primarily a prefill optimization for this graph; batch-1 decode has one input row and does not benefit materially from larger row tiles.

## Prefill profile

The 32K prefill uses 64 chunks of 512 tokens. Its 54.11 seconds of summed GPU kernel time is dominated by W4 QMoE GEMM:

| Category | GPU kernel time | Share |
|---|---:|---:|
| W4 QMoE GEMM | 41.01 s | 75.8% |
| Other GPU kernels | 8.64 s | 16.0% |
| Global attention | 3.17 s | 5.9% |
| Normalization | 0.75 s | 1.4% |
| Dense GEMM | 0.36 s | 0.7% |

QMoE tactic selection and row tiling are therefore the main TTFT opportunities. Testing chunk sizes 256, 512, and 1,024 is worthwhile, but the system has little dedicated GPU-memory headroom, so larger chunks must be qualified for allocation failures and unified-memory migration.

## INT8 KV experiment

The supplied experimental INT8-KV graph was tested with 1,025 cache blocks.

| Context | TTFT | Decode throughput | Result |
|---:|---:|---:|---|
| 4K | 5.74 s | 49.34 tok/s | 256 tokens |
| 16K | 25.23 s | 47.07 tok/s | 256 tokens |
| 32K | 53.55 s | 38.97 tok/s | EOS after 19 tokens |

INT8 KV materially improves bandwidth-sensitive decode, but it fails the deterministic gate: the first generated token differs from FP16 at 4K, 16K, and 32K. The 32K response also terminates after 19 tokens instead of continuing for the requested 256. It must not replace FP16 based only on throughput.

Even an idealized 2x reduction in long-context attention cost would not meet 50 tok/s at 128K because the fixed decoder floor remains too high. INT8 KV is therefore a component of a possible solution, not the complete solution.

## CUDA EP and session-option assessment

| Option or mechanism | Assessment |
|---|---|
| CUDA graphs | Already active and replaying once per decode token; retain |
| `ep.cuda.fpa_intb_gemm` | Correctness-preserving test showed no material speedup |
| QMoE prepacking | Already enabled in all QMoE nodes |
| `ep.cuda.qmoe_row_tile_size` | Useful mainly for prefill; test 32/64/128 |
| TunableOp | Unlikely to affect the dominant custom PagedAttention, QMoE, and `MatMulNBits` kernels |
| `use_tf32` | Not relevant to the dominant FP16 and W4 kernels |
| Arena growth strategy | Can affect allocation and fragmentation, not steady-state kernel latency |
| Larger prefill chunks | Potential TTFT improvement, constrained by nearly saturated GPU memory |
| Per-token host synchronization | Secondary opportunity; cannot remove GPU attention cost |

## Recommended optimization order

1. **Finish native FP16 SM121 XQA H128 qualification.** Performance and dispatch are proven; add tensor-level tolerance tests and run the coding-quality gate because exact-token parity fails.
2. **Fuse Q/K RMSNorm into PagedAttention and residual Add into RMSNorm.** These are the lowest-risk exact graph changes.
3. **Quantize or optimize the FP16 LM head.** Its 5.6 ms/token cost prevents the short-context decoder from reaching 50 tok/s comfortably. A quantized version requires coding-quality validation.
4. **Calibrate FP8 or per-channel INT8 KV for the seven global layers.** The supplied fixed-scale INT8 graph is fast but fails immediate token parity and can alter EOS behavior.
5. **Implement exact prompt-lookup speculative decoding.** Long-context 50 tok/s requires amortizing target weight and KV reads over multiple accepted tokens.
6. **Tune W4 M=1 and QMoE decode kernels for SM121.** Together they cost approximately 10.9 ms/token, but are already near the device's streamed-memory bandwidth.
7. **Tune QMoE prefill tactics and chunk size.** Use context-sized cache allocation to make room for larger chunks and workspaces.
8. **Reduce host synchronization only with correctness tests.** Preserve streaming token delivery, cancellation, failures, and request resource lifetime.

## Feasibility conclusion

The current exact FP16 model cannot reach 50 tok/s at every context through 128K through prepacking, session options, or isolated kernel tuning alone. The fixed dense-W4, active-expert, and FP16-LM-head paths stream approximately 3.8 GiB/token. At 128K, seven global layers add approximately 3.5 GiB/token of FP16 K/V reads. Against the published 273-300 GB/s GB10 memory bandwidth, this exceeds the 20 ms/token budget even before normalization, local attention, routing, sampling, and launch overhead.

At 4K, LM-head compression plus exact graph fusion should be sufficient. At 16K-64K, calibrated KV compression and a better XQA/attention kernel are required. At 128K, the defensible route is an exact speculative scheme such as prompt lookup, so one multi-token target verification amortizes weight and KV traffic across several accepted tokens. Any lossy KV or weight change must pass a coding-quality evaluation rather than a throughput-only gate.

## Artifacts

- Baseline: `results\mai-code-flash-512-20260918\results.json`
- 64K/128K baseline: `results\mai-code-flash-opt\fp16-64k-128k\results.json`
- Runtime-prepack experiment: `results\mai-code-flash-opt\fpa-intb-4k\results.json`
- INT8-KV experiment: `results\mai-code-flash-opt\int8-kv-4k-32k\results.json`
- 4K node trace: `results\mai-code-flash-profile-20260918\mai-4k-decode-nodes.nsys-rep`
- 32K node trace: `results\mai-code-flash-profile-20260918\mai-32k-decode-nodes.nsys-rep`
- 32K prefill trace: `results\mai-code-flash-profile-20260918\mai-32k-prefill.nsys-rep`
- XQA 4K/32K results: `results\mai-code-flash-opt\fp16-xqa-h128-4k-32k-512-sm121\results.json`
- XQA 16K/64K/128K results: `results\mai-code-flash-opt\fp16-xqa-h128-16k-128k-sm121\results.json`
- XQA-disabled control: `results\mai-code-flash-opt\fp16-xqa-h128-disabled-control\results.json`
- XQA 32K node trace: `results\mai-code-flash-profile-20260918\mai-32k-xqa-h128-sm121-nodes.nsys-rep`
- Detailed architecture and implementation roadmap: `MAI_CODE_FLASH_OPTIMIZATION_PLAN.md`
