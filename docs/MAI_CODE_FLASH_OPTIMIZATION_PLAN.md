# MAI Code 1 Flash Performance Optimization Plan

## Objective and acceptance gates

Target: exceed 50 batch-1 decode tokens/s at 4K, 16K, 32K, 64K, and 128K on RTX Spark/GB10.

Every candidate must pass:

1. Three unprofiled repetitions after warmup; report median and range.
2. Exact token parity against FP16 unless the candidate intentionally changes numerics.
3. For lossy candidates, a coding-quality gate plus long-context retrieval and EOS-behavior checks.
4. No hidden unified-memory migration, allocation failure, or context-capacity reduction.
5. Separate reporting of TTFT, decode throughput, model load, and peak memory.

## Measured constraints

### GB10 platform implications

NVIDIA identifies GB10 as compute capability 12.1 (SM121). Public RTX Spark
material describes an integrated Blackwell GPU with coherent LPDDR5 memory at
up to 300 GB/s; the closely related DGX Spark GB10 hardware guide publishes
273 GB/s LPDDR5X. This is much less bandwidth than HBM-class Blackwell server
parts, so batch-1 decode is dominated by streamed weights and long-context KV
traffic well before it approaches peak FP4 Tensor Core throughput.

The public product material lists 6,144 CUDA cores. Approximately 48 SMs is a
useful working estimate if the client Blackwell mapping remains 128 cores/SM,
but it is not a directly published RTX Spark SM count. Kernel decisions must use
measured occupancy and device attributes rather than treating 48 as contractual.

Sources:

- [RTX Spark Porting Guide](https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/latest/overview.html)
- [CUDA GPU compute-capability table](https://developer.nvidia.com/cuda/gpus)
- [DGX Spark hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA LLM inference optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

### Current throughput

| Context | Decode | Latency |
|---:|---:|---:|
| 4K | 44.72 tok/s | 22.36 ms/token |
| 16K | 38.12 tok/s | 26.23 ms/token |
| 32K | 28.99 tok/s | 34.49 ms/token |
| 64K | 21.34 tok/s | 46.87 ms/token |
| 128K | 13.40 tok/s | 74.62 ms/token |

### Per-token fixed work from the 32K node trace

| Component | Approximate latency | Approximate weight traffic |
|---|---:|---:|
| 273 dense W4 projections | 6.86 ms | 1.49 GiB |
| FP16 3584x200064 LM head | 5.65 ms | 1.34 GiB |
| 21 QMoE layers, eight active experts | 4.02 ms | 0.97 GiB active-weight lower bound |
| RMSNorm | 1.19 ms | Activation traffic |
| Local attention | 1.01 ms | Bounded 512-token window |
| Router/selection | 0.28 ms | Small |

The dense W4, LM-head, and active QMoE measurements imply effective bandwidth around 215-240 GiB/s. Kernel tuning can recover inefficiency, but these paths are fundamentally weight-bandwidth-bound at batch size one.

### Long-context attention traffic

Seven global layers read FP16 K and V for eight KV heads of width 128:

```text
bytes/token = context * 7 layers * 8 KV heads * 128 channels * 2 (K,V) * 2 bytes
```

This is approximately:

- 0.11 GiB at 4K
- 0.88 GiB at 32K
- 1.75 GiB at 64K
- 3.50 GiB at 128K

Even at an ideal 273 GB/s, the 128K global-cache read alone is approximately 13 ms/token. Combined with the current fixed-weight traffic, exact FP16 execution has a hardware bandwidth floor above 20 ms/token. Therefore, 50 tok/s at 128K requires reducing bytes per accepted token or amortizing a target pass across multiple accepted tokens.

An optimistic bandwidth-only bound, excluding all activation traffic, cache
miss inefficiency, computation, reductions, sampling, and launch overhead, is:

| Context | Fixed + global KV traffic | Ideal max at 273 GB/s | Ideal max at 300 GB/s |
|---:|---:|---:|---:|
| 4K | ~3.91 GiB/token | ~65 tok/s | ~71 tok/s |
| 16K | ~4.24 GiB/token | ~60 tok/s | ~66 tok/s |
| 32K | ~4.68 GiB/token | ~54 tok/s | ~60 tok/s |
| 64K | ~5.55 GiB/token | ~46 tok/s | ~50 tok/s |
| 128K | ~7.30 GiB/token | ~35 tok/s | ~38 tok/s |

Because real kernels cannot sustain the published peak for every streamed
component and still perform the omitted work, exact FP16 50 tok/s at 64K has no
practical margin and at 128K is impossible under this byte model. This table is
the reason the roadmap changes from kernel tuning at short context to KV
compression and multi-token target verification at long context.

## Priority 0: finish the native FP16 XQA experiment

**Status:** performance validation complete; numerical and quality qualification pending.

Add native FP16 paged XQA for `head_size=128`, GQA group size six. The existing plugin only enables native FP16 paged XQA for head size 256, while MAI uses head size 128.

Validation:

1. Confirm XQA dispatch in a node-level trace.
2. Compare exact 128-token sequences at 4K and 32K.
3. If parity passes and 32K attention improves by at least 15%, run 64K and 128K.
4. Stop if it is slower than FlashAttention or diverges materially.

Expected scope: attention only. It cannot solve the fixed 17-18 ms/token compute/weight floor.

### Measured result

The SM121 plugin was built with native FP16 XQA for head size 128 and group size
six. The optional native FP4 QMoE build was disabled because MAI uses INT4
QMoE, and CUDA 13.4 produced unresolved SM120-only FP4 launcher symbols when
the plugin was compiled at `sm_121a`.

| Context | FlashAttention baseline | XQA H128 | Speedup | First token mismatch |
|---:|---:|---:|---:|---:|
| 4K | 44.72 tok/s | 47.48 tok/s | 1.06x | 177 |
| 16K | 38.12 tok/s | 44.20 tok/s | 1.16x | 38 |
| 32K | 28.99 tok/s | 40.04 tok/s | 1.38x | 458 |
| 64K | 21.34 tok/s | 34.54 tok/s | 1.62x | 122 |
| 128K | 13.40 tok/s | 27.08 tok/s | 2.02x | 101 |

All rows generated 512 tokens. TTFT remained comparable because XQA is a
decode path. A 32K CUDA graph node trace confirms
`H128::grp6_fp16_fp16_paged::kernel_mha` ran for the PagedAttention nodes. The
trace attributes approximately 4.61 ms/token to all 42 XQA attention layers,
versus approximately 13.89 ms/token for global plus local FlashAttention in the
earlier 32K trace, a roughly 3.0x attention-kernel reduction.

The rebuilt plugin with `ORT_ENABLE_XQA=0` reproduces baseline tokens exactly,
isolating the token differences to the XQA arithmetic path. Existing ORT XQA
tests compare with numerical tolerances, commonly `rtol=atol=5e-3`, rather than
requiring bitwise equality. Before upstreaming or enabling this path:

1. Add a native FP16 H128/group-six PagedAttention operator test against the
   fallback and CPU reference at short, split-K long, local-window, fragmented
   page-table, and CUDA-graph replay shapes.
2. Record maximum absolute/relative attention-output error and reject NaN,
   cache, page-table, or boundary discrepancies.
3. Run the coding-quality and long-context retrieval gate because greedy token
   trajectories eventually differ at every measured context.

Artifacts:

- `results\mai-code-flash-opt\fp16-xqa-h128-4k-32k-512-sm121\results.json`
- `results\mai-code-flash-opt\fp16-xqa-h128-16k-128k-sm121\results.json`
- `results\mai-code-flash-opt\fp16-xqa-h128-disabled-control\results.json`
- `results\mai-code-flash-profile-20260918\mai-32k-xqa-h128-sm121-nodes.nsys-rep`

## Priority 1: exact graph and kernel fusion

### 1. Fuse Q/K RMSNorm into PagedAttention

The graph currently executes, per layer:

```text
Q projection -> Reshape -> Cast -> RMSNorm -> Cast -> Reshape
K projection -> Reshape -> Cast -> RMSNorm -> Cast -> Reshape
```

`PagedAttention` already accepts `q_norm_weight`, `k_norm_weight`, and `qk_norm_epsilon` and has a fused QK-Norm/RoPE prologue. The shipped graph leaves those inputs empty.

Rewrite the graph to feed pre-normalized Q/K projections and the norm weights directly to PagedAttention. Remove:

- 84 Q/K RMSNorm nodes.
- 168 Q/K Cast nodes.
- Redundant reshapes where the adapter allows it.

Gate: exact output parity. Expected decode gain: 0.3-0.8 ms/token plus lower graph-node overhead.

### 2. Fuse residual Add + RMSNorm

There are 84 residual Add nodes. Each sum feeds both the next RMSNorm and a later residual edge, matching `SkipSimplifiedLayerNormalization` with its optional residual-sum output.

Rewrite the graph or extend the optimizer so all eligible pairs use the CUDA fused operator.

Gate: exact output parity. Expected decode gain: 0.1-0.4 ms/token.

### 3. Add CUDA W4 shared-input projection fusions

The development branch contains `MatMulNBitsQkv` and `MatMulNBitsMlp` graph fusions only for WebGPU block-size-32 models. MAI is CUDA, W4 block size 64.

Implement CUDA variants for:

- RMSNorm + Q/K/V projections: replace 42 norms/casts and 126 W4 launches.
- RMSNorm + dense MLP gate/up projections: replace 21 norms/casts and 42 W4 launches.

The primary benefit is fused activation loading, fewer graph nodes, and better scheduling. Weight traffic remains unchanged, so expected gain is bounded.

Gate: bitwise or exact-token parity. Expected gain: 0.5-1.5 ms/token.

### 4. Tune the W4 M=1 block-size-64 kernel

The current kernel averages 24.9 microseconds over about 274 launches/token and consumes 6.86 ms/token. Measure:

- Memory-load vectorization and transaction efficiency.
- Scale-load reuse.
- Register pressure and occupancy on 48-SM GB10.
- Projection-shape-specific launch geometry.
- Persistent/shared-input multi-projection kernels.

Do not repeat generic CTA-N=4 or launch-bounds experiments already known to regress ORT INT4 GEMV. Require at least 5% end-to-end gain, not only a microbenchmark win.

## Priority 2: reduce the fixed LM-head cost

The FP16 LM head reads 1.34 GiB and costs 5.65 ms for every output token. This is the largest removable short-context cost.

### Candidate A: W4 or W8 LM head

The GenAI builder supports quantized LM heads; this package excluded the final projection.

Build:

- W8 blockwise LM head as the lower-risk candidate.
- W4 block-size-64 LM head as the maximum-bandwidth candidate.
- GPTQ/AWQ-style calibration if RTN quality is insufficient.

Expected:

- W8: save approximately 2-3 ms/token.
- W4: save approximately 3.5-4.5 ms/token.
- Reduce resident memory by roughly 0.7-1.0 GiB.

This is the most likely path to exceed 50 tok/s at 4K.

Quality gate:

- Deterministic prompt comparison.
- HumanEval+/MBPP+ pass@1 or an equivalent local coding suite.
- Repository-level code-edit prompts with build/test grading.
- EOS length distribution and refusal/repetition checks.

### Candidate B: exact or certified vocabulary reduction

Investigate a hierarchical/candidate LM head:

1. Cluster vocabulary rows offline and store each centroid plus a residual-norm
   bound.
2. Score centroids, then compute exact logits only for rows in candidate
   clusters.
3. Certify the winner with an upper bound such as
   `q dot c + norm(q) * max_residual`; fall back to the full head whenever an
   unvisited cluster can still win.

This has higher implementation complexity but can preserve exact greedy output while avoiding most of the 200,064-column projection on easy tokens.

TensorRT-LLM's Gemma4 implementation provides a concrete centroid-shortlist
precedent, although it is not a generic exact implementation. Treat it as a
design reference rather than directly reusable MAI support.

Source:
[TensorRT-LLM Gemma4 centroid shortlist](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/models/modeling_gemma4.py)

## Priority 3: reduce long-context KV traffic

### 1. Calibrated FP8 KV

FP8 E4M3 halves KV bytes like INT8 but generally offers a better floating-point dynamic range. PagedAttention and XQA source support FP8 KV on Blackwell.

Build per-channel or otherwise calibrated scales from representative code and long-context data. Do not reuse the supplied INT8 graph's fixed per-tensor scale of 0.02.

Expected long-context attention improvement: 1.4-1.9x, depending on kernel efficiency. Also frees roughly 1.75 GiB of cache allocation at 128K.

### 2. Recalibrated per-channel INT8 KV

The supplied INT8 model diverges from token zero and reached EOS after 19 tokens at 32K. Treat it as a kernel/performance proof only.

Generate separate per-head/per-channel K and V scales using activation statistics, percentile clipping, and a held-out quality set. Compare symmetric INT8 with FP8.

### 3. Mixed KV precision

Evaluate:

- FP8 K + FP8 V.
- FP16 K + FP8 V.
- Recent-window FP16 plus older-context FP8.
- FP16 local-layer cache and quantized global-layer cache.

The last option focuses quantization only on the seven bandwidth-scaling layers and retains exact local-attention behavior.

### 4. Global-attention split scheduling

For FlashAttention/XQA, tune the split count and page traversal for:

- 48 query heads, eight KV heads.
- Head size 128.
- Page size 128.
- The measured SM count, with approximately 48 as an initial estimate only.
- KV lengths 32K, 64K, and 128K.

Use Nsight Compute to measure DRAM throughput, L2 hit rate, sectors/request, achieved occupancy, and tensor utilization. Optimize for wall time, not occupancy alone.

TensorRT-LLM recommends multi-block generation when
`batch_size * num_heads < multiprocessor_count`. MAI has enough Q heads in
aggregate, but each GQA/KV-head partition and split-K reduction may still
underfill this small GPU. Sweep multi-block thresholds rather than copying
large-GPU XQA heuristics.

The current ORT XQA implementation already supports an `XQA_NB_SUB_SEQ`
environment override. Its automatic rule is:

```text
min(max(1, SM_count / (batch_size * KV_heads)), ceil(max_seq_len / 256))
```

For batch one, eight KV heads, and the working 48-SM estimate, the automatic
choice is six subsequences at long context. Benchmark `2, 4, 6, 8, 12` at
32K/64K/128K before changing code. Six produces exactly 48 primary CTAs; eight
or twelve may improve tail balance or memory-level parallelism at the cost of
more scratch reduction.

Sources:

- [TensorRT-LLM GPT attention](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html)
- `onnxruntime\contrib_ops\cuda\bert\xqa\mha_impl.cuh`

### 5. Query-aware sparse reads for the seven global layers

The 35 local layers are already bounded to 512 tokens, so any sparse-attention
prototype should target only the seven global layers. Use the existing
128-token page table as the selection unit:

1. Record per-page attention mass and retrieval accuracy on 64K/128K coding
   traces.
2. Build a cheap query-to-page score using page summaries.
3. Run exact attention over selected pages plus recent and sink pages.
4. Fall back to all pages when a confidence or error bound is not met.

This is a high-risk, quality-gated path, but it is the highest-ceiling
non-speculative way to reduce 128K traffic. Stop if preserving the retrieval
gate requires more than 8K retained tokens per global layer or if page
selection plus fallback costs erase a 20% attention gain.

## Priority 4: exact speculative decoding

Compression alone has little margin to exceed 50 tok/s at 128K. Amortizing one target verification pass across multiple accepted tokens is the highest-ceiling end-to-end strategy.

### 1. Prompt-lookup decoding

Implement an n-gram drafter over the existing prompt and generated suffix:

- No additional model weights.
- Exact target verification preserves output.
- Code commonly repeats identifiers, syntax, indentation, imports, and nearby text.
- Long contexts provide a large candidate corpus.

The current GenAI implementation already supplies the required pieces:

- `NGramLookup` with incremental exact history indexing and chained lookup.
- `Engine.max_draft_tokens_per_proposal()`.
- `Request.set_draft_tokens(...)`.
- Device-side verification and accepted-prefix commit in the Engine.

Automatic n-gram proposing is limited to the regular `Generator`, but
caller-supplied Engine proposals are available to continuous batching. The
current engine-wide maximum is seven drafts, constrained by recurrent-state
checkpoint capacity, so test n-grams `3, 4, 5`, draft counts `2, 4, 7`, and
chained lookup on/off. Track lookup coverage, evaluated and accepted drafts,
mean committed tokens per target pass, verification latency, and net tok/s.

At 128K, the current 13.4 tok/s baseline needs about 3.7 accepted output tokens per baseline-equivalent target cost to exceed 50 tok/s. Prompt lookup is the lowest-risk way to test whether the workload provides that acceptance.

If fixed-order n-grams have useful coverage but short continuations, add a
suffix-automaton proposer over the same committed history. It remains
model-free and target-verified while finding variable-length repeated
substrings.

### 2. External draft model

Evaluate a tokenizer-compatible small code model if available. The draft must remain resident without evicting target weights or causing unified-memory migration.

Required metrics:

- Draft tok/s.
- Acceptance length.
- Target multi-token verification cost.
- Additional resident memory.
- End-to-end throughput.

### 3. Self-speculative early exit

Explore an auxiliary draft head from an intermediate MAI layer. This requires training/calibration but avoids a second transformer. It is only worthwhile if the auxiliary head and state fit in the current memory envelope.

## Priority 5: orchestration and CUDA graph pipeline

### 1. Remove duplicate per-token synchronization

The trace shows one `cudaDeviceSynchronize`/stream wait per token. The wait mostly exposes real GPU work, so eliminating it alone is not a 2x optimization.

Refactor the greedy path to:

1. Run the target with `disable_synchronize_execution_providers=1`.
2. Keep sampling and next-token state on the same CUDA stream.
3. Enqueue one asynchronous device-to-host token copy.
4. Wait on one event only when the token must be delivered to the caller.

Expected gain: 0.2-1.0 ms/token, larger at short context.

### 2. One-token-ahead graph pipeline

Reserve B1 cache state up front and keep sequence length, slot mapping, sampled token, and EOS state device-resident. Queue the next CUDA graph replay before the host consumes the previous token.

Requirements:

- Preserve ordered streaming events.
- Mask or discard speculative one-token-ahead work after EOS/cancellation.
- Keep graph input/output addresses stable.
- Add cancellation, failure, and request-lifetime tests.

This overlaps host handling and launch latency but does not reduce weight/KV bytes.

### 3. Multi-token verification graphs

For prompt lookup or another drafter, capture separate graph buckets for verification lengths 2, 4, 8, and 16. Stable buckets avoid recapture and let W4/QMoE execute with larger M, improving utilization while amortizing weight reads.

### 4. Blackwell graph and dependent-launch experiments

CUDA graphs already cover pure decode in the current engine and are not an
unexplored feature toggle. Focus on increasing replay coverage, keeping all
addresses stable, and eliminating capture blockers rather than merely enabling
graphs again.

For kernels owned by ORT, evaluate Programmatic Dependent Launch only where a
secondary kernel has useful setup work that can overlap the tail of its
predecessor. Do not expect it to hide streamed-weight or KV bandwidth.

Sources:

- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [Constant-time CUDA graph launch](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)
- [Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)

## Priority 6: prefill and TTFT

The 32K prefill profile is 75.8% QMoE GEMM.

### 1. Context-sized cache allocation

The default 1,025 blocks reserve 128K global-cache capacity even for 4K-32K qualification. Use the minimum safe block count per test:

```text
ceil((prompt + output + safety) / 128)
```

This can free several GiB for larger prefill workspaces without changing model output.

### 2. Chunk and QMoE tactic sweep

With context-sized cache allocation, test:

- Chunk size: 256, 512, 1,024, 2,048.
- `max_scheduled_tokens` matched to the chunk.
- QMoE row tile: 32, 64, 128, 256, disabled.

Capture selected QMoE tactics and reject settings that trigger memory migration or reduce decode residency.

### 3. QMoE graph cleanup

Each QMoE layer has a dynamic Shape/Gather/Unsqueeze/Concat/Reshape chain to produce `[-1, 3584]`. Replace it with a simpler dynamic reshape contract or fold shape construction into QMoE.

This is a launch/orchestration cleanup; QMoE GEMM tactic and weight bandwidth remain the main TTFT costs.

### 4. Blackwell QMoE kernel path

For larger prefill chunks, evaluate grouped persistent expert GEMMs and native
NVFP4/FP8 Tensor Core paths. NVIDIA reports large grouped-GEMM gains on
datacenter Blackwell, but those figures are not a GB10 forecast. On RTX Spark,
the experiment succeeds only if end-to-end TTFT improves without spilling the
working set or increasing unified-memory migration.

For custom kernels, test TMA bulk staging and 128-byte shared-memory swizzles
where the tile shapes provide meaningful reuse. These mechanisms are more
promising for chunked prefill/grouped GEMM than for weight-streaming M=1 GEMV.

Sources:

- [Blackwell FP4 support in CUDA 12.8](https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/)
- [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [CUDA asynchronous copies and TMA](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)

For decode, benchmark the top QMoE FC1/FC2 shapes against the available
SM120/SM121 FlashInfer and CUTLASS paths before starting a new kernel. Public
TensorRT-LLM code uses FlashInfer for SM121 MoE decode and CUTLASS above a
prefill-row threshold, while some FP8 grouped-GEMM paths remain disabled on
SM120/SM121 pending validation. This makes validation and end-to-end evidence
more important than assuming datacenter Blackwell results transfer to GB10.

Source:
[TensorRT-LLM SM12x fused MoE](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/moe/fused_moe/fused_moe_cute_dsl_b12x.py)

## Priority 7: model loading and residency

The graph references approximately:

- 62.0 GiB of prepacked QMoE expert weights.
- 1.49 GiB of dense W4 projection weights.
- 1.34 GiB of FP16 LM-head weights.
- 1.70 GiB of other initializers.

Actions:

1. Keep a persistent service process; do not pay the approximately 90-second load per request.
2. Record CUDA unified-memory migration and page-fault events explicitly in future profiles.
3. Place external data contiguously by execution order where the loader does demand paging.
4. Test parallel/read-ahead loading and preferred-location advice.
5. After LM-head/KV compression, reinvest freed memory in cache and prefill workspace rather than increasing unrelated arenas.

Hardware-coherent memory does not make demand migration free. CUDA documents
more limited unified-memory behavior on Windows than on Linux platforms with
full system-allocated-memory support. Prefer explicit device allocations and
pinned asynchronous staging in latency-critical paths. Use managed-memory
prefetch/advice only after a trace demonstrates a benefit.

Sources:

- [CUDA unified-memory basics](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)
- [CUDA unified-memory performance tuning](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)

For small hot structures such as page tables and recurrent metadata, evaluate
an L2 access-policy window while leaving streamed weights and KV outside the
persisting region. The benefit must be measured because reserving L2 can hurt
the much larger streaming working set.

Source: [CUDA L2 cache control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)

## Work sequence

| Phase | Work | Stop/advance criterion |
|---|---|---|
| A | Finish FP16 XQA H128 | Advance if parity passes and attention improves >=15% |
| B | Sweep `XQA_NB_SUB_SEQ` at 32K-128K | Keep only >=5% 64K gain with stable numerics |
| C | Engine prompt lookup, widths 2/4/7 | >=2 committed tokens per target pass and positive net gain |
| D | Fuse Q/K norm into PagedAttention; fuse residual+norm | Exact parity and >=0.5 ms/token combined |
| E | Quantize LM head W8/W4 | Coding gate passes; target >50 tok/s at 4K |
| F | Calibrate FP8 and per-channel INT8 global KV | Quality gate passes; >=1.4x attention speedup |
| G | CUDA W4/QMoE kernel tuning | >=5% end-to-end gain per accepted change |
| H | Async graph/token pipeline | Correct event/cancel/EOS semantics and >=3% gain |
| I | Prefill chunk/QMoE tuning | Lower TTFT without decode or memory regression |
| J | Global-layer page selection | >=20% attention gain with retrieval gate passing |

## Expected target path

- **4K:** quantized LM head plus exact graph fusions should be sufficient for >50 tok/s.
- **16K-32K:** add FP16 XQA or calibrated FP8/INT8 KV.
- **64K:** KV compression plus attention tuning; speculative decoding likely needed for comfortable margin.
- **128K:** bandwidth analysis indicates speculative decoding or another method that amortizes target weight/KV reads across multiple accepted tokens is required. Session options or isolated kernel tuning cannot provide the required 3.7x improvement.
