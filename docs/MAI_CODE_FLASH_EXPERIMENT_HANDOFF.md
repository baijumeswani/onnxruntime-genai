# MAI Code 1 Flash experiment handoff

This document is the cross-machine continuation point for the RTX Spark/GB10
MAI Code 1 Flash work. The experimental CUDA code, benchmark harness, frozen
prompt corpus, summarized profile exports, raw benchmark JSON, measured
results, qualification status, and next experiments are all available from the
branches below.

## Remote branches

| Repository | Branch | Commit | Purpose |
|---|---|---|---|
| `baijumeswani/onnxruntime` | `baijumeswani/mai-fp16-xqa-h128` | `2c5e9df3a3c562516cf140ebb876107c0fa3cace` | Native FP16 paged-XQA decode specialization for head size 128 and GQA group size six |
| `baijumeswani/onnxruntime-genai` | `baijumeswani/mai-code-flash-rtx-spark-report` | Use the branch tip containing this file | Harness, frozen inputs, raw measurements, profile summaries, analysis, and optimization roadmap |

The ORT experiment is based on
`0be8bea92f163fdf07adf8a39a926ef8ed582e2e`. It is deliberately isolated from
the primary ORT checkout.

## Included GenAI artifacts

- `tools\benchmark_mai_code_flash_eos.py`
  - Exact tokenizer-context construction.
  - Natural-EOS or fixed-output generation.
  - Cold model-load, tokenizer, and Engine initialization timing.
  - TTFT, prompt throughput, decode throughput, finish reason, and token IDs.
  - GPU, process, and system-memory sampling.
  - CUDA profiler-range capture.
  - Alternate decoder, cache-block, provider-option, and session-option
    overrides.
- `tools\data\mai-code-flash-representative-code-prompts.json`
  - Frozen source corpus used to construct the exact-context prompts.
- `docs\results\mai-code-flash\`
  - Baseline, XQA, XQA-disabled control, runtime W4, and rejected INT8-KV JSON.
- `docs\results\mai-code-flash\tokens\`
  - Exact baseline, XQA, and XQA-disabled output token IDs used for trajectory
    and matching-prefix comparisons.
- `docs\results\mai-code-flash\profiles\`
  - Nsight Systems CUDA-kernel and CUDA-API CSV summaries.
  - The much larger `.nsys-rep` and SQLite databases are not committed.
- `docs\MAI_CODE_FLASH_RTX_SPARK_BENCHMARK.md`
  - Original 4K/16K/32K qualification report.
- `docs\MAI_CODE_FLASH_RTX_SPARK_PROFILE_ANALYSIS.md`
  - Complete 4K-128K baseline, profile attribution, XQA results, and
    correctness status.
- `docs\MAI_CODE_FLASH_OPTIMIZATION_PLAN.md`
  - Ranked optimization portfolio, bandwidth bounds, experiment matrix, and
    stop criteria.

The MAI model itself and the CUDA provider DLL are not committed. The model
must be copied or downloaded separately, and the plugin should be rebuilt
against the runtime used on the destination machine.

## Reference environment

- Windows ARM64 RTX Spark / NVIDIA GB10 / SM121.
- Python 3.13.15 ARM64.
- CUDA and cuDNN 13.4.
- Visual Studio 2026 / MSVC 14.51.
- ONNX Runtime 1.30.0.
- ONNX Runtime GenAI 0.16.0-dev at
  `4809ff9dd2292c64ffe9388f03a14da9b865b550`.
- Model: MAI Code 1 Flash, W4 weights, FP16 KV.
- Batch size one, CUDA graphs enabled.
- Prefill chunk and maximum scheduled tokens: 512.
- Paged-cache block size: 128.
- QMoE row tile: 64.

## Build the experimental ORT CUDA plugin

Clone the ORT fork and check out the exact experiment:

```powershell
git clone https://github.com/baijumeswani/onnxruntime.git
Set-Location onnxruntime
git checkout 2c5e9df3a3c562516cf140ebb876107c0fa3cace
git submodule update --init --recursive
```

Configure and build an ARM64 shared runtime with the CUDA EP plugin. Replace
the paths with the CUDA installation and desired build directory on the new
machine:

```powershell
.\build.bat `
  --config Release `
  --build_dir C:\build\mai-xqa-h128 `
  --arm64 `
  --build_shared_lib `
  --parallel `
  --skip_tests `
  --use_cuda `
  --cuda_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4" `
  --cudnn_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4" `
  --cmake_extra_defines `
    CMAKE_CUDA_ARCHITECTURES=121 `
    onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON `
    onnxruntime_USE_FP4_QMOE=OFF
```

The explicit architecture is required: the first build targeted SM120 and
could not execute on the SM121 device. Native FP4 QMoE is disabled because MAI
uses INT4 QMoE and the optional FP4 path produced unresolved SM120-only
launcher symbols when linked for `sm_121a`.

The plugin target can be rebuilt incrementally with:

```powershell
cmake --build C:\build\mai-xqa-h128\Release `
  --config Release `
  --target onnxruntime_providers_cuda_plugin `
  -- /m
```

The reference DLL was:

```text
C:\Users\Baiju\workspace\engine-sample\source-builds\mai-xqa-h128\Release\Release\onnxruntime_providers_cuda.dll
SHA-256: 0246582643ABE62C22B530C15BC57807828309DE32F98E3AF65DBDA66F6932FF
```

The hash is only a reference for the original build environment; a valid
rebuild on another machine may differ.

## Run the benchmark

Create an ARM64 Python environment containing the matching
`onnxruntime-genai`, `numpy`, and `psutil` packages. Run from the GenAI report
checkout:

```powershell
python tools\benchmark_mai_code_flash_eos.py `
  --model C:\models\mai-code-flash `
  --provider C:\build\mai-xqa-h128\Release\Release\onnxruntime_providers_cuda.dll `
  --source-prompts tools\data\mai-code-flash-representative-code-prompts.json `
  --output-dir C:\results\mai-xqa-h128 `
  --context 4096 `
  --context 16384 `
  --context 32768 `
  --context 65536 `
  --context 131072 `
  --output-tokens 512 `
  --num-blocks 1029
```

Use the same model, prompt corpus, output count, and cache capacity when
comparing variants. Run a short warmup and at least three measured repetitions
for decisions; the original table is a single-run qualification baseline.

Useful controls:

```powershell
# Prove that a rebuilt plugin without XQA reproduces baseline token IDs.
$env:ORT_ENABLE_XQA = "0"

# Sweep long-context XQA split scheduling. The current automatic choice is six
# subsequences for batch one, eight KV heads, and the working 48-SM estimate.
$env:XQA_NB_SUB_SEQ = "8"
```

Remove the environment variable to restore automatic selection:

```powershell
Remove-Item Env:ORT_ENABLE_XQA -ErrorAction SilentlyContinue
Remove-Item Env:XQA_NB_SUB_SEQ -ErrorAction SilentlyContinue
```

For an Nsight range capture, add `--capture decode` and point `--cudart` to the
CUDA runtime DLL. Capture exactly one context per process.

## Measured decode results

All rows below generated 512 tokens.

| Context | FP16 FlashAttention | FP16 XQA H128 | Speedup | First token mismatch |
|---:|---:|---:|---:|---:|
| 4K | 44.72 tok/s | 47.48 tok/s | 1.06x | 177 |
| 16K | 38.12 tok/s | 44.20 tok/s | 1.16x | 38 |
| 32K | 28.99 tok/s | 40.04 tok/s | 1.38x | 458 |
| 64K | 21.34 tok/s | 34.54 tok/s | 1.62x | 122 |
| 128K | 13.40 tok/s | 27.08 tok/s | 2.02x | 101 |

The 32K node trace confirms
`H128::grp6_fp16_fp16_paged::kernel_mha`. All 42 XQA attention layers total
approximately 4.61 ms/token, compared with approximately 13.89 ms/token for
the FlashAttention path in the earlier trace.

## Qualification status and limitations

The XQA branch is an experimental performance candidate, not a production
fix:

- The plugin with `ORT_ENABLE_XQA=0` matches baseline token IDs exactly.
- Enabling XQA eventually changes the greedy token trajectory at every measured
  context.
- The difference is isolated to FP16 arithmetic/reduction ordering in XQA.
- Native H128 FP16 operator-level tolerance tests have not been added.
- Coding-quality and long-context retrieval gates have not been run.
- The new specialization handles one-token paged decode. H128 multi-token
  speculative verification does not yet have a matching native XQA
  specialization and may use a different attention path.
- Do not claim exact parity or production readiness until the tests below pass.

Required qualification:

1. Compare native FP16 H128/group-six PagedAttention with the fallback and a
   CPU/reference implementation.
2. Cover short context, long split-K context, local-window masking, fragmented
   page tables, and CUDA graph replay.
3. Record maximum absolute and relative output error and reject NaNs, cache
   corruption, or page-boundary discrepancies.
4. Run coding tasks and long-context retrieval tests before accepting
   tolerance-based numerical differences.

## Recommended continuation order

1. Sweep `XQA_NB_SUB_SEQ=2,4,6,8,12` at 32K, 64K, and 128K. Keep a setting only
   if it improves 64K by at least 5% without worsening numerical behavior.
2. Add native H128 FP16 XQA operator tests and decide whether the observed
   numerical error is within the project tolerance.
3. Adapt the Engine benchmark to use caller-supplied prompt-lookup proposals:
   `engine.max_draft_tokens_per_proposal()` and
   `request.set_draft_tokens(...)`. The current maximum is seven drafts.
4. Test n-gram sizes 3/4/5, widths 2/4/7, and chained lookup on/off. Require at
   least two committed tokens per target verification and positive net
   throughput.
5. Build LM-head-only W8 and W4 variants. The current FP16
   `[3584, 200064]` head costs approximately 5.65 ms/token and reads 1.34 GiB.
6. Build calibrated FP8 KV for only the seven global-attention layers.
7. Prototype exact Q/K RMSNorm-to-PagedAttention and residual-Add/RMSNorm
   graph fusions.

The realistic path to more than 50 tok/s is:

- 4K-16K: LM-head compression plus exact graph fusion.
- 32K: add qualified XQA or calibrated compressed KV.
- 64K: combine attention/KV improvements with useful speculation.
- 128K: treat 50 tok/s as a workload-dependent stretch target requiring both
  reduced global-KV traffic and multiple accepted tokens per target
  verification.
