# Qwen 3.8 27B latest-main batch-1 performance on RTX Spark

See [QWEN38_DFLASH2_DRAFTER_DISTILLATION.md](QWEN38_DFLASH2_DRAFTER_DISTILLATION.md)
for the one-page drafter conclusions and deployment recommendations.

## Executive summary

The two INT4/INT8-KV DFlash2 packages are the clear decode-throughput leaders.
With their full seven-token draft width enabled, they deliver approximately
**1.9-2.4x** the decode throughput of the standard INT4/INT8-KV model across
4K-262K contexts. The shifted-tap and original DFlash2 packages remain
effectively tied: shifted is 1.7% faster by geometric-mean decode rate, while
original has a 0.27 percentage-point higher mean acceptance rate. Those
differences are below the confidence supported by a single long-context run.

The standard INT4 model has the best TTFT and prompt throughput. NVFP4 DFlash2
is faster than the standard model during decode, but it trails both INT4
DFlash2 packages. Ollama MTP is competitive at short and medium contexts, then
drops to 8.11 tok/s at 262K.

On a high-acceptance code-copy workload, the qualified INT4 DFlash2 model
reaches **78.73 decode tok/s** at 4K with width seven. This demonstrates the
model/runtime's potential to exceed **60 decode tok/s** when the workload
provides sufficiently predictable draft tokens; it is an upper-bound result,
not representative source-analysis throughput.

**Recommendation:** retain
`qwen3.8-27b-int4-int8-kv-dflash2-tooling-validation2` as the currently
qualified handoff package. The shifted-tap graph is the correct forward
architecture for PR #2538, but its unrelated EOS-token difference and stale
manifest should be normalized before it replaces the qualified package.

## Qualified runtime

| Component | Version or revision |
| --- | --- |
| Machine | NVIDIA RTX Spark N1X / GB10, SM121, Windows ARM64 |
| CUDA | 13.4 |
| cuDNN | 9.25.1.1 |
| ONNX Runtime | `9f913ae524b50217e9f7c09ce35b363832a44928`, version 1.30.0 |
| ONNX Runtime GenAI base | `400d5d2a5b41c7d026982ea05f393f9f98ecab13` |
| GenAI PR #2533 | local commits `8eddd0f7`, `4809ff9d` |
| Ollama | 0.33.2, CUDA 13 backend |
| Batch size | 1 |
| Output length | 128 tokens |
| Contexts | 4,096; 16,384; 65,536; 131,072; 262,000 prompt tokens |
| ORT PR #32409 | **Not applied** |

ORT PR #32410 is present through current `main`. No ORT #32408 or #32409
commit was cherry-picked.

### ORT 1.29 compatibility result

ONNX Runtime 1.29.0 from PyPI was tested first, as requested. It is not
sufficient for this latest model/runtime combination:

- GenAI built against current ORT requests C API version 30, while ORT 1.29
  exposes API versions through 29.
- Rebuilding current GenAI against the ORT 1.29 SDK avoids that API mismatch,
  but model loading then fails because the 1.29 core does not contain the
  `com.microsoft.VarlenCausalConvWithState` schema.

The current CUDA plugin declares compatibility with older cores, but a plugin
kernel cannot compensate for a schema absent from the core. Consequently, this
qualification uses the source-built latest-main ORT 1.30 core, CUDA plugin, and
GenAI wheel as one coherent stack.

## Models

| Label | Model directory | Package size | Drafting |
| --- | --- | ---: | --- |
| INT4 DFlash2 shifted | `qwen3.8-27b-int4-int8-kv-dflash2-shifted` | 17.12 GiB | DFlash2, width 7 |
| INT4 DFlash2 original | `qwen3.8-27b-int4-int8-kv-dflash2-tooling-validation2` | 17.12 GiB | DFlash2, width 7 |
| NVFP4 DFlash2 | `qwen3.8-27b-nvfp4-int8-kv-dflash2` | 23.89 GiB | DFlash2, width 7 |
| INT4 standard | `qwen3.8-27b-int4-int8-kv` | 15.88 GiB | None |
| Ollama MTP | `qwen3.8:27b`, Q4_K_M | 16.52 GiB manifest payload | MTP, width 4 |

Greedy decoding was enforced for every DFlash2 request with per-turn
`do_sample=False`, `top_k=1`, and 128 maximum generated tokens. This overrides
the NVFP4 package's stochastic default. Ollama used temperature 0, top-k 1,
flash attention, and Q8 K/V cache.

All DFlash2 measurements explicitly set the top-level runtime policy
`speculative.max_draft_tokens` to 7. `model.dflash2.num_draft_tokens: 7`
describes the graph's maximum output capability; it does not override GenAI's
generic runtime default of four draft tokens.

## Workload

The prompts are built from real source files, not repeated synthetic filler.
The corpus contains current GenAI engine implementation and C++ tests, with
DFlash2, scheduler, request, engine, and cache files prioritized. Each prompt
ends with a SWE-bench-style task asking the model to identify a production
inference bug, explain it, and produce a minimal patch and regression tests.

The same text and prompt hash was used by every model:

| Prompt tokens | SHA-256 |
| ---: | --- |
| 4,096 | `fa3c6d88beb8e4b8be9dae00145143e3efdaa9112e53ddfd1f820a563eef75cc` |
| 16,384 | `e79805a9923544c998ec0720b5e031ea1af1b9394a1fcbb5859063f9fb1758b8` |
| 65,536 | `e71a003be620b31182e3931fe3dcab01427a430431cc4a50ee078a91baa502b7` |
| 131,072 | `abae84cab6cd3acaf7d9788fcae3fb7630743f861913d6cbb9ed3bb5bc210651` |
| 262,000 | `68864a9823ea60d01bb6043068cd4c036f3ffead16baa9de5a16f03bb4e4dfc4` |

For the ORT engine, 4K and 16K warmup requests absorb model prepack and the
single-chunk/chunked-prefill kernel initialization paths. Model-load time is
excluded. The shifted 16K and original 4K cells were remeasured after their
one-time initialization outliers.

Ollama changes runner context capacity for each point and therefore reloads the
runner. Its comparison TTFT subtracts the reported 5.2-6.2 second
`load_duration`; the unadjusted API timings remain in the raw JSON.

## Results

### TTFT

Seconds from request submission to first token. Lower is better.

| Context | INT4 shifted | INT4 original | NVFP4 | INT4 standard | Ollama MTP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 4.28 | 4.22 | 4.80 | **4.18** | 7.96 |
| 16K | 18.03 | 18.14 | 20.39 | **16.79** | 33.63 |
| 64K | 83.59 | 84.20 | 103.37 | **78.55** | 138.08 |
| 128K | 200.47 | 213.45 | 226.64 | **187.68** | 343.41 |
| 262K | 704.86 | 711.25 | 740.54 | **507.77** | 1,251.05 |

### Prompt throughput

Prompt tokens per second. Higher is better.

| Context | INT4 shifted | INT4 original | NVFP4 | INT4 standard | Ollama MTP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 956.10 | 970.59 | 853.90 | **978.99** | 516.59 |
| 16K | 908.55 | 903.20 | 803.41 | **975.69** | 488.17 |
| 64K | 784.06 | 778.29 | 634.00 | **834.37** | 475.40 |
| 128K | 653.81 | 614.06 | 578.32 | **698.39** | 382.11 |
| 262K | 371.71 | 368.37 | 353.79 | **515.99** | 209.54 |

### Decode throughput

Generated tokens per second after the first token. Higher is better.

| Context | INT4 shifted | INT4 original | NVFP4 | INT4 standard | Ollama MTP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 30.45 | **31.88** | 25.03 | 14.98 | 18.37 |
| 16K | **34.71** | 33.33 | 22.43 | 14.72 | 18.78 |
| 64K | **28.48** | 27.50 | 21.45 | 13.70 | 20.11 |
| 128K | 24.44 | **25.62** | 17.19 | 12.31 | 14.10 |
| 262K | **21.78** | 19.70 | 20.06 | 10.21 | 8.11 |

### Decode speedup over standard INT4

| Context | INT4 shifted | INT4 original | NVFP4 | Ollama MTP |
| ---: | ---: | ---: | ---: | ---: |
| 4K | 2.03x | **2.13x** | 1.67x | 1.23x |
| 16K | **2.36x** | 2.26x | 1.52x | 1.28x |
| 64K | **2.08x** | 2.01x | 1.57x | 1.47x |
| 128K | 1.99x | **2.08x** | 1.40x | 1.15x |
| 262K | **2.13x** | 1.93x | 1.96x | 0.79x |

Geometric-mean decode speedup over standard INT4 is 2.110x for shifted INT4
DFlash2, 2.080x for original INT4 DFlash2, and 1.610x for NVFP4 DFlash2.

### High-acceptance code-copy ceiling

This prompt asks the model to continue code already present in its context.
It is useful for measuring the ceiling of speculative decoding, but it is not
a proxy for typical coding-agent or SWE-bench throughput.

| Model | Context | Draft width | Acceptance | Output/target forward | Decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qualified INT4 DFlash2 | 4,058 | 7 | 97.79% | 7.31 | **78.73 tok/s** |
| Qualified INT4 DFlash2 | 4,058 | 4 | 99.51% | 5.02 | 48.35 tok/s |

The width-seven result establishes that this stack can exceed **60 decode
tok/s** on high-acceptance workloads. The representative source-analysis
qualification remains approximately 20-35 tok/s, with a geometric mean near
27 tok/s for the INT4 DFlash2 models.

### Speculative acceptance

Each DFlash2 cell shows `accepted/evaluated` rate and decode output tokens per
speculative round. The latter excludes prefill target forwards.

| Context | INT4 shifted | INT4 original | NVFP4 | Ollama MTP acceptance |
| ---: | ---: | ---: | ---: | ---: |
| 4K | 66.1% / 2.84 | 68.5% / 3.05 | **72.4% / 3.37** | 46.3% |
| 16K | **71.5% / 3.37** | 70.5% / 3.20 | 67.2% / 2.98 | 52.8% |
| 64K | 66.9% / 2.91 | 65.3% / 2.78 | **68.0% / 3.12** | 54.7% |
| 128K | 64.0% / 2.72 | **65.3% / 2.78** | 60.8% / 2.51 | 48.8% |
| 262K | 65.9% / 2.78 | 66.1% / 2.84 | **68.8% / 3.20** | 53.4% |

The original graph has 67.16% mean acceptance and the shifted graph has 66.89%
on this sweep. This reverses the tiny direction seen in the width-four run and
is further evidence that the packages are tied on this workload. The current
sweep does not establish a statistically significant decode-speed difference
between the two INT4 DFlash2 graphs.

### Draft-width impact

The initial latest-main sweep unintentionally used GenAI's generic width-four
default. Correcting the runtime policy to width seven improves geometric-mean
decode by **9.4%** for shifted INT4, **6.9%** for original INT4, and **3.5%**
for NVFP4 on the representative workload. Per-cell effects range from -3.8% to
+16.7%, so width seven is not a universal win for every context/model pair.

The earlier 60-82 tok/s results were high-acceptance ceilings rather than
representative patch-generation rates. On an apples-to-apples 4K code-copy
prompt, width seven reaches **78.73 tok/s**, 97.79% acceptance, and 7.31 output
tokens per target forward, versus **48.35 tok/s** under the accidental
width-four cap. On the more difficult source-analysis prompts, 60-72%
acceptance limits width seven to approximately 17-35 tok/s.

Width seven proposes and verifies up to seven draft tokens per speculative
round instead of four. This reduces the number of expensive target forwards
when acceptance is high, but increases target verification rows, temporary
activation/logit work, and scheduler occupancy when drafts are rejected.
Consequently, it should be treated as a model/deployment policy rather than a
new runtime-wide default.

The three qualified local DFlash2 packages now include this top-level model
configuration:

```json
"speculative": {
  "max_draft_tokens": 7
}
```

Keep `model.dflash2.num_draft_tokens: 7` as the graph capability declaration.
GenAI should retain its conservative generic default of four for unspecified
models. A future Engine request option should allow callers to lower or tune
the width for other devices, batch sizes, or low-acceptance workloads; the
current Engine Python request/turn API does not expose that override.

## Cache budgeting and PR #2533

The model uses 256-token cache blocks. The maximum request needs 1,024 target
blocks, while the 2,048-token DFlash2 sliding window needs eight blocks.
Qualification therefore used an explicit total budget of **1,032 blocks**.

This is the exact case fixed by GenAI PR #2533: the already allocated windowed
drafter pool must be deducted before assigning the remaining explicit budget
to the target. All three DFlash2 models and the standard model completed the
262K request with the explicit budget.

Using a broad free-memory fraction instead was not production-safe on this
unified-memory machine. A 0.9 factor exhausted working memory, while a 0.5
factor reserved far more KV memory than batch 1 required and caused severe
prefill paging. Exact block budgeting restored 4K prompt throughput from
12.77 tok/s to 869.85 tok/s in the diagnostic run.

## Build notes

Current ORT main required two narrowly scoped Windows ARM64 build adjustments:

1. Quantized MoE kernels were disabled for this dense Qwen build.
   `deep_gemm_sm90.cu` uses `__int128_t`, which is unavailable with the MSVC
   ARM64 host compiler. Source references were guarded by the existing
   `USE_FP4_QMOE`/`USE_FP8_QMOE` feature macros.
2. Current GenAI unconditionally enables `/Qspectre`. The installed ARM64
   Visual Studio toolchain lacks the Spectre-mitigated libraries, so a
   `GENAI_ENABLE_SPECTRE` option was restored with a default of `ON` and set to
   `OFF` only for this local benchmark build.

Neither adjustment changes dense-Qwen inference kernels or other-device
defaults. They are build compatibility changes, not performance patches.

## Interpretation

- **Best decode:** the two INT4 DFlash2 models are effectively tied and remain
  near or above 2x the standard model through 262K.
- **Best prefill:** standard INT4 has the lowest TTFT at every context.
- **Shifted taps:** no material performance regression is visible. The mapping
  is architecturally correct, but the package should not supersede the current
  handoff until EOS and manifest cleanup is complete.
- **NVFP4:** it is not the winning model in this mainline-only stack. The open,
  device-scoped FP4 scheduling work from ORT #32408 was not applied, so these
  results should not be treated as the ceiling for GB10 NVFP4.
- **Ollama MTP:** good short-context decode performance, but prompt throughput
  and 262K decode performance fall substantially behind ORT GenAI.

## Reproducibility and limitations

- Qualified wheels:
  - `source-builds\ort-main-20260907\Release\Release\dist\onnxruntime_gpu-1.30.0-cp313-cp313-win_arm64.whl`
    (`EB361B6722669ED748F309C1F1B9C246634C97DF7B39C6F5EBE7D6B2746EF7B9`)
  - `source-builds\wheels-latest-main\onnxruntime_ep_cuda13-0.2.0.dev20260907235005-py3-none-win_arm64.whl`
    (`D8228F1FF60FFFDC523E66C5771FD3F6937B33A59109397079447D099303CC6E`)
  - `source-builds\genai-main-pr2533-20260907\Release\wheel\onnxruntime_genai_cuda-0.16.0.dev0-cp313-cp313-win_arm64.whl`
    (`39D752285E8CA67B075647E4FBBE19E46F4A2A8ED74DABFCEBA47EDE9C26DD3C`)
- Benchmark environment:
  `.venv-latest-main-benchmark`.
- Raw corrected matrix:
  `results\latest-main\combined-results.json`
- Individual results:
  `results\latest-main\int4-int8kv-dflash2-shifted-width7.json`,
  `results\latest-main\int4-int8kv-dflash2-original-width7.json`,
  `results\latest-main\nvfp4-int8kv-dflash2-width7.json`,
  `results\latest-main\int4-int8kv-standard.json`, and
  `results\latest-main\ollama-mtp.json`.
- Width-four diagnostic files are retained alongside the corrected matrix to
  document the configuration error.
- Prompt corpus:
  `results\latest-main\representative-code-prompts.json`.
- Each cell is one measured run after warmup. The 262K points make a
  multi-repeat matrix expensive, so differences below approximately 3-5%
  should be treated as ties until repeated on another RTX Spark.
- This is a performance qualification. It does not replace the existing
  deterministic quality report or a SWE-bench quality assessment.
