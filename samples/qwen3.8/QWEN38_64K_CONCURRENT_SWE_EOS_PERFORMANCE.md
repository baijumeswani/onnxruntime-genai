# Qwen 3.8 27B: concurrent 64K SWE-style requests to EOS on RTX Spark

## Scope and result

The ORT GenAI **INT4 / INT8-KV DFlash2** stack completed **15 requests**
across cohorts of **[1, 2, 4, 8] staggered requests**, each with **65,536 input
tokens**, running to **natural EOS**. There was no 512-token cap or forced
minimum answer length.

At eight concurrent requests, the common decoding interval delivered
**94.06 tok/s aggregate** (2.84x the
single-request value), with median TTFT **391.28 s**.
Aggregate throughput is not per-user delivery speed. Runtime provenance and the
measurement definitions are documented below.

**All rows use a local GenAI EOS-classification patch, not unmodified upstream `main`.** The qualified base and patch provenance are documented below.

Every cohort demonstrated its requested concurrent decoding width: overlapping
decoded turns, and `Run` calls that returned tokens from all N requests while
recording N speculative rounds but only one target and one drafter forward.
There was at least one second during which all N requests were decoding.
The largest observed overlap was **8**. No measured cohort reported
`CAPACITY_BLOCKED`; these are not smaller batches with excess requests held
behind an insufficient KV pool.

The cache was expanded to **2,688 target-block-equivalent blocks** and held
constant for all four points. "No queueing" here means no cache-capacity-driven
waiting: normal arrival handoff, prefill scheduling, and compute contention
remain included in latency. Event-level concurrency is not a claim that each
individual CUDA kernel has a particular batch dimension.

Hardware: Windows ARM64, NVIDIA RTX Spark N1X / GB10, SM121, 48 SMs, CUDA 13.4.
Only this inference stack ran on the GPU. All measured rows use 64K inputs.
Earlier smaller-cache, capped-output, and paused attempts are excluded.

## Throughput

| Staggered requests | Output tokens | Effective prompt tok/s | End-to-end output tok/s | Aggregate visible-span tok/s | All-decoding-overlap tok/s | All-decoding overlap, s | Peak decoded-turn overlap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,194 | 792.80 | 14.73 | 33.11 | 33.11 | 66.24 | 1 |
| 2 | 4,180 | 754.86 | 15.52 | 24.15 | 48.43 | 60.89 | 2 |
| 4 | 11,449 | 746.99 | 21.61 | 26.37 | 69.68 | 84.73 | 4 |
| 8 | 21,901 | 742.32 | 22.82 | 25.29 | 94.06 | 97.24 | 8 |

**Effective prompt tok/s** is total input tokens divided by the interval from
the first arrival until every request has produced its first token. It includes
queueing and interleaved generation, and is not pure prefill-kernel throughput.

**End-to-end output tok/s** is total useful output divided by the time from the
first actual arrival until the last request finishes. It includes prompt
processing, stagger gaps, application handoff, engine queueing, and generation.

**Aggregate visible-span tok/s** is `sum(output_tokens - 1)` divided by the span
from the cohort's first visible token to its last. It still includes intervening
prefills, queueing and gaps; it is not isolated decode-kernel throughput.

**All-decoding-overlap tok/s** measures only the common interval from the last
request's first token until the first request's last token. Its numerator counts
tokens strictly after the interval start and at or before its end. `n/a` means
there was no common interval, not zero throughput. Short overlap intervals should
not be treated as robust steady-state measurements.

## Request latency and perceived decode speed

| Requests | Median TTFT, s | Max TTFT, s | Median request decode tok/s | Min request decode tok/s | Max arrival-to-submission, s | Cohort makespan, s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 82.66 | 82.66 | 33.11 | 33.11 | 0.00 | 148.91 |
| 2 | 134.46 | 172.64 | 19.84 | 15.72 | 8.07 | 269.28 |
| 4 | 224.68 | 347.93 | 9.46 | 5.78 | 8.03 | 529.91 |
| 8 | 391.28 | 699.28 | 6.05 | 2.29 | 8.20 | 959.77 |

TTFT starts at the producer's actual arrival, not at a delayed engine API call.
Per-request decode tok/s is `(output_tokens - 1) / (last_token_time - first_token_time)`.
It includes time spent serving competing prefills after a request's first token,
so it can be much lower than an overlap-only aggregate rate.

There is no Python-visible admission timestamp in this build. The raw records
separate **arrival-to-submission** from **submission-to-first-token**, but the
latter combines engine queueing and prefill. Submission is not called admission.
These are descriptive medians/maxima of small cohorts, not production p99 estimates.

## Cache sizing for concurrent 64K requests

One persistent service configuration was used for every row:
`max_batch_size=8`, `num_blocks=2688`, page size 256,
`max_scheduled_tokens=8192`, seven DFlash2 proposals, graph replay disabled,
and the original greedy draft selector. The service's configured maximum batch
size did not change between rows; the **offered cohort size** changed.

For this model, a target block costs 8 MiB and a BF16 drafter block costs 5 MiB.
The drafter's 2,048-token window requires
`ceil((2048 + 2*8) / 256) + 1 = 10` ring blocks per configured request slot.
At eight slots, that is 80 blocks / 400 MiB. Therefore:

```text
Target-block-equivalent budget = 2688 * 8 MiB = 21504 MiB (21 GiB)
Fixed DFlash2 ring reservation  = 8 * 10 * 5 MiB = 400 MiB
Calculated target pool         = (21504 - 400) / 8 = 2638 blocks
Target token slots             = 2638 * 256 = 675328
One 64K prompt                 = 65536 / 256 = 256 blocks
Eight 64K prompts              = 2048 blocks / 524288 tokens
Additional target token slots  = 151040, before transient draft headroom
```

Cache planning allowed 16,384 generated tokens per concurrent request, plus
conservative draft headroom. **That allowance is not an output-token limit.**
The only generation safety bound was the model's total-context limit:
`262144 - 65536 - 16 = 196592` possible new tokens per request. All requests
ended at EOS before reaching that bound.

The final observed lengths of every cohort, conservatively treated as if every
completed sequence plus draft headroom were retained at once, fit this pool.
The earlier 1,032-block-equivalent pool did not fit four 64K prompts and was
therefore replaced before this qualifying campaign.

The qualified source reserves the currently committed whole sequence during
chunked prefill, not the entire future maximum-generation budget. The paged
planner skips requests that temporarily do not fit and selects an executable
subset. Cache occupancy is not exposed through this Python API; the capacity
calculation above is source/configuration-derived, not a live occupancy reading.
`CAPACITY_BLOCKED` notices are not a count of every queued request: a step may
execute a subset while other requests remain deferred without emitting a notice.
For that reason, zero notices alone is not used as proof: actual all-request
decode overlap and sufficient capacity for observed sequence lengths are also
required.

The hybrid engine also removes optional draft rows from **mixed prefill/decode
steps**, because packed recurrent operators cannot capture intermediate draft
checkpoints while sharing a step with prefill. Thus later long prefills can
temporarily reduce speculative acceleration for already-decoding requests.

These behaviors were inspected at the exact qualified local GenAI commit
`4809ff9dd2292c64ffe9388f03a14da9b865b550`, in
`src\engine\scheduler.cpp`, `src\engine\paged_key_value_cache.cpp`,
`src\engine\cache_manager.cpp`, and `src\dflash2_drafter.cpp`.

## Memory footprint

| Concurrent requests | Samples during cohort | Peak sampled dedicated GPU memory, GiB | Peak sampled shared GPU memory, GiB | Peak sampled GPU committed memory, GiB | Minimum sampled free system RAM, GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 31.25 | 42.17 | 73.42 | 34.85 |
| 2 | 9 | 31.25 | 42.17 | 73.42 | 34.89 |
| 4 | 18 | 31.25 | 42.17 | 73.42 | 35.46 |
| 8 | 31 | 31.25 | 42.17 | 73.42 | 35.17 |

These are Windows `GPUProcessMemory` counters for the owned benchmark processes,
sampled about every 30 seconds. Columns are independent sampled peaks and need
not occur at the same instant. They are not a CUDA allocator breakdown or a
measurement of paging traffic. In particular, sufficient logical KV capacity
does not prove that all pages remain in dedicated GPU memory. The larger cache,
fixed-state pools, model weights, and workspaces can change memory behavior
relative to the earlier smaller-pool batch-one reports.

## Speculation and observed execution overlap

| Requests | Accepted / proposed drafts | Acceptance | Max token-emitting requests in one Run | Full-width speculative steps | Capacity/retry notices |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,554 / 4,480 | 34.69% | 1 | 640 | 0 |
| 2 | 2,931 / 8,687 | 33.74% | 2 | 454 | 0 |
| 4 | 8,270 / 21,917 | 37.73% | 4 | 448 | 0 |
| 8 | 16,019 / 39,606 | 40.45% | 8 | 340 | 0 |

Acceptance uses **accepted / proposed**, not the higher accepted / evaluated
quantity used in some earlier ORT reports. A full-width speculative step returns
tokens from all N requests with exactly N new speculative rounds, one target
forward, and one drafter forward. This distinguishes packed engine execution
from serially issuing N independent target calls. These are engine-counter and
event observations, not Nsight proof of every CUDA kernel's batch dimension.
Per-step forward and speculative-round counter deltas are saved.

## Workload and arrival policy

- Eight distinct offline issue templates cover cache diagnostics, close during
  streaming, stop strings, scheduler fairness, speculative counters, cancellation,
  sampling-state isolation, and guidance validation.
- Cohort N uses the first N templates. Every prompt is exactly 65,536 tokens and
  contains unique source sections from the frozen repository snapshot, not repeated
  filler. The same prompt IDs are reused across cohort sizes.
- A separate producer process schedules arrivals at **0, 1, ..., N-1 seconds**.
  Maximum observed producer lateness was **0.77 ms**.
  With 64K prefills this is a near-burst workload, not a steady-state arrival-rate
  sweep. Source/model initialization and warmups are outside these schedules.
- One engine-owner process performs all engine/request API calls and dequeues
  producer arrivals between `Engine.run` calls. It never calls request APIs
  concurrently with a running engine. Actual submission delay is recorded.
- Warmups used 4K/16K inputs with 16 outputs, including multi-request warmups.
  They are not measured rows. One warmed engine was retained across the sweep;
  remaining first-use effects are not subtracted from measurements.
- Greedy sampling, thinking disabled, no custom stops, no forced minimum output.
  Every measured request returned `FinishReason.EOS`, not an output-budget limit.
  Native usage reported zero cached prompt tokens for all measured requests.
- One cohort per size, no repetition or confidence interval. Increasing N changes
  the issue mix as well as concurrency; response lengths and acceptance differ.
  These are **SWE-bench-style workloads**,
  not official SWE-bench tasks, correctness scores, or a pure batch-kernel scaling test.
  Generated text/patches were saved, never executed.

## Per-request results

| Cohort | Request | Issue | Outputs | TTFT, s | Decode tok/s | Arrival-to-submission, s | Finish |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | cache-capacity-diagnostics | 2194 | 82.66 | 33.11 | 0.00 | EOS |
| 2 | 1 | cache-capacity-diagnostics | 2720 | 96.29 | 15.72 | 0.00 | EOS |
| 2 | 2 | close-during-streaming | 1460 | 172.64 | 23.96 | 8.07 | EOS |
| 4 | 1 | cache-capacity-diagnostics | 2763 | 95.95 | 6.97 | 0.00 | EOS |
| 4 | 2 | close-during-streaming | 1467 | 181.18 | 5.78 | 8.03 | EOS |
| 4 | 3 | stop-string-offline-response | 2699 | 268.17 | 11.96 | 7.03 | EOS |
| 4 | 4 | decode-prefill-fairness | 4520 | 347.93 | 25.25 | 6.03 | EOS |
| 8 | 1 | cache-capacity-diagnostics | 2909 | 94.04 | 3.62 | 0.00 | EOS |
| 8 | 2 | close-during-streaming | 1494 | 177.76 | 2.29 | 8.20 | EOS |
| 8 | 3 | stop-string-offline-response | 4800 | 262.17 | 6.97 | 7.20 | EOS |
| 8 | 4 | decode-prefill-fairness | 4904 | 347.33 | 8.05 | 6.20 | EOS |
| 8 | 5 | speculative-counter-contract | 1346 | 435.23 | 3.69 | 5.20 | EOS |
| 8 | 6 | cancel-and-reuse-request | 1528 | 525.28 | 5.14 | 4.20 | EOS |
| 8 | 7 | per-request-sampling-state | 3259 | 614.80 | 10.97 | 3.20 | EOS |
| 8 | 8 | guidance-validation-lifetime | 1661 | 699.28 | 12.92 | 2.20 | EOS |

## Reproduction and raw evidence

**These rows use an isolated patched GenAI runtime**, not the unchanged
binary used in the preceding batch-one reports. Base commit:
`4809ff9dd2292c64ffe9388f03a14da9b865b550`.

Change: Capture accepted terminal EOS at draft commit, before shared batched-sampler scratch can be compacted/overwritten. Stage completion uses immutable transaction-local EOS state; preserves stop-string > turn-limit > EOS > context-limit precedence and clears state at all existing transaction/turn/close resets. Adds first/middle real-EOS shared-slot regression with a surviving request, repeat staging, direct/queued rollback, retry, and turn reset.

Patch SHA-256: `2ac6d2d68befb164598e5694d6b763fcc45dadd36380e9ccaf7a7a1fb0d47f0d`.
Every cohort in this table was remeasured in the same warmed process with this
runtime. Earlier unpatched measurements, including the four-request attempt
with unexpected `MAX_SESSION_TOKENS` classifications, are excluded; no recorded
finish reasons were relabeled. The original qualified runtime remains intact.

In the excluded four-request attempt, three requests reported
`MAX_SESSION_TOKENS` at sequence lengths far below the configured context
limit. The native regression reproduced a reporting defect: accepted draft EOS
was staged, surviving requests compacted and overwrote the shared sampling
buffer, and a second completion-staging pass reread the overwritten token.
The correction retains the actual accepted EOS in transaction-local state;
it does not increase a context limit, impose a new stop rule, or change the
target/drafter inference algorithm. This is a local correction, not a claim
that current upstream `main` already contains the fix.


ORT `9f913ae524b50217e9f7c09ce35b363832a44928`, the CUDA EP, target graph, and
drafter graph are unchanged from the preceding qualified reference. Native
binary hashes, patch provenance where applicable, and all service settings
are retained in the run manifest and `validated-summary.json`.

Local adaptations:
`tools\prepare_staggered_swe_prompts.py`,
`tools\benchmark_staggered_swe.py`, `tools\report_staggered_swe.py`, and
`tools\sample_benchmark_memory.ps1`.
The existing prompt builder retains its original 64K minimum by default; the
new preparation tool explicitly permits shorter prompts for excluded warmups.

From `engine-sample`, using fresh output paths:

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
$env:PATH = "$(Resolve-Path ..\cuda);$env:PATH"
.\.venv-latest-main-benchmark\Scripts\python.exe tools\benchmark_staggered_swe.py `
  --contexts 65536 --counts 1 2 4 8 `
  --runtime-manifest "runtimes\genai-eos-classification\manifest.json" `
  --until-eos --arrival-interval 1 --num-blocks 2688 --require-full-concurrency `
  --max-batch-size 8 --skip-eos-followups --output-dir results\staggered-repeat
```

Once model loading starts, sample the benchmark's owned Python processes from a
second terminal in the same directory:

```powershell
$workers = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
  Where-Object { $_.CommandLine -like '*benchmark_staggered_swe.py*' -and
                 $_.CommandLine -like '*results\staggered-repeat*' } |
  Select-Object -ExpandProperty ProcessId)
.\tools\sample_benchmark_memory.ps1 -ProcessIds $workers `
  -OutputPath "$(Get-Location)\results\staggered-repeat\memory.jsonl"
```

The sampler exits after those processes exit. Once both commands finish:

```powershell
.\.venv-latest-main-benchmark\Scripts\python.exe tools\report_staggered_swe.py `
  --run-dir results\staggered-repeat --output QWEN38_64K_CONCURRENT_SWE_EOS_REPEAT.md
```

The isolated runtime's local `runtimes\genai-eos-classification\manifest.json`
contains the build commands and binary hashes; `source.patch` beside it contains
the native correction. The original qualified Python environment is not replaced.

Raw evidence is in `results\staggered-swe-20260911\run-64k-eos-corrected`: aggregate results,
per-cohort `result.json`, per-request tokens/timestamps/finish reasons, responses,
step counters, notices, `memory.jsonl`, and `validated-summary.json`. The frozen prompt manifest
is under `results\staggered-swe-20260911\prompts`.
