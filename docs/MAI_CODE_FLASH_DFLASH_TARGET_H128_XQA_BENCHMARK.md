# MAI Code Flash DFlash target: FP16 H128 XQA on RTX Spark

Date: 2026-09-25

This is a **target-only** batch-one comparison on the runnable MAI Code Flash
DFlash package, not a speculative DFlash throughput result. The earlier
[MAI experiment](MAI_CODE_FLASH_EXPERIMENT_HANDOFF.md) used a different original
target with different weights and output tokens. Its performance numbers must
not be substituted for this model's results.

## Matched benchmark

The same ONNX Runtime 1.31 CUDA EP plugin, built from ORT revision
`b195a19c8563d5c475dec21bc7d687447ff2a19e` plus a local four-file
port of the FP16 H128/group-six specialization, was used in both runs.
The change adds an FP16-cache H128 XQA instantiation and routes eligible
`PagedAttention` decode calls to it. GenAI revision
`7f750ddbcbfb4c93d99d23d595c40294904f1b06` and the installed ORT
1.31 wheel were unchanged.

Both fresh processes used Windows ARM64 RTX Spark (GB10/SM121), CUDA 13.4,
identical frozen coding prompts, batch size one, 1,216 paged-cache blocks,
512 greedy output tokens, target-only mode (drafter disabled), and CUDA graphs
with `ORT_ENABLE_CUDNN_FLASH_ATTENTION=0`. Each ran a 256-input/16-output
warmup before measuring contexts in ascending order. The sole runtime
difference was `ORT_ENABLE_XQA_NATIVE_KV=0` versus `1`. Nsight Systems
confirmed 4,074 executions of
`H128::grp6_fp16_fp16_paged::kernel_mha` in a separately profiled 4K
XQA-enabled decode.

| Input tokens | TTFT off (s) | TTFT on (s) | Prompt off (tok/s) | Prompt on (tok/s) | Decode off (tok/s) | Decode on (tok/s) | Decode gain | First differing output token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 6.34 | 6.31 | 646 | 649 | 46.29 | **48.14** | 1.04x | 229 |
| 16,384 | 27.42 | 27.44 | 597 | 597 | 37.95 | **43.66** | 1.15x | 85 |
| 32,768 | 57.12 | 58.86 | 574 | 557 | 29.08 | **40.20** | 1.38x | 27 |
| 65,536 | 121.84 | 120.16 | 538 | 545 | 20.05 | **34.48** | 1.72x | 54 |
| 131,072 | 265.00 | 265.26 | 495 | 494 | 12.57 | **26.98** | 2.15x | 145 |

Prompt rate is input tokens divided by TTFT, not an isolated prefill-kernel
measurement. All outputs hit the 512-token cap rather than EOS. Each cell is
a single run, so there is no variance estimate. The source prompts had
matching hashes between XQA-off and XQA-on processes; model loading is
excluded from TTFT.

**Result:** XQA improves long-context target-only decode but does not reach
the **>50 tok/s at every context from 4K to 128K** goal. It does not establish
a DFlash speculative speedup. Greedy outputs differ after the positions
shown: 282, 422, 483, 457, and 368 of the 512 output positions differ at
ascending contexts. These counts include downstream differences after the
first mismatch and do not isolate their cause. Numerical and coding-quality
equivalence have **not** been established. The experimental kernel should
receive operator-level numerical tests and first-divergence logit/quality
analysis before it is considered qualified for upstream use.

The local qualification workspace retains the full JSON results, output
token IDs, prompts, build log, and Nsight trace. Model weights, binaries,
large traces, and the experimental ORT source port are not published with
this documentation update.
