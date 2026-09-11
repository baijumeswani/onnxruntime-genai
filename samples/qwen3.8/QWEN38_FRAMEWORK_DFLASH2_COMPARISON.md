# Qwen 3.8 27B DFlash2: ORT GenAI versus llama.cpp on RTX Spark

## Result

Both frameworks completed the same three frozen prompts to natural EOS at batch one.
ORT GenAI delivered **1.58x, 2.24x, and 2.32x** llama.cpp's
average decode throughput at 64K, 96K, and 128K input tokens, respectively.
These are single-run deployed-stack comparisons, not a same-quantization kernel
benchmark or an official SWE-bench accuracy evaluation.

Measured September 10-11, 2026 on Windows ARM64, NVIDIA RTX Spark N1X / GB10,
SM121, 48 SMs, CUDA 13.4. Only one inference framework ran at a time.

## Matched-input, natural-EOS results

| Input tokens | Framework | Visible output tokens | TTFT (s) | Prompt tok/s | Decode tok/s | Finish |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 65,536 | ORT GenAI | 2,979 | 81.489 | 804.23 | **36.96** | EOS |
| 65,536 | llama.cpp | 3,294 | 115.370 | 568.05 | 23.44 | EOS |
| 98,304 | ORT GenAI | 2,448 | 140.189 | 701.22 | **38.35** | EOS |
| 98,304 | llama.cpp | 2,228 | 191.367 | 513.69 | 17.09 | EOS |
| 131,072 | ORT GenAI | 5,963 | 205.276 | 638.52 | **34.60** | EOS |
| 131,072 | llama.cpp | 3,667 | 277.828 | 471.77 | 14.93 | EOS |

64K: cache-capacity diagnostics. 96K: closing an engine request during streaming.
128K: stop-string behavior, with an explicit offline/no-tools instruction.
These are source-grounded software-engineering prompts, not repeated filler.
The six responses contain analysis and proposed changes; their correctness was
not scored, and no model-generated commands or code were executed.

## Measurement contract

- All five rendered inputs (three measurements and two warmups) tokenized
  **identically, token for token**, in ORT and llama.cpp. The native llama
  `/completion` endpoint received those exact numeric IDs, bypassing its chat template.
- Batch one, greedy target sampling, thinking disabled in the shared rendered
  prompt, seven draft proposals plus one anchor. No custom stop strings.
- Both models had a 262,144-token total-context ceiling. Generation safety budgets
  were `262144 - input_tokens - 16`; there was no 128/512-token cap or minimum length.
  Every measured request stopped at EOS, not the safety budget or context limit.
- Each framework received the frozen 4K and 16K warmups with 16 output tokens.
  Loading and warmups are excluded from the table; remaining first-use effects are not removed.
  ORT was run first, then llama.cpp; no repetitions or randomized ordering were used.
- TTFT is submission to first visible token event. Prompt tok/s is input tokens
  divided by TTFT, including first-token work; it is not pure prefill-kernel throughput.
- Decode tok/s is `(visible_output_tokens - 1) / (last_visible_time - first_visible_time)`.
  ORT uses in-process engine events; llama uses localhost SSE arrival times.
  Burst delivery from speculation is retained. Terminal EOS tokens are excluded.
- llama's native counters confirm every prompt token was processed and **zero
  prompt tokens were cached**. `cache_prompt=false`, `--cache-ram 0`, and context
  shifting disabled. llama's observed terminal token was 248046 in all three cases.
- The llama client reads SSE with `chunk_size=1` to avoid buffering token events.
  Parsing the large final JSON response (which repeats the prompt) adds post-EOS
  client overhead; it is outside the first-to-last-visible-token interval. Do not
  interpret the saved `request_seconds` as a tuned HTTP-serving latency result.

### Native llama timing cross-check

These native decode rates include the EOS token in their numerator, hence their
small difference from the comparable visible-token rates above.

| Input tokens | Native prompt tok/s | Native decode tok/s | Native generated tokens, including EOS | Cached input tokens |
| ---: | ---: | ---: | ---: | ---: |
| 65,536 | 568.12 | 23.45 | 3,295 | 0 |
| 98,304 | 513.75 | 17.10 | 2,229 | 0 |
| 131,072 | 471.82 | 14.93 | 3,668 | 0 |

## Actual configurations and limits on attribution

| Setting | ORT GenAI | llama.cpp |
| --- | --- | --- |
| Runtime | GenAI `4809ff9dd2292c64ffe9388f03a14da9b865b550`; ORT `9f913ae524b50217e9f7c09ce35b363832a44928` | Official `b10902`, `df03399b885831b2a1603b3abb0d8c156808e363` |
| Target | Local Qwen3.8-27B INT4 block-32, shifted-tap DFlash2 package | Existing Ollama Qwen3.8 27B 0814 Q4_K_M GGUF, reused directly by llama.cpp |
| Drafter | Local five-layer INT4 DFlash2, original greedy selector | Supplied z-lab five-layer DFlash2 Q4_K_M GGUF, selector top-k 16 |
| Target KV | INT8 | Q8_0 K and V |
| Drafter KV | BF16, 2,048-token sliding window | BF16 K and V, 2,048-token sliding window |
| GPU placement | Qualified CUDA EP, unchanged eager stack | Target and drafter all layers requested on GPU; CUDA backend loaded and GPU execution observed |
| Execution | No experimental target/drafter replay or Viterbi overrides | Default runtime execution; no graph-disabling override, no separate graph-capture proof |
| Capacity | 1,032 target-block-equivalent budget, 256 tokens/block | 262,144 context, one slot; fit adjustment disabled |
| Prefill scheduling | Qualified engine defaults | Default logical batch 2,048, physical microbatch 512 |

The ORT block budget resolves to 1,025 target blocks after the fixed ten-block
drafter reservation. It is not 1,032 target blocks plus a free drafter cache.

The llama target has 65 stored blocks including one MTP block; llama loads its
64 main layers and ignores the unused MTP tensors. Speculation explicitly uses
`draft-dflash`, **not MTP**. The drafter GGUF contains `dflash.selector_top_k=16`,
`dflash.block_size=8`, convolution metadata, and five extraction layers; upstream
selects its actual DFlash2 convolution/lattice path from that metadata.

INT4 block-32 and Q4_K_M are **different numerical weight quantizations**, not
equivalent containers. INT8 and Q8_0 caches also differ. The existing Ollama
target is not the separately published ggml-org target GGUF, and byte-identical
source checkpoints across the two target exports have not been established.
Thus these results describe the specified installed stacks, not an isolated
framework-only causal speedup. Different response lengths, acceptance, target
and drafter quantization, and execution strategies all affect the outcome.
No target-only versus speculative output-equivalence study was performed here.

## Speculation counters with a common denominator

Use **accepted / proposed** for both stacks:

| Input tokens | ORT accepted / proposed | ORT fraction | ORT rounds | llama accepted / proposed | llama fraction | llama rounds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 2,189 / 5,530 | 39.58% | 790 | 2,527 / 5,376 | 47.01% | 768 |
| 98,304 | 1,847 / 4,207 | 43.90% | 601 | 1,592 / 4,459 | 35.70% | 637 |
| 131,072 | 4,419 / 10,808 | 40.89% | 1,544 | 2,608 / 7,420 | 35.15% | 1,060 |

Earlier ORT reports used `accepted / evaluated`, which counts only candidates
actually visited during verification and is higher. That number must not be
compared directly with llama's `accepted / proposed`. Raw counters are retained.
The llama rounds are deltas of `llamacpp:spec_decode_num_drafts_total`.

## Provenance and reproduction

Official Windows ARM64 CUDA 13.4 binaries:
[b10902 release](https://github.com/ggml-org/llama.cpp/releases/tag/b10902).
Both downloaded archives matched the release SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| `llama-b10902-bin-win-cuda-13.4-arm64.zip` | `84a06370bb18324d7931754cc8cf86a21617a2b87991509fc92e1cdb25aa33c0` |
| `cudart-llama-bin-win-cuda-13.4-arm64.zip` | `642dcde8805b3e3165ca710a5443b3b4044b27d96bd3ee3132473988c9bcb774` |
| Existing target, 16,810,714,464 bytes | `f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d` |
| DFlash2 Q4_K_M, 1,143,006,816 bytes | `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` |

The drafter is pinned to
[`z-lab/Qwen3.8-27B-DFlash2-GGUF`, revision `2d9571f8ce46e151f61c6499c99dee6079e1d610`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/tree/2d9571f8ce46e151f61c6499c99dee6079e1d610).
Both model files' full SHA-256 values were recomputed locally. The downloaded
target/drafter models declare Apache-2.0; llama.cpp is MIT.

From the local `engine-sample` workspace, the adapted harness files are:
`tools\prepare_framework_prompts.py`, `tools\benchmark_framework_ort.py`,
`tools\benchmark_framework_llama.py`, and `tools\start_framework_llama.ps1`.
Use new output directories/log paths to preserve completed runs. Run frameworks
serially and stop the ORT process before starting llama:

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
$env:PATH = "$(Resolve-Path ..\cuda);$env:PATH"
.\.venv-latest-main-benchmark\Scripts\python.exe tools\benchmark_framework_ort.py --output-dir results\framework-repeat\ort
.\tools\start_framework_llama.ps1 -LogPath "$(Get-Location)\results\framework-repeat\llama-server.log"
# With the server running, from another terminal:
.\.venv\Scripts\python.exe tools\benchmark_framework_llama.py --output-dir results\framework-repeat\llama
```

Raw local evidence is under `results\framework-eos-20260910`: frozen prompts,
both `results.json` files, responses, llama SSE events and native metrics,
server/client logs, `provenance.json`, and `comparison-summary.json`.
The full local artifact directory and model responses are not published.
