# RTX Spark: ORT GenAI DFlash2 versus llama.cpp native MTP

## Result

The exact requested **Unsloth Qwen3.8-27B-UD-Q4_K_XL.gguf** ran with genuine
native MTP enabled in llama.cpp. ORT GenAI retained its qualified eager
INT4 / INT8-KV DFlash2 stack. Both frameworks completed the same frozen
64K, 96K, and 128K input sequences to natural EOS at batch one.

Measured September 11, 2026 on Windows ARM64, NVIDIA RTX Spark N1X / GB10,
SM121, 48 SMs, CUDA 13.4. Each row is one run; no repeated-run uncertainty
estimate or model-quality score is claimed.

## Matched-input results

| Input tokens | Stack | Visible output tokens | TTFT (s) | Prompt tok/s | Decode tok/s | Finish |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 65,536 | ORT GenAI + DFlash2 | 3,214 | 82.338 | 795.94 | **37.30** | EOS |
| 65,536 | llama.cpp + MTP | 5,561 | 114.259 | 573.58 | **21.82** | EOS |
| 98,304 | ORT GenAI + DFlash2 | 2,712 | 135.128 | 727.49 | **35.15** | EOS |
| 98,304 | llama.cpp + MTP | 3,552 | 190.571 | 515.84 | **18.24** | EOS |
| 131,072 | ORT GenAI + DFlash2 | 4,921 | 198.199 | 661.31 | **35.71** | EOS |
| 131,072 | llama.cpp + MTP | 4,024 | 273.114 | 479.92 | **13.43** | EOS |

| Input tokens | ORT DFlash2 / llama MTP decode throughput |
| ---: | ---: |
| 65,536 | 1.71x |
| 98,304 | 1.93x |
| 131,072 | 2.66x |

64K is a cache-capacity diagnostics issue; 96K is a close-during-streaming
issue; 128K is a stop-string issue with an explicit offline/no-tools instruction.
These are source-grounded software-engineering prompts, not repeated filler.
Responses were saved, not executed or scored. EOS marks the end of the model
turn, not successful issue resolution.

## What is matched, and what is not

| Setting | ORT GenAI + DFlash2 | llama.cpp + MTP |
| --- | --- | --- |
| Input | Exact frozen token sequence | Identical token sequence, verified through `/tokenize` |
| Target sampling | Greedy | Greedy, temperature 0, top-k 1, no repetition penalties |
| Batch / parallel requests | 1 | 1 |
| Target model | Local Qwen3.8-27B INT4 block-32 shifted-tap DFlash2 package | Requested Unsloth UD-Q4_K_XL target, not the previous Ollama Q4_K_M file |
| Draft method | Separate five-layer DFlash2 block drafter | Embedded, trained, single-layer Qwen MTP head |
| Maximum proposals | 7, plus target anchor | 3, llama's default maximum, generated through the MTP head |
| Target KV | INT8 | Q8_0 K and V |
| Draft KV | BF16, 2,048-token sliding window | BF16 K and V; full-context MTP attention, not a 2K DFlash window |
| Context ceiling | 262,144 | 262,144; context shifting disabled |
| Cache budget | 1,032 target-block-equivalent blocks | One 262,144-token slot, fit adjustment disabled |
| Execution | Unchanged eager runtime, original greedy draft selector | Default runtime execution, all GPU layers requested, no graph-disabling overrides |
| Runtime | GenAI `4809ff9dd2292c64ffe9388f03a14da9b865b550`; ORT `9f913ae524b50217e9f7c09ce35b363832a44928` | Official Windows ARM64 CUDA 13.4 `b10902`, commit `df03399b885831b2a1603b3abb0d8c156808e363` |

This is a **configured-stack comparison**, intentionally DFlash2 versus MTP.
It is not a pure framework-only or speculation-algorithm-only speedup:
weights, quantization, draft architecture/depth, cache formats, and resulting
continuations differ. Byte-identical source target checkpoints have not been
established. UD-Q4_K_XL is a mixed-precision quantization, not uniform four-bit
weights. Its tensor inventory includes Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ3_S,
IQ4_XS, IQ4_NL, and F32; the MTP projections are mainly Q6_K and Q8_0.

Seven proposals are the trained DFlash2 block width. For MTP, three is the
runtime's default maximum, not a claim of tuned optimal depth. No width search
or target-only speedup/output-equivalence study was performed.

## Timing and run policy

- Frozen prompts are the same manifest used in the earlier DFlash2-versus-DFlash2
  comparison. All three measured prompts and both warmups match token for token.
  llama receives numeric IDs through `/completion`, so its automatic chat template
  cannot change the input. Thinking is disabled in the shared rendered prompt.
- The ORT rows are **fresh measurements**, not copied from the previous report.
  Runtime, provider and model graph hashes and settings match that qualified reference.
  ORT ran before llama; no inference frameworks overlapped on the GPU.
- Both stacks received the 4K and 16K warmups with 16 outputs. Model loading and
  warmups are excluded; any remaining first-use work is included in TTFT.
- No 128/512-token answer cap or minimum output length. The only output safety
  budget is `262144 - input_tokens - 16`. All six measured rows stopped at EOS,
  not a budget or context limit; the visible count excludes the EOS token.
- TTFT is request submission to first visible token event. Prompt tok/s is input
  tokens divided by TTFT, including first-token work, not just prefill kernels.
- Decode tok/s is `(visible_output_tokens - 1) / (last_visible_time - first_visible_time)`.
  ORT uses in-process token events; llama uses localhost SSE arrival times.
  Native llama timing spans first token through EOS and its generated-token count
  includes EOS. At 96K, EOS required a separate final step, about 0.193 seconds
  after the last visible token. That step is excluded from the comparable decode
  rate; native decode throughput there is 18.23 versus 18.24 visible tok/s.
  The first-to-EOS event interval agrees with native timing within 0.1 seconds.
- llama processed every input token with **zero prompt-cache hits**:
  `cache_prompt=false`, `--cache-ram 0`, no context shifting. All observed
  llama terminal IDs are recorded in the raw results.
- The client reads SSE with `chunk_size=1`. Parsing llama's final JSON, which
  repeats the full prompt, adds post-EOS overhead. It does not enter TTFT or
  first-to-last-visible-token decode timing; `request_seconds` is not a tuned
  HTTP-serving latency result.

## Genuine MTP and acceptance evidence

The downloaded GGUF contains 65 stored blocks: 64 target layers and one MTP
layer (`qwen35.nextn_predict_layers=1`). Its 15 MTP tensors include the trained
`blk.64.nextn.eh_proj.weight`, input normalizations, attention, and MLP weights.
The server explicitly selects **`draft-mtp`**, creates an MTP context against
that same model, and reports nonzero proposed and accepted MTP tokens for
every measured request. No DFlash GGUF or generic draft model is loaded by llama.

The denominator below is **all proposed tokens** in both stacks. Earlier ORT
`accepted / evaluated` percentages are not directly comparable to these rates.

| Input tokens | ORT accepted / proposed | ORT fraction | ORT rounds | MTP accepted / proposed | MTP fraction | MTP rounds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 2,368 / 5,922 | 39.99% | 846 | 3,975 / 4,761 | 83.49% | 1,587 |
| 98,304 | 1,982 / 5,110 | 38.79% | 730 | 2,516 / 3,117 | 80.72% | 1,039 |
| 131,072 | 3,677 / 8,708 | 42.23% | 1,244 | 2,651 / 4,125 | 64.27% | 1,375 |

Native timings and `llamacpp:spec_decode_num_drafts_total` counter deltas
are retained in `comparison-summary.json`.

## Model provenance and reproduction

Requested repository: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).
Pinned revision: `4ca720788d1e01f1bff70c033e0d0028fd02e502`.
File: `Qwen3.8-27B-UD-Q4_K_XL.gguf`, **17,559,178,144 bytes**.
Full local SHA-256 matched Hugging Face LFS metadata:
`3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e`.
Model license: Apache-2.0. Runtime: MIT. Model geometry, tensor inventory,
MTP tensor shapes/types, and executable/CUDA-backend hashes are saved in
`results\framework-mtp-eos-20260911\model-provenance.json`.

The launcher uses:

```powershell
llama-server.exe -m Qwen3.8-27B-UD-Q4_K_XL.gguf `
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 `
  -ngl 999 -ngld 999 -fa on `
  -ctk q8_0 -ctv q8_0 -ctkd bf16 -ctvd bf16 `
  -c 262144 -np 1 --fit off --no-context-shift `
  --temp 0 -n -1 --metrics --cache-ram 0 --host 127.0.0.1 --port 8092
```

Local harness commands from `engine-sample` (use fresh output paths for repeats):

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
$env:PATH = "$(Resolve-Path ..\cuda);$env:PATH"
.\.venv-latest-main-benchmark\Scripts\python.exe tools\benchmark_framework_ort.py --output-dir results\mtp-repeat\ort
# After ORT exits, run the server; use another terminal for the client.
.\tools\start_framework_llama_mtp.ps1 -LogPath "$(Get-Location)\results\mtp-repeat\llama-server.log"
.\.venv\Scripts\python.exe tools\benchmark_framework_llama.py --url http://127.0.0.1:8092 --expect-spec-type draft-mtp --output-dir results\mtp-repeat\llama
```

Raw evidence: `results\framework-mtp-eos-20260911` contains both result files,
responses, SSE events, metrics snapshots, logs, provenance, and the compact summary.
The shared frozen inputs remain in `results\framework-eos-20260910\prompts`.
