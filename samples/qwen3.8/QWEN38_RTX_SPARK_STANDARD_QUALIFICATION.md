# Qwen 3.8 27B ORT GenAI Engine Qualification - RTX Spark

## Result

Batch-size-1 measurements completed for the rows marked `passed` below.

## Environment

| Component | Value |
|---|---|
| Machine | ARM64 |
| OS | Windows-11-10.0.28000-SP0 |
| GPU | NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU), 616.02, 32704 MiB<br>NVIDIA NPU, 616.02, [N/A] |
| Python | 3.13.15 |
| ONNX Runtime | 1.30.0.dev20260901002 |
| ONNX Runtime GenAI | 0.16.0-dev1001393453 |
| CUDA plugin | C:\Users\Baiju\workspace\engine-sample\.venv\Lib\site-packages\onnxruntime_ep_cuda\onnxruntime_providers_cuda.dll |
| CUDA binaries | C:\Users\Baiju\workspace\cuda |

## Qualification matrix

| Model | Mode | Status | Detail |
|---|---|---|---|
| INT4 weights / FP16 KV | non-speculative | passed |  |
| INT4 weights / INT8 KV | non-speculative | passed |  |

## Performance results

### Median of three repeats

| Model | Context tokens | TTFT (ms) | Prompt TPS | Decode TPS |
|---|---|---|---|---|
| INT4 weights / FP16 KV | 4096 | 3681.82 | 1112.49 | 15.08 |
| INT4 weights / FP16 KV | 8192 | 7814.71 | 1048.28 | 14.80 |
| INT4 weights / FP16 KV | 16384 | 16226.83 | 1009.69 | 14.21 |
| INT4 weights / FP16 KV | 32768 | 34617.36 | 946.58 | 13.27 |
| INT4 weights / FP16 KV | 65536 | 76758.61 | 853.79 | 11.69 |
| INT4 weights / FP16 KV | 131072 | 195875.29 | 669.16 | 9.57 |
| INT4 weights / FP16 KV | 196608 | 355590.70 | 552.91 | 7.11 |
| INT4 weights / FP16 KV | 262016 | 564727.89 | 463.97 | 6.39 |
| INT4 weights / INT8 KV | 4096 | 3744.54 | 1093.86 | 12.47 |
| INT4 weights / INT8 KV | 8192 | 7201.66 | 1137.52 | 12.37 |
| INT4 weights / INT8 KV | 16384 | 14963.65 | 1094.92 | 12.30 |
| INT4 weights / INT8 KV | 32768 | 31978.34 | 1024.69 | 11.69 |
| INT4 weights / INT8 KV | 65536 | 71227.12 | 920.10 | 11.30 |
| INT4 weights / INT8 KV | 131072 | 173397.02 | 755.91 | 10.53 |
| INT4 weights / INT8 KV | 196608 | 327292.77 | 600.71 | 11.13 |
| INT4 weights / INT8 KV | 262016 | 516242.54 | 507.54 | 10.14 |

### Individual repeats

| Model | Context tokens | Repeat | TTFT (ms) | Prompt TPS | Decode TPS | Output tokens |
|---|---|---|---|---|---|---|
| INT4 weights / FP16 KV | 4096 | 1 | 3481.32 | 1176.56 | 15.25 | 128 |
| INT4 weights / FP16 KV | 4096 | 2 | 3681.82 | 1112.49 | 15.06 | 128 |
| INT4 weights / FP16 KV | 4096 | 3 | 4016.98 | 1019.67 | 15.08 | 128 |
| INT4 weights / FP16 KV | 8192 | 1 | 6888.00 | 1189.32 | 14.82 | 128 |
| INT4 weights / FP16 KV | 8192 | 2 | 7958.84 | 1029.30 | 14.80 | 128 |
| INT4 weights / FP16 KV | 8192 | 3 | 7814.71 | 1048.28 | 13.80 | 128 |
| INT4 weights / FP16 KV | 16384 | 1 | 15157.43 | 1080.92 | 14.21 | 128 |
| INT4 weights / FP16 KV | 16384 | 2 | 16226.83 | 1009.69 | 14.20 | 128 |
| INT4 weights / FP16 KV | 16384 | 3 | 16267.37 | 1007.17 | 14.23 | 128 |
| INT4 weights / FP16 KV | 32768 | 1 | 33693.10 | 972.54 | 13.27 | 128 |
| INT4 weights / FP16 KV | 32768 | 2 | 34617.36 | 946.58 | 13.27 | 128 |
| INT4 weights / FP16 KV | 32768 | 3 | 34686.26 | 944.70 | 13.26 | 128 |
| INT4 weights / FP16 KV | 65536 | 1 | 76758.61 | 853.79 | 11.74 | 128 |
| INT4 weights / FP16 KV | 65536 | 2 | 79512.43 | 824.22 | 11.69 | 128 |
| INT4 weights / FP16 KV | 65536 | 3 | 75612.48 | 866.74 | 10.28 | 128 |
| INT4 weights / FP16 KV | 131072 | 1 | 199260.64 | 657.79 | 9.66 | 128 |
| INT4 weights / FP16 KV | 131072 | 2 | 195875.29 | 669.16 | 9.57 | 128 |
| INT4 weights / FP16 KV | 131072 | 3 | 193309.98 | 678.04 | 8.85 | 128 |
| INT4 weights / FP16 KV | 196608 | 1 | 359078.30 | 547.54 | 7.06 | 128 |
| INT4 weights / FP16 KV | 196608 | 2 | 355590.70 | 552.91 | 7.11 | 128 |
| INT4 weights / FP16 KV | 196608 | 3 | 354845.26 | 554.07 | 7.13 | 128 |
| INT4 weights / FP16 KV | 262016 | 1 | 564727.89 | 463.97 | 6.39 | 128 |
| INT4 weights / FP16 KV | 262016 | 2 | 561032.17 | 467.02 | 6.30 | 128 |
| INT4 weights / FP16 KV | 262016 | 3 | 571484.43 | 458.48 | 7.03 | 128 |
| INT4 weights / INT8 KV | 4096 | 1 | 3550.25 | 1153.72 | 12.75 | 128 |
| INT4 weights / INT8 KV | 4096 | 2 | 3744.54 | 1093.86 | 12.37 | 128 |
| INT4 weights / INT8 KV | 4096 | 3 | 3746.17 | 1093.38 | 12.47 | 128 |
| INT4 weights / INT8 KV | 8192 | 1 | 6627.61 | 1236.04 | 12.17 | 128 |
| INT4 weights / INT8 KV | 8192 | 2 | 7201.66 | 1137.52 | 12.37 | 128 |
| INT4 weights / INT8 KV | 8192 | 3 | 7298.71 | 1122.39 | 12.38 | 128 |
| INT4 weights / INT8 KV | 16384 | 1 | 14046.11 | 1166.44 | 12.30 | 128 |
| INT4 weights / INT8 KV | 16384 | 2 | 15008.85 | 1091.62 | 12.36 | 128 |
| INT4 weights / INT8 KV | 16384 | 3 | 14963.65 | 1094.92 | 12.30 | 128 |
| INT4 weights / INT8 KV | 32768 | 1 | 31010.30 | 1056.68 | 11.69 | 128 |
| INT4 weights / INT8 KV | 32768 | 2 | 31978.34 | 1024.69 | 11.69 | 128 |
| INT4 weights / INT8 KV | 32768 | 3 | 32003.01 | 1023.90 | 12.09 | 128 |
| INT4 weights / INT8 KV | 65536 | 1 | 70226.82 | 933.20 | 11.52 | 128 |
| INT4 weights / INT8 KV | 65536 | 2 | 71227.12 | 920.10 | 11.30 | 128 |
| INT4 weights / INT8 KV | 65536 | 3 | 71364.39 | 918.33 | 11.28 | 128 |
| INT4 weights / INT8 KV | 131072 | 1 | 172267.73 | 760.86 | 10.53 | 128 |
| INT4 weights / INT8 KV | 131072 | 2 | 173444.88 | 755.70 | 10.53 | 128 |
| INT4 weights / INT8 KV | 131072 | 3 | 173397.02 | 755.91 | 10.70 | 128 |
| INT4 weights / INT8 KV | 196608 | 1 | 306099.91 | 642.30 | 9.64 | 128 |
| INT4 weights / INT8 KV | 196608 | 2 | 327292.77 | 600.71 | 11.13 | 128 |
| INT4 weights / INT8 KV | 196608 | 3 | 338327.96 | 581.12 | 11.14 | 128 |
| INT4 weights / INT8 KV | 262016 | 1 | 518195.09 | 505.63 | 10.14 | 128 |
| INT4 weights / INT8 KV | 262016 | 2 | 493331.78 | 531.12 | 8.75 | 128 |
| INT4 weights / INT8 KV | 262016 | 3 | 516242.54 | 507.54 | 10.14 | 128 |

## Methodology

- Batch size: 1
- Context lengths: 4096, 8192, 16384, 32768, 65536, 131072, 196608, 262016 tokens
- Requested output: 128 tokens
- Timed repeats: 3
- Prompt: a deterministic coding-shaped token sequence tiled and truncated to the exact requested token count
- TTFT: request submission to the first `TOKEN` engine event
- Prompt TPS: input token count divided by TTFT
- Decode TPS: tokens after the first token divided by time from first to last token event
- Model loading, tokenizer construction, and warmup are excluded

Raw machine-readable results are in `C:\Users\Baiju\workspace\engine-sample\results-standard\qualification.json` and `C:\Users\Baiju\workspace\engine-sample\results-standard\models\*.json`.
