# Foundry Local source-grounded SWE benchmark

This directory contains the exact source-grounded prompts and streaming
benchmark used to measure the Qwen 3.8 27B INT4 / INT8-KV / INT4 DFlash2 model
through Foundry Local.

The prompt set contains unique ONNX Runtime GenAI source excerpts and a
cache-capacity diagnostics issue. No source section or synthetic filler is
repeated. Foundry Local applies its chat template, so the API-reported input
count is approximately 40 tokens larger than the precomputed prompt count in
`prompts/manifest.json`.

Start the Foundry Local endpoint and run:

```powershell
python .\benchmark_foundry_local_swe.py `
  --manifest .\prompts\manifest.json `
  --context 4096 --context 16384 --context 32768 `
  --context 65536 --context 131072 --context 260000 `
  --repeat 3 `
  --json-out .\foundry-local-swe-results.json
```

The runner requires `requests`. It measures:

- TTFT from HTTP submission to the first non-empty streamed content or
  reasoning delta;
- prompt throughput as API-reported prompt tokens divided by TTFT;
- decode throughput as `(completion tokens - 1) / (last token - first token)`;
- end-to-end output throughput over the complete HTTP request.

The default endpoint is `http://127.0.0.1:5272/v1/chat/completions`, and the
default payload requests 128 deterministic streamed completion tokens with a
final usage event.
