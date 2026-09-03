# Qwen 3.8 INT4 + DFlash2 model tooling

This package reproduces the retained Qwen 3.8 27B model configuration used for
RTX Spark qualification. It combines:

- A base Qwen 3.8 27B INT4 target with INT8 KV cache.
- The DFlash2 graph and state schema from a Qwen 3.8 DFlash2 model package.
- Target-side DFlash2 state capture.
- An INT4 DFlash2 drafter with the target's Q4 vocabulary head.
- Gate/up projection fusion in every target MLP layer.

The default command produces the production candidate. Rejected experimental
fusions, alternate draft widths, alternate lattice sizes, and drafter
prepacking are intentionally excluded.

## Inputs

`--target` must point to the base INT4/INT8-KV model directory and contain:

- `model.onnx`
- `model.onnx.data`
- `genai_config.json`
- Tokenizer and chat-template files

`--dflash2` must point to the DFlash2 model directory and contain:

- `text.onnx`
- `text.onnx.data`
- `dflash2.onnx`
- `dflash2.onnx.data`
- `genai_config.json`

Only the DFlash2 state interface and drafter are taken from the second package.
Its target weights are not copied into the output.

## Dependencies

Use a Python environment with:

- Python 3.10 or newer.
- `onnx`
- `numpy`
- An ONNX Runtime build that exposes
  `onnxruntime.quantization.matmul_nbits_quantizer`.

Run from the ONNX Runtime GenAI repository root so `tools.python` is importable.

## Build the retained model

On Windows:

```powershell
$env:PYTHONPATH = "C:\path\to\onnxruntime-python-package"
python -m tools.python.qwen38_int4_dflash2.build_model `
  --target C:\models\qwen3.8-27b-int4-int8-kv `
  --dflash2 C:\models\qwen3.8-27b-nvfp4-int8-kv-dflash2 `
  --output C:\models\qwen3.8-27b-int4-int8-kv-dflash2-fused
```

The default uses hardlinks for the target's approximately 17 GiB external data.
Use this when the input and output are on the same volume. For a self-contained
copy:

```powershell
python -m tools.python.qwen38_int4_dflash2.build_model `
  --target C:\models\qwen3.8-27b-int4-int8-kv `
  --dflash2 C:\models\qwen3.8-27b-nvfp4-int8-kv-dflash2 `
  --output C:\models\qwen3.8-27b-int4-int8-kv-dflash2-fused `
  --external-data-mode copy
```

The output directory must be empty. The builder validates the final graphs and
writes `MODEL_BUILD_MANIFEST.json` with source and artifact hashes, including
tokenizer and chat-template files, plus the Python, ONNX, NumPy, ONNX Runtime,
and quantizer-module identity. Use `--skip-checksums` only for local iteration.

ONNX deliberately rejects external-data files with multiple hardlinks in its
path-based checker. For hardlink-mode builds, the validator checks every
external tensor's file, offset, and length directly and runs graph-invariant
checks. Copy-mode builds additionally run the full ONNX path-based checker.

## Output

The supported output is:

- `model.onnx`
- `model.onnx.data`
- `dflash2-int4.onnx`
- `dflash2-int4.onnx.data`
- `genai_config.json`
- Tokenizer, chat template, and model metadata files
- `MODEL_BUILD_MANIFEST.json`

`genai_config.json` points directly to `dflash2-int4.onnx`; no benchmark-time
drafter filename override is required.

Validate an existing output with:

```powershell
python -m tools.python.qwen38_int4_dflash2.validate_model `
  --model C:\models\qwen3.8-27b-int4-int8-kv-dflash2-fused
```

## Target graph changes

### 1. Produce logits for every packed verification row

The base target feeds only the last hidden-state row into the Q4 vocabulary
projection. DFlash2 verification needs one logits row for every packed target
row.

The builder changes the input of `/lm_head/MatMul_Q4` from:

```text
/lm_head/last_hidden_state/Gather/output_0
```

to:

```text
/model/layers.64/final_norm_layernorm/output_0
```

The public logits shape is updated from a last-token batch dimension to:

```text
[num_tokens, vocab_size]
```

Metadata-only shape changes are insufficient; bypassing the gather is required.

### 2. Add DFlash2 state-capture inputs

The target receives:

- `state_update_capture_count`
- `state_update_active`

These control compact recurrent-state checkpoint capture during target
verification.

### 3. Add 96 state-update outputs

Qwen 3.8 uses 48 linear-attention layers. Each contributes:

- One `VarlenCausalConvWithState` checkpoint.
- One `GatedDeltaNet` recurrent capsule.

The builder adds 96 outputs:

```text
state_update.<layer>.conv_value
state_update.<layer>.recurrent_capsule
```

Each operator receives `state_update_capacity=7`, matching the seven-token
draft width.

### 4. Add auxiliary hidden-state taps

DFlash2 consumes hidden states from target layers:

```text
5, 19, 33, 47, 61
```

The corresponding layer-normalized tensors are concatenated into:

```text
aux_hidden_states
```

and exposed as a target output.

### 5. Fuse all 64 MLP gate/up projection pairs

Before:

```text
hidden -> gate_proj MatMulNBits -> SiLU --+
hidden -> up_proj   MatMulNBits ----------+-> Mul
```

After:

```text
hidden -> fused gate_up MatMulNBits -> Split -> gate -> SiLU --+
                                             -> up ------------+-> Mul
```

The packed INT4 weights and scales are already contiguous in the base target's
external data. Fusion therefore creates larger initializer views over the same
bytes; it does not requantize or rewrite target weights.

This reduces target `MatMulNBits` nodes from 497 to 433 and produced an
approximately 4.5% incremental end-to-end throughput improvement.

Only gate/up fusion is retained. Attention K/V and recurrent B/A fusion were
tested, did not reduce target-forward cost, changed token trajectories, and are
not part of this tool.

### 6. Update fixed-state configuration

The decoder configuration gains:

- State-capture input mappings.
- State-update output name patterns.
- `state_update_capacity: 7`.
- Fixed convolution and recurrent state-update metadata.
- `key_head_count: 16` for recurrent groups.

## Drafter graph changes

### 1. Share the target embedding and vocabulary head

The drafter uses these target initializers:

- `model.embed_tokens.weight`
- `lm_head.MatMul.weight_Q4`
- `lm_head.MatMul.weight_scales`

The original FP8 drafter vocabulary projection is replaced by a block-32
`MatMulNBits` using the target Q4 head. Shared initializer metadata points to
`model.onnx.data`.

### 2. Quantize constant-weight drafter projections

The source DFlash2 graph has 58 ordinary MatMuls:

- 57 have constant BF16 weights and are quantized to symmetric INT4 with block
  size 32.
- `/dflash2/selector/pair` has a dynamic right-hand input and remains a regular
  MatMul.

Together with the Q4 vocabulary head, the final drafter has 58
`MatMulNBits` nodes and one dynamic MatMul.

Block size 32 is retained because block sizes 64 and 128 reduced speculative
acceptance enough to lose end-to-end throughput.

### 3. Compact external data

ORT quantization may leave original BF16 bytes in its external-data output and
duplicate the target embedding/head. The builder:

1. Restores the three shared initializers to `model.onnx.data`.
2. Copies only referenced quantized drafter slices into
   `dflash2-int4.onnx.data`.
3. Rewrites offsets with 64-byte alignment.

This makes the output self-consistent without retaining unused source-weight
ranges. In the qualified reproduction, drafter external data decreased from
4.59 GiB in the earlier experimental artifact to 1.33 GiB with identical graph
operator counts and token-exact output on the extended instruction subset.

### 4. Retain the qualified DFlash2 configuration

- Draft width: 7.
- State-update capacity: 7.
- Sliding window: 2,048.
- Selector top-k: 16.

## Optional target prepacking

`experimental_prepack_target.py` reproduces the target-side offline fpA/intB
layout used in qualification experiments:

```powershell
python -m tools.python.qwen38_int4_dflash2.experimental_prepack_target `
  --model C:\models\...\model.onnx `
  --output-model C:\models\...\model-prepacked.onnx `
  --output-data C:\models\...\model-prepacked.onnx.data
```

This is deliberately not part of the default build:

- The layout is an ORT/CUTLASS implementation detail.
- It must be compatibility-keyed by architecture and runtime version.
- It does not eliminate tactic profiling.
- Only target output parity has been established.
- Drafter-head offline prepacking produced zero acceptance and must not be used.

The first production release should use runtime target packing unless ORT
formalizes a versioned serialized-prepack contract.

## Excluded experiments

The tool intentionally does not expose:

- Attention K/V or recurrent B/A fusion.
- DFlash2 block sizes 64 or 128.
- Selector top-k 4 or 8.
- Draft widths below 7.
- Sliding windows 512 or 1,024.
- Drafter offline prepacking.
- Native SM121 NVFP4.
- Post-DFlash2 n-gram override.

These were neutral, slower, acceptance-regressing, or not correct enough for a
production model pipeline.

## Qualification evidence

The retained model achieved:

- 41.81 tok/s mean on the general technical-prose workload.
- 81.94 tok/s at 4K and 64.53 tok/s at 262K on high-acceptance coding output.
- 101.4 aggregate tok/s at batch 4.
- 155.6 aggregate tok/s at batch 8.

Quality checks found:

- 10/10 initial target outputs exact against the base INT4/INT8-KV target.
- 10/10 initial offline-prepacked target outputs exact.
- DFlash2 without CUDA Graphs exact on the initial, extended, and 64K/262K
  comparison suites.

CUDA Graphs remain off by default for strict deterministic qualification until
the observed cross-process token variation is root-caused.
