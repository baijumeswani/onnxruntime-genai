#!/usr/bin/env python3
"""Build the retained Qwen 3.8 INT4 + INT4 DFlash2 model configuration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

import numpy
import onnx
import onnxruntime
from onnx import helper

from .validate_model import validate_model_directory

try:
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )
except ImportError:
    DefaultWeightOnlyQuantConfig = None
    MatMulNBitsQuantizer = None

LAYER_COUNT = 64
AUX_LAYERS = (5, 19, 33, 47, 61)
STATE_UPDATE_CAPACITY = 7
DRAFT_WIDTH = 7
DRAFTER_SLIDING_WINDOW = 2048
DRAFTER_SELECTOR_TOP_K = 16
SHARED_INITIALIZER_NAMES = (
    "model.embed_tokens.weight",
    "lm_head.MatMul.weight_Q4",
    "lm_head.MatMul.weight_scales",
)
SUPPORT_FILES = (
    "chat_template.jinja",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def external_metadata(initializer) -> dict[str, str]:
    return {item.key: item.value for item in initializer.external_data}


def node_attributes(node) -> dict[str, object]:
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def set_external_metadata(
    initializer,
    location: str,
    offset: int,
    length: int,
) -> None:
    del initializer.external_data[:]
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        item = initializer.external_data.add()
        item.key = key
        item.value = value
    initializer.data_location = onnx.TensorProto.EXTERNAL
    initializer.ClearField("raw_data")


def external_initializer_config(initializer, data_file: str) -> dict:
    metadata = external_metadata(initializer)
    return {
        "name": initializer.name,
        "data_file": data_file,
        "offset": metadata["offset"],
        "length": metadata["length"],
        "data_type": initializer.data_type,
        "shape": list(initializer.dims),
    }


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    if destination.exists():
        destination.unlink()
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError as error:
            raise RuntimeError(
                f"Could not hardlink {source} to {destination}. "
                "Use --external-data-mode copy when the paths are on different volumes."
            ) from error
    shutil.copy2(source, destination)


def require_external_contiguity(first, second, description: str) -> None:
    first_meta = external_metadata(first)
    second_meta = external_metadata(second)
    if first_meta.get("location") != second_meta.get("location") or int(
        first_meta.get("offset", "-1")
    ) + int(first_meta.get("length", "-1")) != int(second_meta.get("offset", "-2")):
        raise RuntimeError(
            f"{description} tensors are not contiguous in external data."
        )


def fuse_gate_up_projections(model: onnx.ModelProto) -> None:
    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    nodes = {node.name: node for node in graph.node}
    replacements: dict[str, list] = {}
    removed_nodes: set[str] = set()
    removed_initializers: set[str] = set()

    for layer in range(LAYER_COUNT):
        prefix = f"/model/layers.{layer}/mlp"
        gate = nodes.get(f"{prefix}/gate_proj/MatMul_Q4")
        up = nodes.get(f"{prefix}/up_proj/MatMul_Q4")
        if gate is None or up is None:
            raise RuntimeError(
                f"Layer {layer} does not contain the expected gate/up projections."
            )
        if gate.input[0] != up.input[0]:
            raise RuntimeError(f"Layer {layer} gate/up activations differ.")
        if len(gate.input) != 3 or len(up.input) != 3:
            raise RuntimeError(
                f"Layer {layer} gate/up fusion supports only symmetric "
                "MatMulNBits nodes without zero points."
            )

        gate_weight = initializers[gate.input[1]]
        up_weight = initializers[up.input[1]]
        gate_scale = initializers[gate.input[2]]
        up_scale = initializers[up.input[2]]
        require_external_contiguity(
            gate_weight, up_weight, f"Layer {layer} gate/up weight"
        )
        require_external_contiguity(
            gate_scale, up_scale, f"Layer {layer} gate/up scale"
        )

        fused_weight = copy.deepcopy(gate_weight)
        fused_weight.name = f"model.layers.{layer}.mlp.gate_up_proj.MatMul.weight_Q4"
        fused_weight.dims[0] = gate_weight.dims[0] + up_weight.dims[0]
        fused_weight_meta = external_metadata(fused_weight)
        set_external_metadata(
            fused_weight,
            fused_weight_meta["location"],
            int(fused_weight_meta["offset"]),
            int(external_metadata(gate_weight)["length"])
            + int(external_metadata(up_weight)["length"]),
        )

        fused_scale = copy.deepcopy(gate_scale)
        fused_scale.name = f"model.layers.{layer}.mlp.gate_up_proj.MatMul.weight_scales"
        fused_scale.dims[0] = gate_scale.dims[0] + up_scale.dims[0]
        fused_scale_meta = external_metadata(fused_scale)
        set_external_metadata(
            fused_scale,
            fused_scale_meta["location"],
            int(fused_scale_meta["offset"]),
            int(external_metadata(gate_scale)["length"])
            + int(external_metadata(up_scale)["length"]),
        )

        fused = copy.deepcopy(gate)
        fused.name = f"{prefix}/gate_up_proj/MatMul_Q4"
        fused.input[1] = fused_weight.name
        fused.input[2] = fused_scale.name
        fused.output[0] = f"{prefix}/gate_up_proj/MatMul/output_0"
        n_attribute = next(
            (attribute for attribute in fused.attribute if attribute.name == "N"),
            None,
        )
        if n_attribute is None:
            raise RuntimeError(f"Layer {layer} gate projection has no N attribute.")
        n_attribute.i = gate_weight.dims[0] + up_weight.dims[0]
        split = helper.make_node(
            "Split",
            [fused.output[0]],
            [gate.output[0], up.output[0]],
            name=f"{prefix}/gate_up_proj/Split",
            axis=-1,
            num_outputs=2,
        )

        replacements[gate.name] = [fused, split]
        removed_nodes.add(up.name)
        removed_initializers.update(
            (gate_weight.name, up_weight.name, gate_scale.name, up_scale.name)
        )
        graph.initializer.extend((fused_weight, fused_scale))

    rewritten_nodes = []
    for node in graph.node:
        if node.name in replacements:
            rewritten_nodes.extend(replacements[node.name])
        elif node.name not in removed_nodes:
            rewritten_nodes.append(node)
    del graph.node[:]
    graph.node.extend(rewritten_nodes)

    retained_initializers = [
        initializer
        for initializer in graph.initializer
        if initializer.name not in removed_initializers
    ]
    del graph.initializer[:]
    graph.initializer.extend(retained_initializers)


def transform_target(
    target_model_path: Path,
    reference_model_path: Path,
    aux_layers: tuple[int, ...],
) -> tuple[onnx.ModelProto, dict[str, object], dict[str, int]]:
    model = onnx.load(str(target_model_path), load_external_data=False)
    reference = onnx.load(str(reference_model_path), load_external_data=False)
    reference_inputs = {value.name: value for value in reference.graph.input}
    reference_outputs = {value.name: value for value in reference.graph.output}

    fuse_gate_up_projections(model)

    logits = next(
        (value for value in model.graph.output if value.name == "logits"), None
    )
    if logits is None:
        raise RuntimeError("Target graph has no logits output.")
    logits.type.tensor_type.shape.dim[0].dim_param = "num_tokens"
    logits_node = next(
        (node for node in model.graph.node if "logits" in node.output),
        None,
    )
    if logits_node is None or logits_node.op_type != "MatMulNBits":
        raise RuntimeError("Target graph has no expected MatMulNBits logits node.")
    logits_node.input[0] = "/model/layers.64/final_norm_layernorm/output_0"

    for name in ("state_update_capture_count", "state_update_active"):
        if name not in reference_inputs:
            raise RuntimeError(f"DFlash2 reference target has no {name} input.")
        model.graph.input.append(copy.deepcopy(reference_inputs[name]))

    state_output_names = []
    for node in model.graph.node:
        if "/linear_attn/" not in node.name:
            continue
        layer_marker = "/model/layers."
        if not node.name.startswith(layer_marker):
            continue
        layer = int(node.name[len(layer_marker) :].split("/", 1)[0])
        if node.op_type == "VarlenCausalConvWithState":
            node.input.append("state_update_capture_count")
            output_name = f"state_update.{layer}.conv_value"
            node.output.append(output_name)
            node.attribute.append(
                helper.make_attribute("state_update_capacity", STATE_UPDATE_CAPACITY)
            )
            state_output_names.append(output_name)
        elif node.op_type == "GatedDeltaNet":
            node.input.extend(("state_update_capture_count", "state_update_active"))
            output_name = f"state_update.{layer}.recurrent_capsule"
            node.output.append(output_name)
            node.attribute.append(
                helper.make_attribute("state_update_capacity", STATE_UPDATE_CAPACITY)
            )
            state_output_names.append(output_name)

    if len(state_output_names) != 96:
        raise RuntimeError(
            f"Expected 96 state-update outputs, found {len(state_output_names)}."
        )
    for name in state_output_names:
        if name not in reference_outputs:
            raise RuntimeError(f"DFlash2 reference target has no {name} output.")
        model.graph.output.append(copy.deepcopy(reference_outputs[name]))

    aux_inputs = [
        f"/model/layers.{layer}/input_layernorm/output_3" for layer in aux_layers
    ]
    model.graph.node.append(
        helper.make_node(
            "Concat",
            aux_inputs,
            ["aux_hidden_states"],
            name="/model/aux_hidden_states/Concat",
            axis=-1,
        )
    )
    if "aux_hidden_states" not in reference_outputs:
        raise RuntimeError("DFlash2 reference target has no aux_hidden_states output.")
    model.graph.output.append(copy.deepcopy(reference_outputs["aux_hidden_states"]))

    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    embedding = initializers.get("model.embed_tokens.weight")
    if embedding is None:
        raise RuntimeError("Target graph has no model.embed_tokens.weight initializer.")
    head_weight = initializers[logits_node.input[1]]
    head_scale = initializers[logits_node.input[2]]
    logits_attributes = node_attributes(logits_node)
    head_attributes = {
        key: int(logits_attributes[key]) for key in ("K", "N", "bits", "block_size")
    }
    shared = {
        initializer.name: copy.deepcopy(initializer)
        for initializer in (embedding, head_weight, head_scale)
    }
    return model, shared, head_attributes


def transform_drafter(
    drafter_model_path: Path,
    shared_initializers: dict[str, object],
    head_attributes: dict[str, int],
) -> onnx.ModelProto:
    model = onnx.load(str(drafter_model_path), load_external_data=False)
    head = next(
        (node for node in model.graph.node if node.name == "/lm_head/MatMul"),
        None,
    )
    if head is None:
        raise RuntimeError("DFlash2 graph has no expected /lm_head/MatMul node.")
    if len(head.input) < 3:
        raise RuntimeError("DFlash2 vocabulary head has no weight and scale inputs.")

    old_head_initializers = set(head.input[1:])
    head_input = head.input[0]
    head.op_type = "MatMulNBits"
    head.domain = "com.microsoft"
    del head.input[:]
    head.input.extend(
        (
            head_input,
            "lm_head.MatMul.weight_Q4",
            "lm_head.MatMul.weight_scales",
        )
    )
    del head.attribute[:]
    head.attribute.extend(
        helper.make_attribute(name, head_attributes[name])
        for name in ("K", "N", "bits", "block_size")
    )

    retained = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name not in old_head_initializers
        and initializer.name not in SHARED_INITIALIZER_NAMES
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(
        retained
        + [
            copy.deepcopy(shared_initializers[name])
            for name in SHARED_INITIALIZER_NAMES
        ]
    )
    return model


def materialize_external_data(
    model: onnx.ModelProto,
    source_files: dict[str, Path],
    output_dir: Path,
) -> None:
    streams = {}
    offsets: dict[str, int] = {}
    try:
        for initializer in model.graph.initializer:
            metadata = external_metadata(initializer)
            if not metadata:
                continue
            location = metadata["location"]
            source_path = source_files.get(location)
            if source_path is None:
                raise RuntimeError(
                    f"No source file was provided for external data {location}."
                )
            if location not in streams:
                streams[location] = (
                    source_path.open("rb"),
                    (output_dir / location).open("wb"),
                )
                offsets[location] = 0
            source, destination = streams[location]
            length = int(metadata["length"])
            source.seek(int(metadata.get("offset", "0")))
            output_offset = offsets[location]
            remaining = length
            while remaining:
                chunk = source.read(min(16 * 1024 * 1024, remaining))
                if not chunk:
                    raise EOFError(
                        f"Unexpected end of {source_path} while reading "
                        f"{initializer.name}."
                    )
                destination.write(chunk)
                remaining -= len(chunk)
            set_external_metadata(
                initializer,
                location,
                output_offset,
                length,
            )
            offsets[location] += length
    finally:
        for source, destination in streams.values():
            source.close()
            destination.close()


def quantize_drafter(input_model: Path, output_model: Path) -> None:
    if DefaultWeightOnlyQuantConfig is None or MatMulNBitsQuantizer is None:
        raise RuntimeError(
            "The installed ONNX Runtime package does not expose "
            "onnxruntime.quantization.matmul_nbits_quantizer. Use an ORT build "
            "that includes the Python quantization tools."
        )

    config = DefaultWeightOnlyQuantConfig(
        block_size=32,
        is_symmetric=True,
        op_types_to_quantize=("MatMul",),
        bits=4,
    )
    quantizer = MatMulNBitsQuantizer(model=str(input_model), algo_config=config)
    quantizer.process()
    quantizer.model.save_model_to_file(str(output_model), True)


def compact_drafter_external_data(
    quantized_model_path: Path,
    output_model_path: Path,
    output_data_path: Path,
    shared_initializers: dict[str, object],
) -> None:
    model = onnx.load(str(quantized_model_path), load_external_data=False)
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }

    for name, source_initializer in shared_initializers.items():
        initializer = initializers.get(name)
        if initializer is None:
            raise RuntimeError(f"Quantized drafter lost shared initializer {name}.")
        metadata = external_metadata(source_initializer)
        set_external_metadata(
            initializer,
            "model.onnx.data",
            int(metadata["offset"]),
            int(metadata["length"]),
        )

    with output_data_path.open("wb") as destination:
        for initializer in model.graph.initializer:
            if initializer.name in shared_initializers:
                continue
            metadata = external_metadata(initializer)
            if not metadata:
                continue
            source_path = quantized_model_path.parent / metadata["location"]
            length = int(metadata["length"])
            with source_path.open("rb") as source:
                source.seek(int(metadata.get("offset", "0")))
                padding = (-destination.tell()) % 64
                if padding:
                    destination.write(bytes(padding))
                output_offset = destination.tell()
                remaining = length
                while remaining:
                    chunk = source.read(min(16 * 1024 * 1024, remaining))
                    if not chunk:
                        raise EOFError(
                            f"Unexpected end of {source_path} while reading "
                            f"{initializer.name}."
                        )
                    destination.write(chunk)
                    remaining -= len(chunk)
            set_external_metadata(
                initializer,
                output_data_path.name,
                output_offset,
                length,
            )

    onnx.save_model(model, str(output_model_path))


def update_config(
    target_config: dict,
    dflash2_config: dict,
    shared_initializers: dict[str, object],
) -> dict:
    config = copy.deepcopy(target_config)
    decoder = config["model"]["decoder"]
    decoder["filename"] = "model.onnx"
    decoder["inputs"]["state_update_capture_count"] = "state_update_capture_count"
    decoder["inputs"]["state_update_active"] = "state_update_active"
    decoder["outputs"]["state_update_conv_value_names"] = "state_update.%d.conv_value"
    decoder["outputs"][
        "state_update_recurrent_capsule_names"
    ] = "state_update.%d.recurrent_capsule"
    decoder["outputs"]["aux_hidden_states"] = "aux_hidden_states"
    decoder["state_update_capacity"] = STATE_UPDATE_CAPACITY
    for group in decoder["state_groups"]:
        if group["kind"] == "fixed_conv":
            group["state_update"] = {"capacity": STATE_UPDATE_CAPACITY}
        elif group["kind"] == "fixed_recurrent":
            group["state_update"] = {
                "capacity": STATE_UPDATE_CAPACITY,
                "key_head_count": 16,
            }

    drafter = copy.deepcopy(dflash2_config["model"]["dflash2"])
    drafter["filename"] = "dflash2-int4.onnx"
    drafter["shared_initializers"] = [
        external_initializer_config(shared_initializers[name], "model.onnx.data")
        for name in SHARED_INITIALIZER_NAMES
    ]
    config["model"]["dflash2"] = drafter
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hashes_for_existing_files(
    directory: Path, names: tuple[str, ...]
) -> dict[str, str]:
    return {
        name: sha256(directory / name) for name in names if (directory / name).is_file()
    }


def write_manifest(
    output: Path,
    target: Path,
    dflash2: Path,
    external_data_mode: str,
) -> None:
    artifacts = {}
    for name in (
        "model.onnx",
        "model.onnx.data",
        "dflash2-int4.onnx",
        "dflash2-int4.onnx.data",
        "genai_config.json",
        *SUPPORT_FILES,
    ):
        path = output / name
        if not path.is_file():
            continue
        artifacts[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    quantizer_path = Path(inspect.getfile(MatMulNBitsQuantizer)).resolve()
    manifest = {
        "tool": "tools.python.qwen38_int4_dflash2.build_model",
        "build_environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": sys.platform,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "numpy": numpy.__version__,
            "quantizer_module": str(quantizer_path),
            "quantizer_module_sha256": sha256(quantizer_path),
        },
        "sources": {
            "target": {
                "path": str(target),
                "model.onnx": sha256(target / "model.onnx"),
                "model.onnx.data": sha256(target / "model.onnx.data"),
                "genai_config.json": sha256(target / "genai_config.json"),
                **hashes_for_existing_files(target, SUPPORT_FILES),
            },
            "dflash2": {
                "path": str(dflash2),
                "text.onnx": sha256(dflash2 / "text.onnx"),
                "dflash2.onnx": sha256(dflash2 / "dflash2.onnx"),
                "dflash2.onnx.data": sha256(dflash2 / "dflash2.onnx.data"),
                "genai_config.json": sha256(dflash2 / "genai_config.json"),
            },
        },
        "external_data_mode": external_data_mode,
        "retained_configuration": {
            "target_weights": "INT4 block size 32",
            "kv_cache": "INT8 per channel",
            "gate_up_fusion_layers": LAYER_COUNT,
            "drafter_weights": "symmetric INT4 block size 32",
            "draft_width": DRAFT_WIDTH,
            "sliding_window": DRAFTER_SLIDING_WINDOW,
            "selector_top_k": DRAFTER_SELECTOR_TOP_K,
            "state_update_capacity": STATE_UPDATE_CAPACITY,
        },
        "artifacts": artifacts,
    }
    (output / "MODEL_BUILD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Base Qwen 3.8 INT4/INT8-KV model directory.",
    )
    parser.add_argument(
        "--dflash2",
        type=Path,
        required=True,
        help="Qwen 3.8 DFlash2 model directory containing text.onnx and dflash2.onnx.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--external-data-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to place the target's 17 GiB external data in the output.",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip the final SHA-256 manifest for faster local iteration.",
    )
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    dflash2 = args.dflash2.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    target_config_path = target / "genai_config.json"
    dflash2_config_path = dflash2 / "genai_config.json"
    target_config = json.loads(target_config_path.read_text(encoding="utf-8"))
    dflash2_config = json.loads(dflash2_config_path.read_text(encoding="utf-8"))
    drafter_config = dflash2_config["model"]["dflash2"]
    expected_drafter_config = {
        "num_draft_tokens": DRAFT_WIDTH,
        "sliding_window": DRAFTER_SLIDING_WINDOW,
        "selector_top_k": DRAFTER_SELECTOR_TOP_K,
    }
    for key, expected in expected_drafter_config.items():
        actual = drafter_config.get(key)
        if actual != expected:
            raise RuntimeError(
                f"The source DFlash2 package has {key}={actual}; "
                f"this retained pipeline requires {expected}."
            )
    aux_layers = tuple(drafter_config.get("aux_hidden_state_layers", ()))
    if aux_layers != AUX_LAYERS:
        raise RuntimeError(
            "The source DFlash2 auxiliary layers do not match the retained "
            f"configuration: {aux_layers} != {AUX_LAYERS}."
        )
    target_model_path = target / target_config["model"]["decoder"]["filename"]
    reference_model_path = dflash2 / dflash2_config["model"]["decoder"]["filename"]
    drafter_model_path = dflash2 / dflash2_config["model"]["dflash2"]["filename"]

    for path in (
        target_model_path,
        reference_model_path,
        drafter_model_path,
        target / "model.onnx.data",
        dflash2 / "dflash2.onnx.data",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    for name in SUPPORT_FILES:
        source = target / name
        if source.is_file():
            shutil.copy2(source, output / name)
    link_or_copy(
        target / "model.onnx.data",
        output / "model.onnx.data",
        args.external_data_mode,
    )

    target_model, shared_initializers, head_attributes = transform_target(
        target_model_path, reference_model_path, aux_layers
    )
    onnx.save_model(target_model, str(output / "model.onnx"))

    with tempfile.TemporaryDirectory(prefix=".qwen38-build-", dir=output) as temp:
        work = Path(temp)
        drafter_model = transform_drafter(
            drafter_model_path,
            shared_initializers,
            head_attributes,
        )
        materialize_external_data(
            drafter_model,
            {
                "dflash2.onnx.data": dflash2 / "dflash2.onnx.data",
                "model.onnx.data": target / "model.onnx.data",
            },
            work,
        )
        unquantized_path = work / "dflash2.onnx"
        quantized_path = work / "dflash2-int4.onnx"
        onnx.save_model(drafter_model, str(unquantized_path))
        quantize_drafter(unquantized_path, quantized_path)
        compact_drafter_external_data(
            quantized_path,
            output / "dflash2-int4.onnx",
            output / "dflash2-int4.onnx.data",
            shared_initializers,
        )

    config = update_config(target_config, dflash2_config, shared_initializers)
    (output / "genai_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_model_directory(output)
    if not args.skip_checksums:
        write_manifest(output, target, dflash2, args.external_data_mode)
    print(f"Created validated model in {output}")


if __name__ == "__main__":
    main()
