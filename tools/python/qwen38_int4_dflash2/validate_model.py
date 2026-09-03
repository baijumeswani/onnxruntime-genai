#!/usr/bin/env python3
"""Validate retained Qwen 3.8 INT4 + DFlash2 graph invariants."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import onnx
from onnx import numpy_helper


def validate_external_data(model: onnx.ModelProto, model_dir: Path) -> bool:
    external_files = set()
    for initializer in model.graph.initializer:
        metadata = {item.key: item.value for item in initializer.external_data}
        if not metadata:
            continue
        path = model_dir / metadata["location"]
        if not path.is_file():
            raise RuntimeError(
                f"External data for {initializer.name} does not exist: {path}"
            )
        offset = int(metadata.get("offset", "0"))
        length = int(metadata["length"])
        if offset < 0 or length < 0 or offset + length > path.stat().st_size:
            raise RuntimeError(f"External range for {initializer.name} exceeds {path}.")
        external_files.add(path)
    return all(os.stat(path).st_nlink == 1 for path in external_files)


def validate_model_directory(model_dir: Path) -> None:
    config = json.loads((model_dir / "genai_config.json").read_text(encoding="utf-8"))
    decoder = config["model"]["decoder"]
    drafter = config["model"]["dflash2"]
    if decoder["filename"] != "model.onnx":
        raise RuntimeError("Decoder must use model.onnx.")
    if drafter["filename"] != "dflash2-int4.onnx":
        raise RuntimeError("DFlash2 must use dflash2-int4.onnx.")
    if decoder.get("state_update_capacity") != 7:
        raise RuntimeError("Decoder state_update_capacity must be 7.")
    if drafter.get("num_draft_tokens") != 7:
        raise RuntimeError("DFlash2 num_draft_tokens must be 7.")
    if drafter.get("sliding_window") != 2048:
        raise RuntimeError("DFlash2 sliding_window must be 2048.")
    if drafter.get("selector_top_k") != 16:
        raise RuntimeError("DFlash2 selector_top_k must be 16.")
    if tuple(drafter.get("aux_hidden_state_layers", ())) != (5, 19, 33, 47, 61):
        raise RuntimeError("DFlash2 auxiliary hidden-state layers are unexpected.")

    target = onnx.load(str(model_dir / "model.onnx"), load_external_data=False)
    target_nodes = {node.name: node for node in target.graph.node}
    for layer in range(64):
        fused = f"/model/layers.{layer}/mlp/gate_up_proj/MatMul_Q4"
        split = f"/model/layers.{layer}/mlp/gate_up_proj/Split"
        if fused not in target_nodes or split not in target_nodes:
            raise RuntimeError(f"Layer {layer} is missing gate/up fusion.")
        if f"/model/layers.{layer}/mlp/gate_proj/MatMul_Q4" in target_nodes:
            raise RuntimeError(f"Layer {layer} retains the original gate projection.")
        if f"/model/layers.{layer}/mlp/up_proj/MatMul_Q4" in target_nodes:
            raise RuntimeError(f"Layer {layer} retains the original up projection.")

    output_names = {output.name for output in target.graph.output}
    state_outputs = {name for name in output_names if name.startswith("state_update.")}
    if len(state_outputs) != 96:
        raise RuntimeError(f"Expected 96 state outputs, found {len(state_outputs)}.")
    if "aux_hidden_states" not in output_names:
        raise RuntimeError("Target is missing aux_hidden_states.")
    aux_concat = target_nodes.get("/model/aux_hidden_states/Concat")
    expected_aux_inputs = [
        f"/model/layers.{layer}/input_layernorm/output_3"
        for layer in drafter["aux_hidden_state_layers"]
    ]
    if aux_concat is None or list(aux_concat.input) != expected_aux_inputs:
        raise RuntimeError("Target auxiliary hidden-state taps disagree with config.")
    logits = next(output for output in target.graph.output if output.name == "logits")
    if logits.type.tensor_type.shape.dim[0].dim_param != "num_tokens":
        raise RuntimeError("Target logits must expose the num_tokens dimension.")
    target_head = next(node for node in target.graph.node if "logits" in node.output)
    target_head_attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in target_head.attribute
    }

    drafter_graph = onnx.load(
        str(model_dir / "dflash2-int4.onnx"), load_external_data=False
    )
    paged_attention_nodes = [
        node for node in drafter_graph.graph.node if node.op_type == "PagedAttention"
    ]
    if len(paged_attention_nodes) != 5:
        raise RuntimeError(
            f"Expected 5 DFlash2 PagedAttention nodes, found {len(paged_attention_nodes)}."
        )
    for node in paged_attention_nodes:
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        if attributes.get("local_window_size") != drafter["sliding_window"]:
            raise RuntimeError(
                f"{node.name} local_window_size disagrees with genai_config.json."
            )
    top_k_constant = next(
        (
            node
            for node in drafter_graph.graph.node
            if node.op_type == "Constant"
            and node.output
            and node.output[0] == "dflash2.const.12"
        ),
        None,
    )
    if top_k_constant is None:
        raise RuntimeError("DFlash2 selector top-k constant is missing.")
    top_k_tensor = next(
        attribute.t
        for attribute in top_k_constant.attribute
        if attribute.name == "value"
    )
    if (
        int(numpy_helper.to_array(top_k_tensor).reshape(-1)[0])
        != drafter["selector_top_k"]
    ):
        raise RuntimeError("DFlash2 selector graph disagrees with selector_top_k.")
    initializer_names = {
        initializer.name for initializer in drafter_graph.graph.initializer
    }
    constant_matmuls = [
        node
        for node in drafter_graph.graph.node
        if node.op_type == "MatMul"
        and len(node.input) > 1
        and node.input[1] in initializer_names
    ]
    if constant_matmuls:
        raise RuntimeError(
            "DFlash2 retains constant-weight MatMuls: "
            + ", ".join(node.name for node in constant_matmuls)
        )
    nbits_nodes = [
        node for node in drafter_graph.graph.node if node.op_type == "MatMulNBits"
    ]
    if len(nbits_nodes) != 58:
        raise RuntimeError(
            f"Expected 58 DFlash2 MatMulNBits nodes, found {len(nbits_nodes)}."
        )
    drafter_head = next(
        (node for node in nbits_nodes if node.name == "/lm_head/MatMul"),
        None,
    )
    if drafter_head is None:
        raise RuntimeError("DFlash2 Q4 vocabulary head is missing.")
    drafter_head_attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in drafter_head.attribute
    }
    for name in ("K", "N", "bits", "block_size"):
        if drafter_head_attributes.get(name) != target_head_attributes.get(name):
            raise RuntimeError(
                f"DFlash2 vocabulary-head {name} disagrees with the target."
            )
    head_weight = next(
        initializer
        for initializer in drafter_graph.graph.initializer
        if initializer.name == "lm_head.MatMul.weight_Q4"
    )
    expected_weight_dims = [
        drafter_head_attributes["N"],
        drafter_head_attributes["K"] // drafter_head_attributes["block_size"],
        drafter_head_attributes["block_size"] // 2,
    ]
    if list(head_weight.dims) != expected_weight_dims:
        raise RuntimeError(
            "DFlash2 Q4 vocabulary-head weight shape disagrees with its attributes."
        )
    dynamic_matmuls = [
        node for node in drafter_graph.graph.node if node.op_type == "MatMul"
    ]
    if [node.name for node in dynamic_matmuls] != ["/dflash2/selector/pair"]:
        raise RuntimeError("DFlash2 must retain only the dynamic selector MatMul.")

    shared = drafter.get("shared_initializers", [])
    shared_names = {initializer["name"] for initializer in shared}
    expected_shared = {
        "model.embed_tokens.weight",
        "lm_head.MatMul.weight_Q4",
        "lm_head.MatMul.weight_scales",
    }
    if shared_names != expected_shared:
        raise RuntimeError("DFlash2 shared initializer metadata is incomplete.")
    for initializer in drafter_graph.graph.initializer:
        if initializer.name not in expected_shared:
            continue
        metadata = {item.key: item.value for item in initializer.external_data}
        if metadata.get("location") != "model.onnx.data":
            raise RuntimeError(
                f"Shared initializer {initializer.name} does not use model.onnx.data."
            )

    target_safe_for_checker = validate_external_data(target, model_dir)
    drafter_safe_for_checker = validate_external_data(drafter_graph, model_dir)
    if target_safe_for_checker:
        onnx.checker.check_model(str(model_dir / "model.onnx"))
    if drafter_safe_for_checker:
        onnx.checker.check_model(str(model_dir / "dflash2-int4.onnx"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.model.expanduser().resolve()
    validate_model_directory(model_dir)
    print(f"Validated {model_dir}")


if __name__ == "__main__":
    main()
