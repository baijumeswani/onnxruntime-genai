#!/usr/bin/env python3
"""Offline-pack eligible target weights for CUDA fpA/intB experiments.

The serialized layout is an ORT/CUTLASS implementation detail. This tool is
target-only and is not part of the default production conversion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

ROW_PERMUTATION_16X4 = np.asarray(
    [
        0,
        1,
        8,
        9,
        16,
        17,
        24,
        25,
        2,
        3,
        10,
        11,
        18,
        19,
        26,
        27,
        4,
        5,
        12,
        13,
        20,
        21,
        28,
        29,
        6,
        7,
        14,
        15,
        22,
        23,
        30,
        31,
    ],
    dtype=np.int64,
)


def attributes(node) -> dict:
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def external_metadata(initializer) -> dict[str, str]:
    return {item.key: item.value for item in initializer.external_data}


def set_external_metadata(initializer, location: str, offset: int, length: int) -> None:
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


def pack_int4_sm80(raw: bytes, n: int, k: int) -> np.ndarray:
    quantized = np.frombuffer(raw, dtype=np.uint8).reshape(n, k // 2)
    unpacked = np.empty((n, k), dtype=np.uint8)
    unpacked[:, 0::2] = quantized & 0x0F
    unpacked[:, 1::2] = quantized >> 4
    signed_transpose = ((unpacked.astype(np.int16) - 8) & 0x0F).astype(np.uint8).T
    tensor = signed_transpose[:, 0::2] | (signed_transpose[:, 1::2] << 4)

    rows = tensor.shape[0]
    row_index = (
        np.arange(rows, dtype=np.int64) // 32 * 32
        + ROW_PERMUTATION_16X4[np.arange(rows, dtype=np.int64) % 32]
    )
    tensor = tensor[row_index, :]

    low = (tensor & 0x0F).T
    high = (tensor >> 4).T
    transposed = np.stack((low, high), axis=1).reshape(n, k)
    tensor = (transposed[:, 0::2] | (transposed[:, 1::2] << 4)).reshape(k, n // 2)
    tensor = tensor.reshape(-1, 4, k // 64, 32).transpose(0, 2, 1, 3).reshape(k, n // 2)

    low = (tensor & 0x0F)[..., None]
    high = (tensor >> 4)[..., None]
    rebias = np.concatenate((low, high), axis=-1).reshape(k, n)
    rebias = rebias.reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].reshape(k, n)
    rebias = rebias.astype(np.int16)
    rebias += -16 * (rebias > 7) + 8
    return (rebias[:, 0::2] | (rebias[:, 1::2] << 4)).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    output_model = args.output_model.expanduser().resolve()
    output_data = args.output_data.expanduser().resolve()
    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    value_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)
    }
    packed_names = set()
    packed_nodes = 0
    output_data.parent.mkdir(parents=True, exist_ok=True)

    with output_data.open("wb") as destination:
        for node in model.graph.node:
            if node.op_type != "MatMulNBits" or len(node.input) != 3:
                continue
            attrs = attributes(node)
            if (
                attrs.get("bits") != 4
                or attrs.get("block_size") != 32
                or attrs.get("N", 0) % 64 != 0
                or attrs.get("K", 0) % 32 != 0
                or value_types.get(node.input[0]) != onnx.TensorProto.FLOAT16
            ):
                continue

            weight = initializers[node.input[1]]
            if weight.name not in packed_names:
                metadata = external_metadata(weight)
                source_path = model_path.parent / metadata["location"]
                length = int(metadata["length"])
                with source_path.open("rb") as source:
                    source.seek(int(metadata["offset"]))
                    raw = source.read(length)
                if len(raw) != length:
                    raise RuntimeError(f"Could not read all bytes for {weight.name}.")

                packed = pack_int4_sm80(raw, int(attrs["N"]), int(attrs["K"])).reshape(
                    -1
                )
                if packed.nbytes != length:
                    raise RuntimeError(
                        f"Packed byte count changed for {weight.name}: "
                        f"{packed.nbytes} != {length}."
                    )
                padding = (-destination.tell()) % 64
                if padding:
                    destination.write(bytes(padding))
                offset = destination.tell()
                packed.tofile(destination)
                set_external_metadata(weight, output_data.name, offset, length)
                packed_names.add(weight.name)

            node.attribute.append(helper.make_attribute("weight_prepacked", 1))
            packed_nodes += 1

    onnx.save_model(model, str(output_model))
    print(
        f"Packed {len(packed_names)} target weights used by {packed_nodes} "
        f"MatMulNBits nodes into {output_data} "
        f"({output_data.stat().st_size / 2**30:.2f} GiB)."
    )


if __name__ == "__main__":
    main()
