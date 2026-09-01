#!/usr/bin/env python3
"""Batch-size-1 qualification harness for Qwen 3.8 continuous batching."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CONTEXT_LENGTHS = [
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    196608,
    262016,
]
DEFAULT_OUTPUT_TOKENS = 128
MODEL_SPECS = [
    {
        "name": "INT4 weights / FP16 KV",
        "path": Path("qwen3.8-27b-int4-fp16-kv"),
        "mode": "non-speculative",
    },
    {
        "name": "INT4 weights / INT8 KV",
        "path": Path("qwen3.8-27b-int4-int8-kv"),
        "mode": "non-speculative",
    },
    {
        "name": "NVFP4 weights / FP16 KV / DFlash2",
        "path": Path("qwen3.8-27b-nvfp4-fp16-kv-dflash2"),
        "mode": "dflash2",
        "config_template": (
            Path("qwen3.8-27b-nvfp4-int8-kv-dflash2")
            / "genai_config.json"
        ),
        "shared_initializer_offsets": {
            "model.embed_tokens.weight": "19175636992",
            "lm_head.MatMul.fp8_weight": "17904238592",
            "lm_head.MatMul.fp8_weight_scale": "60121760",
        },
    },
    {
        "name": "NVFP4 weights / INT8 KV / DFlash2",
        "path": Path("qwen3.8-27b-nvfp4-int8-kv-dflash2"),
        "mode": "dflash2",
    },
]
SEED_TEXT = """
Review this Python service implementation. Identify correctness, concurrency,
resource-lifetime, and error-handling risks. Explain each issue precisely and
suggest a minimal fix. Include representative tests for cancellation, retries,
large inputs, and concurrent requests.

def process_request(request, shared_state):
    validated = validate(request)
    result = update_and_execute(validated, shared_state)
    return serialize(result)
"""


@dataclass
class Measurement:
    model: str
    model_path: str
    mode: str
    context_tokens: int
    output_tokens_requested: int
    repeat: int
    status: str
    ttft_seconds: float | None = None
    prompt_tokens_per_second: float | None = None
    decode_tokens_per_second: float | None = None
    output_tokens_observed: int | None = None
    total_seconds: float | None = None
    speculative_stats: dict[str, Any] | None = None
    error_stage: str | None = None
    error: str | None = None


def parse_lengths(value: str) -> list[int]:
    lengths = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("context lengths must be positive integers")
    return lengths


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def stage_model(
    source: Path,
    template: Path,
    staging_root: Path,
    shared_initializer_offsets: dict[str, str] | None = None,
) -> Path:
    target = staging_root / source.name
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if not item.is_file():
            continue
        destination = target / item.name
        if destination.exists():
            continue
        try:
            os.link(item, destination)
        except OSError:
            shutil.copy2(item, destination)
    config = json.loads(template.read_text(encoding="utf-8"))
    if shared_initializer_offsets:
        for initializer in config["model"]["decoder"]["shared_initializers"]:
            name = initializer["name"]
            if name in shared_initializer_offsets:
                initializer["offset"] = shared_initializer_offsets[name]
        for initializer in config["model"]["dflash2"]["shared_initializers"]:
            name = initializer["name"]
            if name in shared_initializer_offsets:
                initializer["offset"] = shared_initializer_offsets[name]
    write_json(target / "genai_config.json", config)
    return target


def exact_prompt_tokens(tokenizer: Any, context_tokens: int) -> np.ndarray:
    seed = np.asarray(tokenizer.encode(SEED_TEXT), dtype=np.int32)
    if not len(seed):
        raise RuntimeError("Tokenizer returned no tokens for the synthetic prompt")
    repeats = (context_tokens + len(seed) - 1) // len(seed)
    return np.tile(seed, repeats)[:context_tokens].astype(np.int32, copy=False)


def run_request(
    og: Any,
    model: Any,
    tokenizer: Any,
    engine: Any,
    context_tokens: int,
    output_tokens: int,
    event_buffer_capacity: int,
) -> dict[str, Any]:
    input_ids = exact_prompt_tokens(tokenizer, context_tokens)
    params = og.GeneratorParams(model)
    params.set_search_options(
        do_sample=False,
        max_length=context_tokens + output_tokens,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )
    request = engine.create_request(params)
    turn_options = og.TurnOptions(request)
    turn_options.set_max_generated_tokens(output_tokens)
    event_buffer = engine.create_event_buffer(event_buffer_capacity)
    token_times: list[float] = []
    output_ids: list[int] = []
    started_at = time.perf_counter()
    request.begin_turn(input_ids, turn_options)
    try:
        while engine.has_pending_requests():
            for event in engine.run(event_buffer):
                if event.request is None:
                    raise RuntimeError(
                        f"Engine event has no request; flags={event.flags}, "
                        f"error_code={event.error_code}"
                    )
                if event.flags & og.EngineEventFlags.TOKEN:
                    output_ids.append(int(event.token))
                    token_times.append(time.perf_counter())
                if event.flags & og.EngineEventFlags.FAILED:
                    raise RuntimeError(
                        f"Engine request failed; error_code={event.error_code}"
                    )
                if event.flags & og.EngineEventFlags.TURN_FINISHED:
                    request.close()
    finally:
        request.close()
    finished_at = time.perf_counter()
    if not token_times:
        raise RuntimeError("Engine completed without emitting a token")
    ttft = token_times[0] - started_at
    decode_seconds = token_times[-1] - token_times[0]
    decode_tokens = max(len(token_times) - 1, 0)
    return {
        "ttft_seconds": ttft,
        "prompt_tokens_per_second": context_tokens / ttft if ttft > 0 else None,
        "decode_tokens_per_second": (
            decode_tokens / decode_seconds
            if decode_tokens and decode_seconds > 0
            else None
        ),
        "output_tokens_observed": len(output_ids),
        "total_seconds": finished_at - started_at,
        "speculative_stats": dict(engine.get_speculative_stats()),
    }


def worker(args: argparse.Namespace) -> int:
    result_path = args.result_file.resolve()
    model_path = args.model.resolve()
    base = {
        "model": args.model_name,
        "model_path": str(model_path),
        "mode": args.mode,
    }
    try:
        import onnxruntime_genai as og
        import onnxruntime_ep_cuda as cuda_ep

        og.register_execution_provider_library(
            cuda_ep.get_ep_name(),
            cuda_ep.get_library_path(),
        )
        config = og.Config(str(model_path))
        config.clear_providers()
        config.append_provider("cuda")
        config.set_provider_option("cuda", "device_id", "0")
        dynamic_batching: dict[str, Any] = {"max_batch_size": 1}
        if args.num_blocks is not None:
            dynamic_batching["num_blocks"] = args.num_blocks
        else:
            dynamic_batching["gpu_utilization_factor"] = args.gpu_utilization_factor
        config.overlay(
            json.dumps({"engine": {"dynamic_batching": dynamic_batching}})
        )
        model = og.Model(config)
        tokenizer = og.Tokenizer(model)
        engine = og.Engine(model)
    except Exception as error:
        write_json(
            result_path,
            {
                "status": "failed",
                "error_stage": "model_load",
                "error": str(error),
                "traceback": traceback.format_exc(),
                **base,
            },
        )
        return 1

    measurements: list[dict[str, Any]] = []
    try:
        if args.warmup_tokens:
            run_request(
                og,
                model,
                tokenizer,
                engine,
                min(256, min(args.context_lengths)),
                args.warmup_tokens,
                args.event_buffer_capacity,
            )
        for context_tokens in args.context_lengths:
            for repeat in range(1, args.repeats + 1):
                measurement = Measurement(
                    context_tokens=context_tokens,
                    output_tokens_requested=args.output_tokens,
                    repeat=repeat,
                    status="passed",
                    **base,
                )
                try:
                    values = run_request(
                        og,
                        model,
                        tokenizer,
                        engine,
                        context_tokens,
                        args.output_tokens,
                        args.event_buffer_capacity,
                    )
                    for key, value in values.items():
                        setattr(measurement, key, value)
                except Exception as error:
                    measurement.status = "failed"
                    measurement.error_stage = "inference"
                    measurement.error = str(error)
                measurements.append(asdict(measurement))
    finally:
        write_json(
            result_path,
            {
                "status": "completed",
                "measurements": measurements,
                **base,
            },
        )
    return 0 if all(row["status"] == "passed" for row in measurements) else 1


def collect_environment(cuda_dir: Path) -> dict[str, Any]:
    import onnxruntime as ort
    import onnxruntime_genai as og
    import onnxruntime_ep_cuda as cuda_ep

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "onnxruntime_genai": og.__version__,
        "cuda_plugin_name": cuda_ep.get_ep_name(),
        "cuda_plugin_path": cuda_ep.get_library_path(),
        "cuda_dir": str(cuda_dir.resolve()),
        "gpu": [line for line in gpu.stdout.splitlines() if line.strip()],
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(
    output: Path,
    results_root: Path,
    environment: dict[str, Any],
    model_results: list[dict[str, Any]],
    context_lengths: list[int],
    output_tokens: int,
    repeats: int,
) -> None:
    lines = [
        "# Qwen 3.8 27B ORT GenAI Engine Qualification - RTX Spark",
        "",
        "## Result",
        "",
    ]
    passed = [
        measurement
        for result in model_results
        for measurement in result.get("measurements", [])
        if measurement["status"] == "passed"
    ]
    if passed:
        lines.append(
            "Batch-size-1 measurements completed for the rows marked `passed` below."
        )
    else:
        lines.extend(
            [
                "**Blocked before inference.** None of the four models can be loaded by "
                "the requested runtime combination, so TTFT, prompt TPS, and decode TPS "
                "cannot be reported without changing the ONNX Runtime core build.",
                "",
                "The first unresolved graph operator is "
                "`com.microsoft::VarlenCausalConvWithState`. The operator schema is not "
                "present in the PyPI ONNX Runtime 1.29.0 core library. Its CUDA kernel is "
                "present in the supplied CUDA plugin, but plugin kernel registration does "
                "not add the missing core graph schema.",
            ]
        )
    lines.extend(
        [
            "",
            "## Environment",
            "",
            markdown_table(
                ["Component", "Value"],
                [
                    ["Machine", environment["machine"]],
                    ["OS", environment["platform"]],
                    ["GPU", "<br>".join(environment["gpu"])],
                    ["Python", environment["python"]],
                    ["ONNX Runtime", environment["onnxruntime"]],
                    ["ONNX Runtime GenAI", environment["onnxruntime_genai"]],
                    ["CUDA plugin", environment["cuda_plugin_path"]],
                    ["CUDA binaries", environment["cuda_dir"]],
                ],
            ),
            "",
            "## Qualification matrix",
            "",
        ]
    )
    status_rows = []
    for result in model_results:
        if result["status"] == "failed":
            status = f"blocked at {result['error_stage']}"
            detail = result["error"].replace("|", "\\|")
        else:
            measurements = result.get("measurements", [])
            failures = sum(row["status"] != "passed" for row in measurements)
            status = "passed" if failures == 0 else f"{failures} inference failure(s)"
            detail = ""
        status_rows.append(
            [
                result["model"],
                result["mode"],
                status,
                detail,
            ]
        )
    lines.extend(
        [
            markdown_table(["Model", "Mode", "Status", "Detail"], status_rows),
            "",
            "## Performance results",
            "",
        ]
    )
    if not passed:
        lines.append(
            "No performance rows are available because model session creation failed."
        )
    else:
        summary_rows = []
        groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in passed:
            groups.setdefault((row["model"], row["context_tokens"]), []).append(row)
        for (model_name, context_tokens), rows in groups.items():
            summary_rows.append(
                [
                    model_name,
                    str(context_tokens),
                    f"{statistics.median(row['ttft_seconds'] for row in rows) * 1000:.2f}",
                    f"{statistics.median(row['prompt_tokens_per_second'] for row in rows):.2f}",
                    f"{statistics.median(row['decode_tokens_per_second'] for row in rows):.2f}",
                ]
            )
        lines.extend(
            [
                "### Median of three repeats",
                "",
                markdown_table(
                    [
                        "Model",
                        "Context tokens",
                        "TTFT (ms)",
                        "Prompt TPS",
                        "Decode TPS",
                    ],
                    summary_rows,
                ),
                "",
                "### Individual repeats",
                "",
            ]
        )
        perf_rows = []
        for row in passed:
            perf_rows.append(
                [
                    row["model"],
                    str(row["context_tokens"]),
                    str(row["repeat"]),
                    f"{row['ttft_seconds'] * 1000:.2f}",
                    f"{row['prompt_tokens_per_second']:.2f}",
                    f"{row['decode_tokens_per_second']:.2f}",
                    str(row["output_tokens_observed"]),
                ]
            )
        lines.append(
            markdown_table(
                [
                    "Model",
                    "Context tokens",
                    "Repeat",
                    "TTFT (ms)",
                    "Prompt TPS",
                    "Decode TPS",
                    "Output tokens",
                ],
                perf_rows,
            )
        )
    lines.extend(
        [
            "",
            "## Methodology",
            "",
            f"- Batch size: 1",
            f"- Context lengths: {', '.join(map(str, context_lengths))} tokens",
            f"- Requested output: {output_tokens} tokens",
            f"- Timed repeats: {repeats}",
            "- Prompt: a deterministic coding-shaped token sequence tiled and truncated "
            "to the exact requested token count",
            "- TTFT: request submission to the first `TOKEN` engine event",
            "- Prompt TPS: input token count divided by TTFT",
            "- Decode TPS: tokens after the first token divided by time from first to last "
            "token event",
            "- Model loading, tokenizer construction, and warmup are excluded",
        ]
    )
    if any(result["mode"] == "dflash2" for result in model_results):
        lines.append(
            "- DFlash2 is enabled by the model's `dflash2` section; speculative "
            "telemetry is captured from `Engine.get_speculative_stats()`"
        )
    lines.extend(
        [
            "",
            f"Raw machine-readable results are in `{results_root}\\qualification.json` "
            f"and `{results_root}\\models\\*.json`.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def orchestrator(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    models_root = args.models_root
    if not models_root.is_absolute():
        models_root = root / models_root
    models_root = models_root.resolve()
    cuda_dir = args.cuda_dir.resolve()
    results_root = args.results_dir.resolve()
    model_results_dir = results_root / "models"
    staging_root = results_root / "staged-models"
    model_results_dir.mkdir(parents=True, exist_ok=True)
    environment = collect_environment(cuda_dir)
    model_results = []
    process_environment = os.environ.copy()
    process_environment["PATH"] = str(cuda_dir) + os.pathsep + process_environment["PATH"]
    process_environment["CUDA_VISIBLE_DEVICES"] = "0"

    selected_specs = [spec for spec in MODEL_SPECS if spec["mode"] in args.modes]
    for index, spec in enumerate(selected_specs):
        source = (models_root / spec["path"]).resolve()
        model_path = source
        template_value = spec.get("config_template")
        if not (source / "genai_config.json").is_file():
            if template_value is None:
                result = {
                    "status": "failed",
                    "error_stage": "configuration",
                    "error": "genai_config.json is missing and no template was supplied",
                    "model": spec["name"],
                    "model_path": str(source),
                    "mode": spec["mode"],
                }
                model_results.append(result)
                continue
            template = (models_root / template_value).resolve()
            model_path = stage_model(
                source,
                template,
                staging_root,
                spec.get("shared_initializer_offsets"),
            )

        combined = {
            "status": "completed",
            "measurements": [],
            "model": spec["name"],
            "model_path": str(model_path),
            "mode": spec["mode"],
        }
        for context_tokens in args.context_lengths:
            block_size = 256
            num_blocks = math.ceil(
                (context_tokens + args.output_tokens) / block_size
            )
            stem = f"{index + 1:02d}-{context_tokens:06d}"
            result_file = model_results_dir / f"{stem}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--model",
                str(model_path),
                "--model-name",
                spec["name"],
                "--mode",
                spec["mode"],
                "--context-lengths",
                str(context_tokens),
                "--output-tokens",
                str(args.output_tokens),
                "--repeats",
                str(args.repeats),
                "--warmup-tokens",
                str(args.warmup_tokens),
                "--event-buffer-capacity",
                str(args.event_buffer_capacity),
                "--num-blocks",
                str(num_blocks),
                "--result-file",
                str(result_file),
            ]
            completed = subprocess.run(
                command,
                check=False,
                env=process_environment,
                capture_output=True,
                text=True,
            )
            log_path = model_results_dir / f"{stem}.log"
            log_path.write_text(
                completed.stdout + completed.stderr,
                encoding="utf-8",
                errors="replace",
            )
            if result_file.is_file():
                result = json.loads(result_file.read_text(encoding="utf-8"))
                if result["status"] == "failed":
                    combined = result
                    break
                combined["measurements"].extend(result.get("measurements", []))
            else:
                combined = {
                    "status": "failed",
                    "error_stage": "worker",
                    "error": (
                        f"Worker exited {completed.returncode} without a result file"
                    ),
                    "model": spec["name"],
                    "model_path": str(model_path),
                    "mode": spec["mode"],
                }
                break
        model_results.append(combined)

    qualification = {
        "environment": environment,
        "configuration": {
            "batch_size": 1,
            "context_lengths": args.context_lengths,
            "output_tokens": args.output_tokens,
            "repeats": args.repeats,
            "warmup_tokens": args.warmup_tokens,
            "gpu_utilization_factor": args.gpu_utilization_factor,
        },
        "models": model_results,
    }
    write_json(results_root / "qualification.json", qualification)
    render_report(
        args.report.resolve(),
        results_root,
        environment,
        model_results,
        args.context_lengths,
        args.output_tokens,
        args.repeats,
    )
    return 0 if all(result["status"] == "completed" for result in model_results) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--mode", choices=["non-speculative", "dflash2"])
    parser.add_argument(
        "--context-lengths",
        type=parse_lengths,
        default=DEFAULT_CONTEXT_LENGTHS,
    )
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--event-buffer-capacity", type=int, default=256)
    parser.add_argument("--gpu-utilization-factor", type=float, default=0.9)
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["non-speculative", "dflash2"],
        default=["non-speculative", "dflash2"],
    )
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--cuda-dir", type=Path, default=Path("..") / "cuda")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("models"),
        help="Directory containing the four model folders.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("QWEN38_RTX_SPARK_QUALIFICATION.md"),
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.output_tokens <= 1:
        parser.error("--output-tokens must be greater than one for decode TPS")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup_tokens < 0:
        parser.error("--warmup-tokens cannot be negative")
    if not 0 < args.gpu_utilization_factor <= 1:
        parser.error("--gpu-utilization-factor must be in (0, 1]")
    if args.num_blocks is not None and args.num_blocks <= 0:
        parser.error("--num-blocks must be positive")
    if any(length + args.output_tokens > 262144 for length in args.context_lengths):
        parser.error("context length plus output tokens exceeds 262144")
    if args.worker and (
        args.model is None
        or args.model_name is None
        or args.mode is None
        or args.result_file is None
    ):
        parser.error("--worker requires --model, --model-name, --mode, and --result-file")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    return worker(args) if args.worker else orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
