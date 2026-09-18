"""Benchmark MAI Code 1 Flash at exact context lengths through natural EOS."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import onnxruntime_genai as og
import psutil


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_value(value: str) -> tuple[str, str]:
    key, separator, option = value.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError("Expected KEY=VALUE")
    return key, option


def gpu_memory_mib() -> float | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        try:
            return float(line.strip())
        except ValueError:
            continue
    return None


class MemoryMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.process = psutil.Process()
        self.samples: list[dict] = []
        self.phase = "startup"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self.sample()

    def sample(self) -> None:
        virtual = psutil.virtual_memory()
        self.samples.append(
            {
                "time": time.time(),
                "phase": self.phase,
                "gpu_memory_mib": gpu_memory_mib(),
                "process_rss_mib": self.process.memory_info().rss / 2**20,
                "system_used_mib": virtual.used / 2**20,
                "system_available_mib": virtual.available / 2**20,
            }
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def summary(self, phase: str | None = None) -> dict:
        rows = [row for row in self.samples if phase is None or row["phase"] == phase]
        if not rows:
            return {}

        def maximum(key: str) -> float | None:
            values = [row[key] for row in rows if row[key] is not None]
            return max(values) if values else None

        def minimum(key: str) -> float | None:
            values = [row[key] for row in rows if row[key] is not None]
            return min(values) if values else None

        return {
            "samples": len(rows),
            "peak_gpu_memory_mib": maximum("gpu_memory_mib"),
            "peak_process_rss_mib": maximum("process_rss_mib"),
            "peak_system_used_mib": maximum("system_used_mib"),
            "minimum_system_available_mib": minimum("system_available_mib"),
        }


def encode(tokenizer, template: str, text: str) -> np.ndarray:
    rendered = tokenizer.apply_chat_template(
        messages=json.dumps([{"role": "user", "content": text}]),
        template_str=template,
        add_generation_prompt=True,
    )
    return np.asarray(tokenizer.encode(rendered), dtype=np.int32)


def exact_prompt(tokenizer, template: str, source: str, target: int) -> tuple[str, np.ndarray]:
    low, high = 0, len(source)
    if len(encode(tokenizer, template, source)) < target:
        raise ValueError(f"Source corpus is too small for {target:,} tokens")
    while low < high:
        middle = (low + high + 1) // 2
        if len(encode(tokenizer, template, source[:middle])) <= target:
            low = middle
        else:
            high = middle - 1
    for end in range(max(0, low - 512), min(len(source), low + 513)):
        text = source[:end]
        ids = encode(tokenizer, template, text)
        if len(ids) == target:
            return text, ids
    raise ValueError(f"Could not construct an exact {target:,}-token prompt")


def run_turn(
    engine,
    tokenizer,
    event_buffer,
    input_ids,
    maximum_output: int,
    profiler=None,
    capture: str | None = None,
) -> dict:
    request = engine.create_request()
    options = og.TurnOptions(request)
    options.set_max_generated_tokens(maximum_output)
    options.set_do_sample(False)
    options.set_top_k(1)
    options.set_top_p(1.0)
    options.set_temperature(1.0)
    started = time.perf_counter()
    token_times: list[float] = []
    output_ids: list[int] = []
    finish_reason = None
    profiling = False
    if capture in ("full", "prefill"):
        if profiler.cudaProfilerStart() != 0:
            raise RuntimeError("cudaProfilerStart failed")
        profiling = True
    request.begin_turn(input_ids, options)
    try:
        while engine.has_pending_requests():
            for event in engine.run(event_buffer):
                if event.flags & og.EngineEventFlags.FAILED:
                    raise RuntimeError(f"Engine request failed: {event.error_code}")
                if event.flags & og.EngineEventFlags.TOKEN:
                    output_ids.append(int(event.token))
                    token_times.append(time.perf_counter())
                    if capture == "prefill" and profiling:
                        if profiler.cudaProfilerStop() != 0:
                            raise RuntimeError("cudaProfilerStop failed")
                        profiling = False
                    elif capture == "decode" and not profiling:
                        if profiler.cudaProfilerStart() != 0:
                            raise RuntimeError("cudaProfilerStart failed")
                        profiling = True
                if event.flags & og.EngineEventFlags.TURN_FINISHED:
                    finish_reason = str(event.finish_reason)
                    request.close()
    finally:
        request.close()
        if profiling:
            if profiler.cudaProfilerStop() != 0:
                raise RuntimeError("cudaProfilerStop failed")
            profiling = False
    ended = time.perf_counter()
    if not token_times:
        raise RuntimeError("Request completed without output tokens")
    decode_seconds = token_times[-1] - token_times[0]
    return {
        "prompt_tokens": len(input_ids),
        "output_tokens": len(output_ids),
        "finish_reason": finish_reason,
        "ended_at_eos": finish_reason == str(og.FinishReason.EOS),
        "ttft_seconds": token_times[0] - started,
        "prompt_tokens_per_second": len(input_ids) / (token_times[0] - started),
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (
            (len(output_ids) - 1) / decode_seconds if len(output_ids) > 1 and decode_seconds > 0 else None
        ),
        "total_seconds": ended - started,
        "output_token_ids": output_ids,
        "response": tokenizer.decode(output_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--source-prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", type=int, action="append", required=True)
    parser.add_argument(
        "--output-tokens",
        type=int,
        help="Generate exactly this many tokens instead of running to EOS.",
    )
    parser.add_argument("--capture", choices=("prefill", "decode", "full"))
    parser.add_argument("--cudart", type=Path)
    parser.add_argument("--decoder-filename")
    parser.add_argument("--num-blocks", type=int)
    parser.add_argument("--session-option", type=key_value, action="append", default=[])
    parser.add_argument("--provider-option", type=key_value, action="append", default=[])
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    contexts = sorted(set(args.context))
    if args.capture and len(contexts) != 1:
        parser.error("--capture requires exactly one context")
    if args.capture and args.cudart is None:
        parser.error("--capture requires --cudart")
    existing = json.loads(result_path.read_text()) if args.skip_completed and result_path.exists() else None
    completed = {row["prompt_tokens"] for row in existing.get("rows", [])} if existing else set()

    monitor = MemoryMonitor()
    monitor.start()
    baseline_gpu = gpu_memory_mib()
    result = existing or {
        "model": str(args.model.resolve()),
        "provider": str(args.provider.resolve()),
        "contexts": contexts,
        "output_tokens": args.output_tokens,
        "baseline_gpu_memory_mib": baseline_gpu,
        "runtime": {
            "onnxruntime_genai": og.__version__,
            "onnxruntime_genai_commit": getattr(og, "__commit__", None),
        },
        "configuration": {
            "decoder_filename": args.decoder_filename,
            "num_blocks": args.num_blocks,
            "session_options": dict(args.session_option),
            "provider_options": dict(args.provider_option),
        },
        "load": {},
        "prompts": [],
        "rows": [],
    }
    try:
        monitor.phase = "model_load"
        og.register_execution_provider_library("CUDA.GenAI", str(args.provider.resolve()))
        config = og.Config(str(args.model.resolve()))
        config.clear_providers()
        config.append_provider("cuda")
        config.set_provider_option("cuda", "device_id", "0")
        for key, value in args.provider_option:
            config.set_provider_option("cuda", key, value)
        overlay = {"engine": {"dynamic_batching": {"max_batch_size": 1}}}
        if args.num_blocks is not None:
            overlay["engine"]["dynamic_batching"]["num_blocks"] = args.num_blocks
        if args.decoder_filename is not None or args.session_option:
            decoder = overlay.setdefault("model", {}).setdefault("decoder", {})
            if args.decoder_filename is not None:
                decoder["filename"] = args.decoder_filename
            decoder["session_options"] = dict(args.session_option)
        config.overlay(json.dumps(overlay))

        started = time.perf_counter()
        model = og.Model(config)
        result["load"]["model_seconds"] = time.perf_counter() - started

        monitor.phase = "tokenizer_init"
        started = time.perf_counter()
        tokenizer = og.Tokenizer(model)
        result["load"]["tokenizer_seconds"] = time.perf_counter() - started

        monitor.phase = "engine_init"
        started = time.perf_counter()
        engine = og.Engine(model)
        event_buffer = engine.create_event_buffer(64)
        result["load"]["engine_seconds"] = time.perf_counter() - started
        result["load"]["total_seconds"] = sum(
            result["load"][key] for key in ("model_seconds", "tokenizer_seconds", "engine_seconds")
        )
        result["load"]["memory"] = monitor.summary()
        save(result_path, result)

        template = (args.model / "chat_template.jinja").read_text(encoding="utf-8")
        frozen = json.loads(args.source_prompts.read_text(encoding="utf-8"))
        source = max(frozen, key=lambda row: row["context_length"])["text"]
        prompts = {}
        for context in contexts:
            text, ids = exact_prompt(tokenizer, template, source, context)
            prompts[context] = ids
            prompt_path = args.output_dir / f"prompt-{context}.txt"
            prompt_path.write_text(text, encoding="utf-8", newline="\n")
            metadata = {
                "prompt_tokens": context,
                "path": str(prompt_path),
                "sha256": sha256(prompt_path),
            }
            if not any(row["prompt_tokens"] == context for row in result["prompts"]):
                result["prompts"].append(metadata)
        save(result_path, result)

        monitor.phase = "warmup"
        warmup_ids = prompts[min(contexts)][: min(256, min(contexts))]
        run_turn(engine, tokenizer, event_buffer, warmup_ids, 16)

        profiler = ctypes.CDLL(str(args.cudart.resolve())) if args.capture else None
        context_limit = json.loads((args.model / "genai_config.json").read_text())["model"]["context_length"]
        for context in contexts:
            if context in completed:
                continue
            monitor.phase = f"context_{context}"
            before = len(monitor.samples)
            maximum_output = args.output_tokens or (context_limit - context - 16)
            row = run_turn(
                engine,
                tokenizer,
                event_buffer,
                prompts[context],
                maximum_output,
                profiler=profiler,
                capture=args.capture,
            )
            response_path = args.output_dir / f"response-{context}.txt"
            response_path.write_text(row.pop("response"), encoding="utf-8", newline="\n")
            token_path = args.output_dir / f"response-{context}-tokens.json"
            token_path.write_text(json.dumps(row.pop("output_token_ids")) + "\n", encoding="utf-8")
            row.update(
                response_path=str(response_path),
                response_tokens_path=str(token_path),
                maximum_output_safety=maximum_output,
                completed_requested_output=(
                    args.output_tokens is not None and row["output_tokens"] == args.output_tokens
                ),
                profiler_capture=args.capture,
                memory=monitor.summary(f"context_{context}"),
                memory_sample_start=before,
            )
            result["rows"].append(row)
            save(result_path, result)
            print(json.dumps(row), flush=True)
    finally:
        monitor.phase = "shutdown"
        monitor.stop()
        result["memory"] = {
            "baseline_gpu_memory_mib": baseline_gpu,
            "overall": monitor.summary(),
            "samples": monitor.samples,
        }
        save(result_path, result)


if __name__ == "__main__":
    main()
