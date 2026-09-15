from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import requests


DEFAULT_PAYLOAD = Path(__file__).with_name("payload.json")


def run_request(
    endpoint: str,
    payload_template: dict,
    prompt: str,
    timeout: int,
) -> dict:
    body = json.loads(json.dumps(payload_template))
    body["messages"][0]["content"] = prompt
    started = time.perf_counter()
    first_event_at = None
    first_token_at = None
    last_token_at = None
    finish_reason = None
    usage = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    token_events = 0

    with requests.post(endpoint, json=body, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            now = time.perf_counter()
            if first_event_at is None:
                first_event_at = now
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                if content or reasoning:
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    token_events += 1
                    content_parts.append(content)
                    reasoning_parts.append(reasoning)

    completed = time.perf_counter()
    if first_event_at is None or first_token_at is None or last_token_at is None:
        raise RuntimeError("The streaming response did not contain token events.")
    if usage is None:
        raise RuntimeError("The streaming response did not include usage.")

    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    reasoning_tokens = int(
        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    )
    ttft = first_token_at - started
    decode_seconds = last_token_at - first_token_at
    wall_seconds = completed - started
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "visible_content_tokens": completion_tokens - reasoning_tokens,
        "token_events": token_events,
        "finish_reason": finish_reason,
        "ttft_seconds": ttft,
        "first_event_seconds": first_event_at - started,
        "prompt_tokens_per_second": prompt_tokens / ttft,
        "decode_tokens_per_second": (
            (completion_tokens - 1) / decode_seconds
            if completion_tokens > 1 and decode_seconds > 0
            else None
        ),
        "end_to_end_output_tokens_per_second": completion_tokens / wall_seconds,
        "wall_seconds": wall_seconds,
        "content": "".join(content_parts),
        "reasoning_preview": "".join(reasoning_parts)[:500],
    }


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for context in sorted({row["target_context"] for row in rows}):
        group = [row for row in rows if row["target_context"] == context]
        summaries.append(
            {
                "target_context": context,
                "actual_prompt_tokens_median": statistics.median(
                    row["prompt_tokens"] for row in group
                ),
                "ttft_seconds_median": statistics.median(
                    row["ttft_seconds"] for row in group
                ),
                "prompt_tokens_per_second_median": statistics.median(
                    row["prompt_tokens_per_second"] for row in group
                ),
                "decode_tokens_per_second_median": statistics.median(
                    row["decode_tokens_per_second"] for row in group
                ),
                "end_to_end_output_tokens_per_second_median": statistics.median(
                    row["end_to_end_output_tokens_per_second"] for row in group
                ),
                "completion_tokens_median": statistics.median(
                    row["completion_tokens"] for row in group
                ),
                "reasoning_tokens_median": statistics.median(
                    row["reasoning_tokens"] for row in group
                ),
            }
        )
    return summaries


def load_prompts(manifests: list[Path], case_id: str) -> dict[int, tuple[str, Path]]:
    prompts: dict[int, tuple[str, Path]] = {}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["prompts"]:
            if row["case_id"] != case_id:
                continue
            context = int(row["nominal_context"])
            path = Path(row["text_file"])
            if not path.is_absolute():
                path = manifest_path.parent / path
            prompts[context] = (path.read_text(encoding="utf-8"), path)
    if not prompts:
        raise ValueError(f"No prompts found for case {case_id!r}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark source-grounded SWE prompts through Foundry Local."
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:5272/v1/chat/completions")
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--case-id", default="cache-capacity-diagnostics")
    parser.add_argument("--context", type=int, action="append")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    payload["max_tokens"] = args.max_tokens
    if payload.get("messages") != [{"role": "user", "content": "{{PROMPT}}"}]:
        raise ValueError(
            f"{args.payload} must contain one user message whose content is {{PROMPT}}"
        )
    prompts = load_prompts(args.manifest, args.case_id)
    contexts = sorted(set(args.context or prompts))
    missing = set(contexts) - prompts.keys()
    if missing:
        raise ValueError(f"Missing requested contexts: {sorted(missing)}")

    rows = []
    if args.json_out.is_file():
        rows = json.loads(args.json_out.read_text(encoding="utf-8")).get("rows", [])
    completed = {(row["target_context"], row["repeat"]) for row in rows}

    for context in contexts:
        prompt, prompt_path = prompts[context]
        for repeat in range(1, args.repeat + 1):
            if (context, repeat) in completed:
                continue
            print(f"context={context} case={args.case_id} repeat={repeat}", flush=True)
            row = run_request(args.endpoint, payload, prompt, args.timeout)
            row.update(
                {
                    "target_context": context,
                    "case_id": args.case_id,
                    "repeat": repeat,
                    "prompt_path": str(prompt_path.resolve()),
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "request_bytes": len(prompt.encode("utf-8")),
                }
            )
            rows.append(row)
            output = {
                "endpoint": args.endpoint,
                "model": payload["model"],
                "case_id": args.case_id,
                "max_tokens": args.max_tokens,
                "repeat": args.repeat,
                "manifests": [str(path.resolve()) for path in args.manifest],
                "rows": rows,
                "summary": summarize(rows),
            }
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(output, indent=2), encoding="utf-8")
            visible = {
                key: row[key]
                for key in row
                if key not in ("content", "reasoning_preview")
            }
            print(json.dumps(visible), flush=True)


if __name__ == "__main__":
    main()
