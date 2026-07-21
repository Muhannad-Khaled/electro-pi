"""Concurrency benchmark for the streaming endpoint.

Runs a single-request warm baseline, then N concurrent streaming requests,
and records per-request TTFT (time to the first *token SSE event*, not first
HTTP byte), total latency, and tokens/sec. The server serializes inference
behind a lock, so under concurrency the requests queue — the point of this
test is to measure that degradation honestly.

Usage: python load_test.py [base_url] [label]
  label controls output filenames: results/load_test_<label>.{json,md}
  (defaults: http://localhost:8000, native)
"""

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "native"

PROMPT = "Explain, for a general audience, why the sky is blue."
MAX_TOKENS = 128
CONCURRENCY = 10
# Generous per-request timeout: with 10 queued requests at ~10s each, the
# last one can legitimately take ~2 minutes on CPU.
TIMEOUT = httpx.Timeout(600, connect=10)


async def stream_once(client: httpx.AsyncClient, request_id: int) -> dict:
    payload = {"prompt": PROMPT, "max_tokens": MAX_TOKENS, "temperature": 0.7}
    t_start = time.perf_counter()
    ttft = None
    tokens = 0
    async with client.stream("POST", f"{BASE}/v1/generate/stream", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"server error: {event['error']}")
            if ttft is None:
                ttft = time.perf_counter() - t_start
            tokens += 1
    if ttft is None:
        raise RuntimeError(f"request {request_id}: stream produced no tokens")
    total = time.perf_counter() - t_start
    return {
        "request_id": request_id,
        "ttft_s": round(ttft, 3),
        "total_s": round(total, 3),
        "tokens": tokens,
        "tokens_per_s": round(tokens / (total - ttft), 2) if total > ttft else None,
    }


def summarize(rows: list[dict]) -> dict:
    ttfts = [r["ttft_s"] for r in rows]
    totals = [r["total_s"] for r in rows]
    return {
        "ttft_min_s": min(ttfts),
        "ttft_median_s": round(statistics.median(ttfts), 3),
        "ttft_max_s": max(ttfts),
        "total_min_s": min(totals),
        "total_median_s": round(statistics.median(totals), 3),
        "total_max_s": max(totals),
    }


def write_markdown(path: Path, baseline: dict, rows: list[dict], summary: dict) -> None:
    lines = [
        f"# Load test — {LABEL}",
        "",
        f"Prompt: fixed, `max_tokens={MAX_TOKENS}`, concurrency={CONCURRENCY}, endpoint `/v1/generate/stream`.",
        "",
        "## Single-request baseline (warm)",
        "",
        "| TTFT (s) | Total (s) | Tokens | Tokens/s |",
        "|---:|---:|---:|---:|",
        f"| {baseline['ttft_s']} | {baseline['total_s']} | {baseline['tokens']} | {baseline['tokens_per_s']} |",
        "",
        f"## {CONCURRENCY} concurrent requests",
        "",
        "| Request | TTFT (s) | Total (s) | Tokens | Tokens/s |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: r["ttft_s"]):
        lines.append(
            f"| {r['request_id']} | {r['ttft_s']} | {r['total_s']} | {r['tokens']} | {r['tokens_per_s']} |"
        )
    lines += [
        "",
        "## Summary",
        "",
        "| Metric | Min | Median | Max |",
        "|---|---:|---:|---:|",
        f"| TTFT (s) | {summary['ttft_min_s']} | {summary['ttft_median_s']} | {summary['ttft_max_s']} |",
        f"| Total latency (s) | {summary['total_min_s']} | {summary['total_median_s']} | {summary['total_max_s']} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Warm-up: first inference after model load pays one-time costs
        # (page-cache warmup); discard it so the baseline is representative.
        print("warm-up request ...")
        await stream_once(client, -1)

        print("single-request baseline ...")
        baseline = await stream_once(client, 0)
        print(f"  baseline: TTFT {baseline['ttft_s']}s, total {baseline['total_s']}s")

        print(f"{CONCURRENCY} concurrent requests ...")
        rows = await asyncio.gather(
            *(stream_once(client, i + 1) for i in range(CONCURRENCY))
        )

    summary = summarize(rows)
    out = {
        "label": LABEL,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "concurrency": CONCURRENCY,
        "baseline": baseline,
        "concurrent_requests": sorted(rows, key=lambda r: r["ttft_s"]),
        "summary": summary,
    }
    json_path = results_dir / f"load_test_{LABEL}.json"
    md_path = results_dir / f"load_test_{LABEL}.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_markdown(md_path, baseline, rows, summary)
    print(f"wrote {json_path} and {md_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
