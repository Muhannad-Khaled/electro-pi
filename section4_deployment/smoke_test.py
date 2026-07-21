"""Minimal end-to-end check: is the server up and does the model answer?

Usage: python smoke_test.py [base_url]   (default http://localhost:8000)
"""

import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def main() -> int:
    health = httpx.get(f"{BASE}/health", timeout=10).json()
    print(f"health: {health}")
    if not health.get("model_loaded"):
        print("FAIL: model not loaded")
        return 1

    resp = httpx.post(
        f"{BASE}/v1/generate",
        json={"prompt": "In one sentence, what is a quantized LLM?", "max_tokens": 64},
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()
    print(f"completion ({body['completion_tokens']} tokens, {body['latency_ms']:.0f} ms):")
    print(body["text"])
    return 0 if body["text"].strip() else 1


if __name__ == "__main__":
    sys.exit(main())
