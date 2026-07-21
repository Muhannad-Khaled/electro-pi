# Load test — native

Prompt: fixed, `max_tokens=128`, concurrency=10, endpoint `/v1/generate/stream`.

## Single-request baseline (warm)

| TTFT (s) | Total (s) | Tokens | Tokens/s |
|---:|---:|---:|---:|
| 0.352 | 10.284 | 128 | 12.89 |

## 10 concurrent requests

| Request | TTFT (s) | Total (s) | Tokens | Tokens/s |
|---:|---:|---:|---:|---:|
| 9 | 0.362 | 6.692 | 72 | 11.37 |
| 3 | 6.77 | 16.737 | 119 | 11.94 |
| 4 | 16.806 | 27.463 | 128 | 12.01 |
| 10 | 27.548 | 37.812 | 128 | 12.47 |
| 7 | 37.901 | 47.973 | 128 | 12.71 |
| 8 | 48.063 | 58.409 | 128 | 12.37 |
| 1 | 58.499 | 65.172 | 82 | 12.29 |
| 2 | 65.239 | 72.271 | 90 | 12.8 |
| 6 | 72.332 | 82.246 | 128 | 12.91 |
| 5 | 82.336 | 92.437 | 128 | 12.67 |

## Summary

| Metric | Min | Median | Max |
|---|---:|---:|---:|
| TTFT (s) | 0.362 | 42.982 | 82.336 |
| Total latency (s) | 6.692 | 53.191 | 92.437 |
