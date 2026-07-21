# Load test — docker

Prompt: fixed, `max_tokens=128`, concurrency=10, endpoint `/v1/generate/stream`.

## Single-request baseline (warm)

| TTFT (s) | Total (s) | Tokens | Tokens/s |
|---:|---:|---:|---:|
| 0.543 | 10.25 | 93 | 9.58 |

## 10 concurrent requests

| Request | TTFT (s) | Total (s) | Tokens | Tokens/s |
|---:|---:|---:|---:|---:|
| 2 | 0.408 | 22.088 | 119 | 5.49 |
| 3 | 22.231 | 97.796 | 128 | 1.69 |
| 7 | 98.297 | 116.868 | 128 | 6.89 |
| 1 | 117.875 | 139.658 | 128 | 5.88 |
| 5 | 139.82 | 148.496 | 88 | 10.14 |
| 6 | 148.628 | 159.622 | 128 | 11.64 |
| 9 | 159.779 | 170.746 | 128 | 11.67 |
| 10 | 170.844 | 184.432 | 128 | 9.42 |
| 8 | 184.518 | 192.554 | 79 | 9.83 |
| 4 | 192.67 | 201.603 | 88 | 9.85 |

## Summary

| Metric | Min | Median | Max |
|---|---:|---:|---:|
| TTFT (s) | 0.408 | 144.224 | 192.67 |
| Total latency (s) | 22.088 | 154.059 | 201.603 |
