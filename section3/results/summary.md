## Quantization trade-off summary — Qwen/Qwen2.5-1.5B-Instruct (T4, greedy, 256 max new tokens)

| Metric | fp16 (baseline) | NF4 4-bit (bitsandbytes) | Delta |
| --- | --- | --- | --- |
| Weights VRAM | 2.88 GB | 1.08 GB | -62% |
| Peak VRAM (generation) | 2.90 GB | 1.15 GB | — |
| Avg throughput | 27.55 tok/s | 12.65 tok/s | x0.46 |
| Load time | 3.9s | 6.2s | — |
| Output quality | reference | see results/outputs.md (per-prompt comparison) | — |
