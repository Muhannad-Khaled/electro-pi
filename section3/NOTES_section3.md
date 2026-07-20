# Section 3 — Quantization: Write-up (FINAL)

## Setup

- Model: Qwen/Qwen2.5-1.5B-Instruct
- Baseline: fp16 (not bf16 — the T4 is a Turing GPU with no native bf16 support)
- Quantized: bitsandbytes NF4 4-bit with double quantization, fp16 compute dtype
- Decoding: greedy (`do_sample=False`) so output differences are attributable to
  quantization, not sampling noise. One warmup generation excluded from timing.
- Memory measured with `torch.cuda.max_memory_allocated()` to isolate model
  allocations from CUDA context overhead (~500 MB on T4).

## Results (measured on Colab free T4 — raw data in results/results.json)

| Metric | fp16 | NF4 4-bit | Delta |
| --- | --- | --- | --- |
| Weights VRAM | 2.88 GB | 1.08 GB | **-62%** |
| Peak VRAM (generation) | 2.90 GB | 1.15 GB | -60% |
| Avg throughput | 27.55 tok/s | 12.65 tok/s | **x0.46 (slower)** |
| Load time | 3.9s | 6.2s | quantize-at-load cost |
| Quality (5 fixed prompts) | reference | degraded on code + Arabic; equal elsewhere | — |

**Honest observation #1 — quantization did NOT make inference faster.** NF4 cut
memory by 62% but *halved* throughput. This is expected behavior, not a bug:
bitsandbytes dequantizes weights to fp16 on the fly at every forward pass, so
it trades compute overhead for memory savings. Its win is fitting bigger models
on small GPUs, not speed. Methods with optimized INT4 kernels (GPTQ/AWQ) are
the ones that deliver actual speedups.

**Honest observation #2 — quality degradation was concentrated, not uniform.**
Factual QA and summarization were essentially unaffected. The damage appeared
exactly where low-bit quantization is known to hurt first:

- **Code generation:** the NF4 version produced logically broken code — a
  memoized helper that recurses on accumulators while its base cases check an
  outer `n` that never changes, i.e. guaranteed infinite recursion for n ≥ 2,
  plus a wrong inline comment (`fib(5) # Output: 3`). For fairness: the fp16
  version also had a defect (used `@lru_cache` without importing it, and hit
  the 256-token generation cap), but its *logic* was sound — a one-line import
  fix vs. a broken algorithm. That distinction is the real quality gap.
- **Arabic generation:** the NF4 output leaked Chinese characters mid-Arabic
  ("التدريب预先") — a classic quantization artifact on underrepresented
  languages (amplified here by Qwen's Chinese-heavy pretraining) — and its
  example was semantically incoherent. fp16 stayed fluent.
- **Instruction following (subtle):** NF4 silently switched the currency from
  EGP to $ in the math problem and ignored the "exactly 4 bullet points"
  format constraint; fp16 respected both.

**Limitations:** 5 prompts is sufficient for the qualitative comparison asked
here, but any statistical claim would need a proper perplexity/benchmark eval
on a larger set. Greedy decoding makes results deterministic but represents
only one decoding configuration.

## When would I pick GPTQ/AWQ over bitsandbytes, or GGUF over both?

**bitsandbytes** is a development-time tool for me, not a production one. Its
strength is zero friction: no calibration data, no pre-quantized artifact, just
a config flag at load time — ideal for experimentation and QLoRA fine-tuning.
But my own measurement above shows the cost: you pay an on-the-fly
dequantization tax on every request, forever.

**GPTQ/AWQ** are what I would deploy on a GPU server. Both are post-training
methods that produce a pre-quantized artifact using a calibration set, and both
have optimized INT4 kernels (Marlin, ExLlama) that make inference genuinely
*faster* than fp16 — the opposite of what I measured with bitsandbytes. AWQ
tends to preserve quality slightly better on instruction-tuned models because
it protects activation-salient weights; GPTQ has broader ecosystem support. The
costs: a one-time calibration step, and the risk that a poorly chosen
calibration set skews quality on real traffic. Both are first-class citizens in
vLLM/TGI, which is exactly the serving path for Section 4.

**GGUF (llama.cpp)** wins whenever the target is CPU, Apple Silicon, or edge
devices — environments where CUDA-centric stacks don't exist. Its k-quant
schemes (e.g. Q4_K_M) give fine-grained size/quality control, and a single
portable file with no Python dependency is operationally attractive for
on-device or air-gapped deployment. I would not pick it for a high-throughput
GPU server: llama.cpp's batching and paged-attention story is far behind vLLM.

**Rule of thumb:** bitsandbytes for experiments and QLoRA; GPTQ/AWQ + vLLM for
GPU production serving; GGUF for CPU/edge/local distribution. Given the quality
cliff I observed on code and Arabic at 4-bit, for a production assistant
serving Arabic users I would also evaluate 8-bit (or AWQ 4-bit with a
domain-matched calibration set) before committing to NF4-class quantization.
