#!/usr/bin/env python3
"""Section 3 - Quantization benchmark: fp16 vs bitsandbytes NF4 (4-bit).

Compares Qwen/Qwen2.5-1.5B-Instruct on:
  - memory footprint: weights VRAM after load + peak VRAM during generation
  - throughput: tokens/sec (greedy decoding, warmup run excluded)
  - qualitative output on 5 fixed prompts (see prompts.json)

Target environment: Google Colab free tier (NVIDIA T4, 16 GB VRAM).

Setup (Colab cell):
    !pip install -q "transformers>=4.44" accelerate bitsandbytes

Run:
    !python benchmark.py --out results/

Outputs:
    results/results.json   - raw measurements + generations
    results/outputs.md     - side-by-side answers per prompt (fp16 vs nf4)
    results/summary.md     - trade-off table (paste into README/NOTES)

Design notes / trade-offs (intentional):
  - do_sample=False (greedy) so both variants are deterministic and any
    output difference is attributable to quantization, not sampling noise.
  - One warmup generation before timing: the first call pays CUDA kernel
    init / cudnn autotune costs that would otherwise pollute tokens/sec.
  - torch.cuda.max_memory_allocated() instead of nvidia-smi: isolates the
    model's allocations from the CUDA context overhead (~500 MB on T4).
  - fp16, not bf16: T4 is Turing and has no native bf16 support.
  - Models are benchmarked sequentially (load -> measure -> free) so both
    runs see the same clean GPU state.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 256
WARMUP_PROMPT = "Say hello in one short sentence."
GB = 1024**3


# --------------------------------------------------------------------------- #
# Environment / memory helpers
# --------------------------------------------------------------------------- #
def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU not found. In Colab: Runtime -> Change runtime type -> T4 GPU."
        )


def free_gpu() -> None:
    """Release cached allocations and reset peak-memory counters."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(mode: str):
    """Load the model in the requested precision.

    mode: 'fp16' (baseline) or 'nf4' (bitsandbytes 4-bit NormalFloat).
    """
    kwargs: dict = {"device_map": {"": 0}, "low_cpu_mem_usage": True}

    if mode == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif mode == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,       # quantizes the quant constants too (~0.4 bit/param saved)
            bnb_4bit_compute_dtype=torch.float16,  # matmuls still run in fp16 after dequant
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'fp16' or 'nf4')")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Generation + timing
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def generate(model, tokenizer, user_prompt: str) -> dict:
    """Greedy-generate a reply and return text + timing stats."""
    messages = [{"role": "user", "content": user_prompt}]
    # return_dict=True explicitly: newer transformers versions return a
    # BatchEncoding here by default; passing that positionally to generate()
    # crashes on `.shape`. Unpacking with ** also forwards attention_mask.
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    new_tokens = int(output.shape[1] - prompt_len)
    text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    return {
        "text": text,
        "new_tokens": new_tokens,
        "seconds": round(elapsed, 3),
        "tokens_per_sec": round(new_tokens / elapsed, 2),
    }


# --------------------------------------------------------------------------- #
# Benchmark one precision mode
# --------------------------------------------------------------------------- #
def benchmark_mode(mode: str, prompts: list[dict], tokenizer) -> dict:
    print(f"\n{'=' * 60}\n[{mode}] loading {MODEL_ID} ...")
    free_gpu()

    t0 = time.perf_counter()
    model = load_model(mode)
    load_seconds = time.perf_counter() - t0
    weights_gb = torch.cuda.memory_allocated() / GB
    print(f"[{mode}] loaded in {load_seconds:.1f}s | weights VRAM: {weights_gb:.2f} GB")

    # Warmup (excluded from all measurements)
    generate(model, tokenizer, WARMUP_PROMPT)
    torch.cuda.reset_peak_memory_stats()

    runs, total_tokens, total_seconds = [], 0, 0.0
    for p in prompts:
        r = generate(model, tokenizer, p["prompt"])
        runs.append({"id": p["id"], "category": p["category"], **r})
        total_tokens += r["new_tokens"]
        total_seconds += r["seconds"]
        print(f"[{mode}] {p['id']:<14} {r['new_tokens']:>4} tok  {r['tokens_per_sec']:>7.2f} tok/s")

    peak_gb = torch.cuda.max_memory_allocated() / GB
    result = {
        "mode": mode,
        "model_id": MODEL_ID,
        "load_seconds": round(load_seconds, 1),
        "weights_vram_gb": round(weights_gb, 2),
        "peak_vram_gb": round(peak_gb, 2),
        "avg_tokens_per_sec": round(total_tokens / total_seconds, 2),
        "total_new_tokens": total_tokens,
        "runs": runs,
    }

    del model
    free_gpu()
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def summary_table(fp16: dict, nf4: dict) -> str:
    mem_saving = 100 * (1 - nf4["weights_vram_gb"] / fp16["weights_vram_gb"])
    speed_ratio = nf4["avg_tokens_per_sec"] / fp16["avg_tokens_per_sec"]
    return (
        f"## Quantization trade-off summary — {MODEL_ID} (T4, greedy, "
        f"{MAX_NEW_TOKENS} max new tokens)\n\n"
        "| Metric | fp16 (baseline) | NF4 4-bit (bitsandbytes) | Delta |\n"
        "| --- | --- | --- | --- |\n"
        f"| Weights VRAM | {fp16['weights_vram_gb']:.2f} GB | {nf4['weights_vram_gb']:.2f} GB "
        f"| -{mem_saving:.0f}% |\n"
        f"| Peak VRAM (generation) | {fp16['peak_vram_gb']:.2f} GB | {nf4['peak_vram_gb']:.2f} GB | — |\n"
        f"| Avg throughput | {fp16['avg_tokens_per_sec']:.2f} tok/s | {nf4['avg_tokens_per_sec']:.2f} tok/s "
        f"| x{speed_ratio:.2f} |\n"
        f"| Load time | {fp16['load_seconds']}s | {nf4['load_seconds']}s | — |\n"
        "| Output quality | reference | see results/outputs.md (per-prompt comparison) | — |\n"
    )


def outputs_md(fp16: dict, nf4: dict, prompts: list[dict]) -> str:
    lines = ["# Qualitative comparison — fp16 vs NF4 (same fixed prompts, greedy decoding)\n"]
    fp_runs = {r["id"]: r for r in fp16["runs"]}
    q_runs = {r["id"]: r for r in nf4["runs"]}
    for p in prompts:
        pid = p["id"]
        lines += [
            f"\n## [{pid}] ({p['category']})\n",
            f"**Prompt:** {p['prompt']}\n",
            f"\n### fp16 ({fp_runs[pid]['tokens_per_sec']} tok/s)\n\n{fp_runs[pid]['text']}\n",
            f"\n### NF4 ({q_runs[pid]['tokens_per_sec']} tok/s)\n\n{q_runs[pid]['text']}\n",
            "\n---\n",
        ]
    return "".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--prompts", default="prompts.json", help="path to prompts file")
    args = parser.parse_args()

    require_cuda()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Sequential runs on a clean GPU state; each result saved immediately so a
    # Colab disconnect after the first model doesn't lose everything.
    results = {}
    for mode in ("fp16", "nf4"):
        results[mode] = benchmark_mode(mode, prompts, tokenizer)
        (out_dir / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    table = summary_table(results["fp16"], results["nf4"])
    (out_dir / "summary.md").write_text(table, encoding="utf-8")
    (out_dir / "outputs.md").write_text(
        outputs_md(results["fp16"], results["nf4"], prompts), encoding="utf-8"
    )

    print("\n" + table)
    print(f"Done. Full results in: {out_dir}/")


if __name__ == "__main__":
    main()
