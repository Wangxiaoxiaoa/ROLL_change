#!/usr/bin/env python3
import json
import statistics
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from roll.pipeline.rlvr.rlvr_vlm_pipeline import format_prompt


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "examples/gui_lora_rlvr/data/rl_compressed_v1_messages_prompt.jsonl"
MODEL = Path("/mnt/data1/outputs/qwen3.5-9B-gui-lora-sft-llamafactory-remote1-8gpu-sdpa-full-len65536-cot/models")
OUT_JSON = ROOT / "examples/gui_lora_rlvr/data/prompt_length_stats_actual.json"
OUT_MD = ROOT / "examples/gui_lora_rlvr/PROMPT_LENGTH_STATS.md"


def percentile(sorted_values, q):
    if not sorted_values:
        return None
    idx = round((len(sorted_values) - 1) * q)
    return sorted_values[idx]


def main():
    processor = AutoProcessor.from_pretrained(str(MODEL), trust_remote_code=True)
    image_processor = processor.image_processor
    patch_size = getattr(image_processor, "patch_size", 16)
    merge_size = getattr(image_processor, "merge_size", 2)
    merge_length = merge_size * merge_size
    factor = patch_size * merge_size if "Qwen" in image_processor.image_processor_type else 28
    min_pixels = getattr(image_processor, "min_pixels", 3136)
    max_pixels = getattr(image_processor, "max_pixels", 1048576)
    lengths = []
    top = []
    counts = {4096: 0, 6144: 0, 8192: 0, 12288: 0, 16384: 0, 32768: 0}
    image_counts = {}
    image_token_cache = {}

    with DATASET.open("r", encoding="utf-8") as fin:
        for idx, line in enumerate(tqdm(fin, total=142163, desc="prompt lengths")):
            row = json.loads(line)
            prompt = row.get("prompt", row.get("messages"))
            if not isinstance(prompt, str):
                prompt = format_prompt(prompt, processor, use_image=True, prompt_image_token="<image>")

            images = row.get("images") or []
            if not isinstance(images, list):
                images = [images]
            image_counts[len(images)] = image_counts.get(len(images), 0) + 1

            text_len = len(processor.tokenizer(prompt)["input_ids"])
            image_extra_tokens = 0
            processed_sizes = []
            for image in images:
                cached = image_token_cache.get(image)
                if cached is None:
                    with Image.open(image) as img:
                        height, width = img.height, img.width
                    resized_height, resized_width = smart_resize(
                        height,
                        width,
                        factor=factor,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                    )
                    grid_h = resized_height // patch_size
                    grid_w = resized_width // patch_size
                    image_tokens = grid_h * grid_w // merge_length
                    cached = (image_tokens, [resized_width, resized_height])
                    image_token_cache[image] = cached
                image_tokens, processed_size = cached
                processed_sizes.append(processed_size)
                image_extra_tokens += image_tokens - 1

            length = text_len + image_extra_tokens
            lengths.append(length)
            for limit in counts:
                if length > limit:
                    counts[limit] += 1
            if len(top) < 20 or length > top[-1]["length"]:
                top.append(
                    {
                        "index": idx,
                        "length": length,
                        "image_count": len(images),
                        "images": images[:5],
                        "processed_sizes": processed_sizes,
                        "instruction": row.get("instruct", "")[:300],
                    }
                )
                top.sort(key=lambda x: x["length"], reverse=True)
                top = top[:20]

    sorted_lengths = sorted(lengths)
    stats = {
        "dataset": str(DATASET),
        "model": str(MODEL),
        "count": len(lengths),
        "min": sorted_lengths[0],
        "max": sorted_lengths[-1],
        "mean": statistics.mean(lengths),
        "p50": percentile(sorted_lengths, 0.50),
        "p90": percentile(sorted_lengths, 0.90),
        "p95": percentile(sorted_lengths, 0.95),
        "p99": percentile(sorted_lengths, 0.99),
        "p999": percentile(sorted_lengths, 0.999),
        "count_gt": counts,
        "image_counts": image_counts,
        "unique_images": len(image_token_cache),
        "patch_size": patch_size,
        "merge_size": merge_size,
        "resize_factor": factor,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "top": top,
    }
    OUT_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# GUI RL Prompt Length Stats",
        "",
        f"- Dataset: `{DATASET}`",
        f"- Model processor: `{MODEL}`",
        f"- Count: {stats['count']}",
        f"- Min / P50 / P90 / P95 / P99 / P99.9 / Max: {stats['min']} / {stats['p50']} / {stats['p90']} / {stats['p95']} / {stats['p99']} / {stats['p999']} / {stats['max']}",
        f"- Mean: {stats['mean']:.2f}",
        f"- Patch size / merge size: {patch_size} / {merge_size}",
        f"- Resize factor / min pixels / max pixels: {factor} / {min_pixels} / {max_pixels}",
        "",
        "## Image Count Distribution",
        "",
    ]
    for image_count in sorted(image_counts):
        count = image_counts[image_count]
        pct = count / len(lengths) * 100
        lines.append(f"- `{image_count}` images: {count} ({pct:.4f}%)")
    lines += [
        "",
        "## Counts Above Limits",
        "",
    ]
    for limit, count in counts.items():
        pct = count / len(lengths) * 100
        lines.append(f"- `>{limit}`: {count} ({pct:.4f}%)")
    lines += ["", "## Longest Samples", ""]
    for item in top:
        lines.append(
            f"- index `{item['index']}`, length `{item['length']}`, "
            f"images `{item['image_count']}`, first image `{item['images'][0] if item['images'] else ''}`"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
