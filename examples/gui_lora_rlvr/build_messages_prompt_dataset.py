#!/usr/bin/env python3
"""Build a ROLL VLM JSONL view that feeds production messages as prompt."""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jpeg-dir", default=None)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def jpeg_path_for(src_path: str, jpeg_dir: Path) -> str:
    src = Path(src_path)
    name = src.name
    if name.endswith("_c75.png"):
        name = name[:-8] + "_jpeg75.jpg"
    else:
        name = src.stem + "_jpeg75.jpg"
    return str(jpeg_dir / name)


def convert_one(task):
    src_path, dst_path, quality = task
    dst = Path(dst_path)
    if dst.exists() and dst.stat().st_size > 0:
        return dst_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src_path) as image:
            image = image.convert("RGB")
            image.save(dst, format="JPEG", quality=quality)
    except Exception as exc:
        raise RuntimeError(f"failed to convert {src_path} -> {dst_path}: {exc}") from exc
    return dst_path


def rewrite_message_image_urls(value, path_map):
    if isinstance(value, list):
        return [rewrite_message_image_urls(item, path_map) for item in value]
    if isinstance(value, dict):
        rewritten = {key: rewrite_message_image_urls(val, path_map) for key, val in value.items()}
        if rewritten.get("type") == "image_url":
            image_url = rewritten.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if url in path_map:
                    rewritten["image_url"] = {**image_url, "url": path_map[url]}
        return rewritten
    return value


def main():
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    jpeg_dir = Path(args.jpeg_dir) if args.jpeg_dir else None
    path_map = {}
    if jpeg_dir:
        image_paths = []
        seen = set()
        with src.open("r", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, start=1):
                if args.limit and line_no > args.limit:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                for path in row.get("images") or []:
                    if path not in seen:
                        seen.add(path)
                        image_paths.append(path)
        tasks = []
        for image_path in image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"missing compressed image: {image_path}")
            jpeg_path = jpeg_path_for(image_path, jpeg_dir)
            path_map[image_path] = jpeg_path
            if not (os.path.exists(jpeg_path) and os.path.getsize(jpeg_path) > 0):
                tasks.append((image_path, jpeg_path, args.jpeg_quality))

        print(f"converting {len(tasks)} missing images to JPEG quality={args.jpeg_quality} in {jpeg_dir}")
        if tasks:
            done = 0
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(convert_one, task) for task in tasks]
                for future in as_completed(futures):
                    future.result()
                    done += 1
                    if done % 1000 == 0 or done == len(futures):
                        print(f"converted {done}/{len(futures)}")

    written = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if args.limit and written >= args.limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            images = row.get("images") or []
            if not isinstance(messages, list):
                raise ValueError(f"line {line_no}: messages must be a list")
            if not images or not all(isinstance(path, str) for path in images):
                raise ValueError(f"line {line_no}: images must be a non-empty string list")
            if jpeg_dir:
                images = [path_map[path] for path in images]
                messages = rewrite_message_image_urls(messages, path_map)
            if any(("images_compressed_75" not in path and "images_jpeg75" not in path) for path in images):
                raise ValueError(f"line {line_no}: image path is not from compressed image dir: {images}")
            missing = [path for path in images if not os.path.exists(path)]
            if missing:
                raise FileNotFoundError(f"line {line_no}: missing compressed image(s): {missing[:2]}")

            row["images"] = images
            row["messages"] = messages
            row["prompt"] = messages
            row["reward_model"] = {
                "ground_truth": row.get("ground_true")
                or (row.get("reward_model") or {}).get("ground_truth")
                or ""
            }
            row["data_source"] = row.get("data_source") or "gui_lora_rlvr"
            fout.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1

    print(f"wrote {written} rows to {dst}")


if __name__ == "__main__":
    main()
