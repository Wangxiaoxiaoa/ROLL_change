import json
import os
from transformers import AutoProcessor
from tqdm import tqdm

model_path = "/mnt/data1/outputs/qwen3.5-9B-gui-lora-sft-llamafactory-remote1-8gpu-sdpa-full-len65536-cot/models"
dataset_path = "/mnt/data0/xiao/RL/ROLL/examples/gui_lora_rlvr/data/rl_compressed_v1_messages_prompt.jsonl"

print(f"Loading processor from {model_path}...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

count_gt_30k = 0
total_count = 0
max_tokens = 0
image_token_estimate = 1296 # For max_pixels 1048576 (1024x1024)

print(f"Processing dataset {dataset_path}...")
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=142163):
        try:
            data = json.loads(line)
            messages = data.get("messages", [])
            
            # Count images
            num_images = 0
            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and (item.get("type") == "image" or item.get("type") == "image_url"):
                            num_images += 1
            
            # Apply chat template to get text tokens
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text_tokens = len(processor.tokenizer(text)["input_ids"])
            
            # Calculate total estimate
            # Each image expands into image_token_estimate tokens.
            # In apply_chat_template, image tags are already present but they only count as a few tokens.
            # We add the expansion.
            total_estimate = text_tokens + num_images * (image_token_estimate - 1)
            
            if total_estimate > 30000:
                count_gt_30k += 1
            
            if total_estimate > max_tokens:
                max_tokens = total_estimate
            
            total_count += 1
            
        except Exception as e:
            continue

print(f"\nResults (using estimate of {image_token_estimate} tokens per image):")
print(f"Total samples processed: {total_count}")
print(f"Samples with length > 30,000: {count_gt_30k}")
print(f"Percentage: {(count_gt_30k / total_count * 100):.2f}%" if total_count > 0 else "N/A")
print(f"Max length estimate: {max_tokens}")
