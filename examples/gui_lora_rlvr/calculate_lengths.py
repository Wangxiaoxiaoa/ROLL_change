import json
import os
from transformers import AutoProcessor
from tqdm import tqdm
import torch

model_path = "/mnt/data1/outputs/qwen3.5-9B-gui-lora-sft-llamafactory-remote1-8gpu-sdpa-full-len65536-cot/models"
dataset_path = "/mnt/data0/xiao/RL/ROLL/examples/gui_lora_rlvr/data/rl_compressed_v1_messages_prompt.jsonl"

print(f"Loading processor from {model_path}...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

count_gt_30k = 0
total_count = 0
max_len = 0

print(f"Processing dataset {dataset_path}...")
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=142163):
        try:
            data = json.loads(line)
            messages = data.get("messages", [])
            
            # Apply chat template to get the prompt string
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Tokenize to get the actual length
            # Note: For VLM, we should ideally use processor(text=text, images=...) but 
            # if we just want a rough count or if the images are already represented by tags
            # apply_chat_template with tokenize=True is better.
            
            # For Qwen2-VL, the processor expands <|image_pad|> based on vision_tokens.
            # We use the processor's tokenizer logic.
            inputs = processor.tokenizer(text)
            length = len(inputs["input_ids"])
            
            if length > 30000:
                count_gt_30k += 1
            
            if length > max_len:
                max_len = length
            
            total_count += 1
            
        except Exception as e:
            print(f"Error processing line: {e}")
            continue

print(f"\nResults:")
print(f"Total samples processed: {total_count}")
print(f"Samples with length > 30,000: {count_gt_30k}")
print(f"Percentage: {(count_gt_30k / total_count * 100):.2f}%" if total_count > 0 else "N/A")
print(f"Max length found: {max_len}")
