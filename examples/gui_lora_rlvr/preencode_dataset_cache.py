#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


ROLL_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = ROLL_ROOT / "examples"
sys.path.insert(0, str(ROLL_ROOT))
sys.path.insert(0, str(EXAMPLES_ROOT))

from dacite import from_dict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from roll.models.model_providers import default_processor_provider
from roll.pipeline.rlvr.rlvr_vlm_pipeline import RLVRConfig, encode_function, get_vlm_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="gui_lora_rlvr")
    parser.add_argument("--config_name", default="roll_gui_lora_rlvr_9b_messages_c75")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cache_path", default=None)
    args = parser.parse_args()

    config_dir = Path(args.config_path)
    if not config_dir.is_absolute():
        config_dir = EXAMPLES_ROOT / config_dir
    with initialize_config_dir(config_dir=str(config_dir), job_name="preencode", version_base=None):
        cfg = compose(config_name=args.config_name)
    config = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    data_args = config.actor_train.data_args
    if args.cache_path:
        data_args.cache_path = args.cache_path
    if not data_args.cache_path:
        raise ValueError("actor_train.data_args.cache_path must be set")

    os.makedirs(data_args.cache_path, exist_ok=True)
    data_args.preprocessing_num_workers = args.workers

    processor = default_processor_provider(config.actor_train.model_args)
    dataset = get_vlm_dataset(data_args, encode_function, processor, get_eval=False)
    print(dataset)
    print(f"cache_path={os.path.join(data_args.cache_path, 'train')}")


if __name__ == "__main__":
    main()
