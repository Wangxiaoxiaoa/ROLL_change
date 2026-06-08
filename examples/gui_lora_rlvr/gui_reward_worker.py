import json
import os
from typing import Dict, List

import torch

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_processor_provider

from gui_lora_rlvr.reward import score_gui_response


class GuiLoraRlvrRewardWorker(Worker):
    def __init__(self, worker_config):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.processor = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.processor = default_processor_provider(self.worker_config.model_args)
        self.tokenizer = self.processor.tokenizer
        self.logger.info(f"{self.worker_name} initialized without reward model")

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        responses: List[str] = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        prompts: List[str] = self.tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
        ground_truths = data.non_tensor_batch["ground_truth"]

        scores = []
        logs = []
        for prompt, response, ground_truth in zip(prompts, responses, ground_truths):
            reward = score_gui_response(response=response, ground_truth=ground_truth)
            scores.append(reward["score"])
            logs.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "ground_truth": ground_truth,
                    "reward": reward,
                }
            )

        log_path = os.environ.get("GUI_ROLL_ROLLOUT_LOG")
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as fout:
                for item in logs:
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        score_tensor = torch.tensor(scores, dtype=torch.float32)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": score_tensor,
                "scores": score_tensor,
            }
        )
        return output
