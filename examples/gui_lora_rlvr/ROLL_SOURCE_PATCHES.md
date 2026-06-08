# ROLL Source Patches For GUI LoRA RLVR

Date: 2026-05-29

Base revision:

```text
51c123f11800e4be603e64dc67520feb17037f50
```

The training environment and scripts live under `examples/gui_lora_rlvr`. The following source patches were applied only after explicit approval because the clean `origin/main` smoke run failed before rollout training.

## Applied From PR #438

Reference: <https://github.com/alibaba/ROLL/pull/438>

### `roll/third_party/deepspeed/model_update.py`

Reason: GUI RL uses LoRA training with vLLM inference. Clean `origin/main` tried to broadcast full PEFT-wrapped parameter names such as `base_model...` into vLLM and failed at step 0:

```text
ValueError: There is no module or parameter named 'base_model' in Qwen3_5ForConditionalGeneration
```

Patch:

- In LoRA mode, gather only adapter parameters instead of all model parameters.
- Pass `is_lora=True` through colocated and separated weight update paths.
- After adapter tensors are transferred, call infer workers' `add_lora` with the PEFT config.

### `roll/third_party/vllm/worker.py`

Reason: Keep vLLM LoRA tensor injection path aligned with PR #438 for vLLM 0.17.0. The generic `process_weights_after_loading` call is skipped for `>=0.11.1`, matching the PR branch.

### `roll/utils/deepspeed_utils.py`

Reason: DeepSpeed can fail when an optimizer group is empty after LoRA filtering. The patch drops empty parameter groups.

### `roll/distributed/scheduler/initialize.py`

Reason: Ray failed with Unix socket paths longer than 107 bytes when using the deep example directory as `TMPDIR`. The patch supports `ROLL_RAY_TEMP_DIR` and passes it to `ray start --temp-dir`.

### `roll/pipeline/rlvr/rlvr_vlm_pipeline.py`

Reason: This run has a single domain (`gui_lora_rlvr`). Clean `origin/main` still maps and filters the whole multimodal dataset, which is slow and can trigger Arrow offset limits on large image columns. The patch directly adds `domain` for single-domain runs and keeps the old multi-domain behavior.

Additional local fix: `process_image` now falls back to `min_pixels=3136` and `max_pixels=1048576` if a pickled Qwen image processor in a dataset map worker does not retain those dynamic attributes. These values are the same as the GUI training config.

### `roll/configs/data_args.py`

Reason: `roll/pipeline/rlvr/rlvr_vlm_pipeline.py` already has a `cache_path` load/save branch, but `DataArguments` did not declare the field. The GUI dataset has 142163 compressed screenshot samples; without a declared `cache_path`, the formal run would re-encode every image on each launch. The patch adds `cache_path: Optional[str]` so the existing cache branch can be used by config.

## Example-Level Training Files

### `examples/gui_lora_rlvr/preencode_dataset_cache.py`

Reason: Precompute the same ROLL VLM encoded dataset outside the Ray training process. This keeps the model input path identical to training (`get_vlm_dataset` + `encode_function` + Qwen processor image resize), but lets preprocessing use multiple workers before Ray starts.

### `examples/gui_lora_rlvr/build_messages_prompt_dataset.py`

Reason: Rebuild the GUI RL JSONL from `rl_compressed_v1.jsonl` while preserving the production `messages` structure. The current production node `10.83.115.20` resizes screenshots and serializes them as RGB JPEG with `quality=75`; therefore the builder now converts the referenced resized screenshots into true JPEG files under `/mnt/data1/datas/rl_train_datas_cot10_first_half/images_jpeg75` and rewrites both `images` and `messages[*].content[*].image_url.url` to those JPEG paths. This avoids training on PNG files when production sends JPEG bytes. The builder enables PIL truncated-image loading because `traj_000634_step005_c75.png` is a truncated PNG but can still be decoded to the expected resized frame and then serialized as JPEG75.

### `examples/gui_lora_rlvr/roll_gui_lora_rlvr_9b_messages_c75*.yaml`

Reason: The formal config now points at `examples/gui_lora_rlvr/data/cache/rl_compressed_v1_messages_prompt_jpeg75`. The smoke config points at its own small JPEG75 cache and was used only to verify that rollout, reward, backward, and optimizer update can complete.

Formal length policy:

- `prompt_length: 32768`
- `response_length: 2048`
- `sequence_length: 34816`

These values intentionally keep the prompt budget above 30K and the generation budget at or above 2048, matching the required production-training budget. The earlier prompt length statistics file is retained only as a diagnostic estimate from a Qwen image-grid formula; it must not be used to lower the formal prompt budget below 30K.

## Not Applied Yet From PR #437

Reference: <https://github.com/alibaba/ROLL/pull/437>

PR #437 adds Qwen2/3-VL `mm_token_type_ids` and 3D RoPE alignment handling. It was not applied in this pass because the first hard failure was LoRA sync before generation/training. If the next smoke run reaches a Qwen3-VL position-id or `mm_token_type_ids` error, apply the minimal #437 collator/model-provider patch next.

## Verification Log

- Clean `origin/main` smoke data encoding passed.
- Clean `origin/main` failed at `pipeline step 0` during LoRA weight sync into vLLM.
- After applying the patch, first re-smoke failed before training because `ROLL_RAY_TEMP_DIR` was still too deep. The launcher now sets `ROLL_RAY_TEMP_DIR=/mnt/data0/raytmp_roll` by default.
- Re-smoke `smoke_p438_shorttmp_20260529_093826` passed data encoding, worker initialization, LoRA sync, and 8 rollout generations. It did not perform an optimizer update because the temporary `response_length: 128` truncated all generations before a final action/tool call, producing `final_response_mask.sum() == 0`.
- The smoke config was changed to `response_length: 1024` to verify a real training update path before the formal run.
- Smoke cache pre-encoding with `num_proc=4` reached `Encoding dataset` but exposed a Qwen processor pickle issue: `Qwen2VLImageProcessor` in worker processes lacked `min_pixels/max_pixels`. The `process_image` fallback patch above addresses this while preserving the configured resize limits.
- Smoke run `smoke_rank64_len4608_nogc_g7_20260529_142848` completed one full RL step on 2026-05-29. All 7 actor train ranks reached `train global step 0: 100%|1/1`, the driver logged `pipeline step 0 finished`, and the run ended with `pipeline complete!`.
- The formal run should use the full dataset, LoRA rank 64, 7 rollout samples per group for the 7 train GPUs, gradient checkpointing disabled for the ZeRO-3 checkpoint mismatch observed in smoke, and the formal 32768/2048/34816 length budget above.
