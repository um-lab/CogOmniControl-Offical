# CogOmniControl

CogOmniControl is a multimodal control model built on top of Wan2.2, implemented by combining CogVLM and CogOmniDiT.

## Environment Setup

The code is implemented with **PyTorch 2.8 + CUDA 12.6** on a Linux machine with 32 H20 GPUs.

```bash
# 1. Create an environment (conda or venv)
conda create -n CogOmniControl python=3.11 -y
conda activate CogOmniControl

# 2. Install the dependencies
bash init_env.sh
```

## Training Weights

Currently supports Stage-3 Joint Training.

| Component | Model | Interface |
| --- | --- | --- |
| Base Wan2.2 DiT | [`Wan-AI/Wan2.2-T2V-A14B`](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | `MODEL_NAME` |
| Base VLM | `Qwen3-VL-8B-Thinking` | `LLM_NAME` |
| CogVLM LoRA | [`yang1232009/CogVLM`](https://huggingface.co/yang1232009/CogVLM) | `LLM_LORA_NAME` |
| CogOmniDiT LoRA | [`yang1232009/CogOmniControl`](https://huggingface.co/yang1232009/CogOmniControl) | `LORA_PATH` `CONNECTOR_PATH` |

Alternatively, you can download the weights with the following command:

```bash
bash scripts/download_weights.sh
```

### Notes

1. The connector and LoRA can point to the same checkpoint file; the corresponding weight parts are read automatically. This is because Stage-2 training saves them separately, while Stage-3 saves them together.

## CogVLM

CogVLM is a LoRA model trained on top of Qwen-3-VL-8B-Thinking.

Performance:

| Model | Multimodal Intent | Physical Plausibility | Information Completeness | Dynamic Description | Average |
| :-----| :----------:| :----------:| :----------:| :--------:| :-----:|
| Qwen-3-VL-8B-Instruct | 2.480 | 4.045 | 3.905 | 4.420 | 3.712 |
| Qwen-3-VL-8B-Thinking | 2.670 | 3.824 | 3.829 | 4.727 | 3.752 |
| [`CogVLM`](https://huggingface.co/yang1232009/CogVLM/) | **4.302** | **4.658** | **4.623** | **4.814** | **4.599** |

## Training Data

The training data uses the **JSONL** format (one JSON object per line). See the example file
`data/data_format.jsonl`. Each sample in the file describes a paired
"control video → target video" data item, used for Stage-3 Joint Training.

### Single Sample Example

```json
{
  "path": "/path/to/output/video.mp4",
  "caption": [
    {
      "content": "In a futuristic dimly lit corridor, the camera remains static, shooting in a medium shot, focusing on the two characters."
    }
  ],
  "dense_caption": [
    {
      "content": "In a futuristic dimly lit corridor, two men stand side by side..."
    }
  ],
  "control_path": "/path/to/input/video.mp4",
  "type": "video",
  "ref_img_infos": [
    {
      "ref_imgs": [
        {
          "img_path": "/path/to/ref_imgs/ref-img-0.png"
        },
        {
          "img_path": "/path/to/ref_imgs/ref-img-1.png"
        }
      ]
    }
  ]
}
```

Here `control_path` points to the control video (e.g., a storyboard sketch), and `path` points to the corresponding ground-truth target video.
`caption` and `dense_caption` can coexist; the `caption_mode` training argument configures which text description takes priority
(see the `--caption_mode` description in the training script, which supports `dense_caption`, `caption`, or weighted mixed modes).

## CogControlBench

[`CogControlBench`](https://huggingface.co/datasets/yang1232009/CogControlBench) is a dataset for evaluating professional film and video production. It covers storyboards and white models (grey-box renders) from real studio productions, white models from world rendering competitions, and part of the data comes from other benchmarks.
It contains 200 samples in total. To stay close to real studio production, evaluation is generally performed at 720P resolution.

## Training Code

Launch script for the high-noise model:

```bash
bash /exps/260307_CogOmniControl_train/train_high_cogomni_icl_vlm_connector_joint_viusal_cot_prompting.sh
```

Launch script for the low-noise model:

```bash
bash /exps/260307_CogOmniControl_train/train_low_cogomni_icl_vlm_connector_joint_viusal_cot_prompting.sh
```

Python file: `/scripts/wan2.2_cogomni_control/train_icl_vlm_connector_joint_viusal_cot_prompting_model_fast.py`

## Inference

```bash
bash /exps/CogOmniControl_eval/scripts/eval_cogomni_control.sh
```

Make sure to modify:
`/exps/CogOmniControl_eval/configs/flow_len81_cogomni_connector_720p_infer.yaml`

## Harness for Best-of-N

The CogVLM-v1 version not only infers multimodal intent, but is also equipped with a harness that generates dynamic evaluator selections for each video (for example, face IDs require invoking ID consistency, content involving non-cartoon/non-anime requires invoking aesthetics, etc.) to decide the Best-of-N strategy.
We designed 11 VLM-based evaluators and 3 general model-based evaluators (aesthetics, flicker/jitter, motion).

Available VLM evaluator IDs:

| English ID | Chinese Name |
| --- | --- |
| `prompt_adherence` | Text Adherence Verifier |
| `character_consistency` | Subject ID Preservation Verifier |
| `image_reference_visual_consistency` | Reference Image Visual Feature Verifier |
| `image_reference_pixel_consistency` | Reference Image Pixel Alignment Verifier |
| `control_video_following` | Control Video Following Verifier |
| `physical_effects` | Physical Dynamic Effects Verifier |
| `cross_modal_causality` | Multimodal Implicit Causality Verifier |
| `temporal_spatial_smoothness` | Temporal-Spatial Smoothness Verifier |
| `interaction_logic` | Interaction Logic Verifier |
| `artifact_detection` | Negative Artifact Detection Verifier |
| `storyboard_annotation_following` | Storyboard Annotation Following Verifier |

The VLM-based evaluators use Qwen3-VL-30B-A3B-Thinking-FP8 as the VLM, combined with the designed prompts for evaluation.
See [`HARNESS.md`](./qwen3vl_harness/HARNESS.md) for details.
