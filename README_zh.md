# CogOmniControl

CogOmniControl是一个基于Wan2.2的多模态控制模型，结合CogVLM以及CogOmniDiT实现


## 环境设置

代码实现在 **PyTorch 2.8 + CUDA 12.6**, 32张H20的Linux机器

```bash
# 1. Create an environment (conda or venv)
conda create -n CogOmniControl python=3.11 -y
conda activate CogOmniControl

# 2. Install the dependencies
bash init_env.sh
```

## 训练权重

目前支持Stage-3 Joint Training

| 部分　　　　　　　　　　　　　　　　 | 模型      | 接口　　　　　　 |
| --------------------------------------| ----------------------------------------| ------------------|
| Base Wan2.2 DiT　　　| [`Wan-AI/Wan2.2-T2V-A14B`](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)　　　| `MODEL_NAME`　　 |
| Base VLM | `Qwen3-VL-8B-Thinking`　| `LLM_NAME`　　　 |
| CogVLM LoRA　| [`yang1232009/CogVLM`](`https://huggingface.co/yang1232009/CogVLM`)　　　 | `LLM_LORA_NAME`　|
| CogOmniDiT LoRA | [`yang1232009/CogOmniControl`](https://huggingface.co/yang1232009/CogOmniControl) | `LORA_PATH` `CONNECTOR_PATH`|
| CogOmniDiT LoRA | [`yang1232009/CogOmniControl`](https://huggingface.co/yang1232009/CogOmniControl) | `LORA_PATH` `CONNECTOR_PATH`|


或者使用以下命令下载权重

```bash
bash scripts/download_weights.sh
```

### 注意
1. Connector和LoRA可以指向同一份权重文件，会自动读取对应的权重部分（这是因为Stage-2训练时是分开保存，而Stage-3是一块保存的）

## CogVLM
CogVLM是基于Qwen-3-VL-8B-Thinking训练的LoRA模型。

性能表现

| 模型 | 多模态意图 | 物理合理性 | 信息完整性 | 动态描述 | 平均　|
| :-----| :----------:| :----------:| :----------:| :--------:| :-----:|
| Qwen-3-VL-8B-Instruct | 2.480　　　| 4.045　　　| 3.905　　　| 4.420　　| 3.712 |
| Qwen-3-VL-8B-Thinking | 2.670　　　| 3.824　　　| 3.829　　　| 4.727　　| 3.752 |
| [`CogVLM`](https://huggingface.co/yang1232009/CogVLM/) | **4.302**　　　| **4.658**　　　| **4.623**　　　| **4.814**　　| **4.599** |



## 训练数据

训练数据使用 **JSONL** 格式（每行一个 JSON 对象），参考示例文件
`data/data_format.jsonl`。文件中的每一条样本描述了一段
"控制视频 → 目标视频" 的配对数据，用于 Stage-3 Joint Training。
### 单条样本示例

```json
{
  "path": "/path/to/output/video.mp4",
  "caption": [
    {
      "content": "在一个未来主义的昏暗走廊中，摄像机保持静止，以中景镜头拍摄，聚焦在这两个角色身上。",
    }
  ],
  "dense_caption": [
    {
      "content": "在一个未来主义的昏暗走廊中，两名男子并肩而立……"
    }
  ],
  "control_path": "/path/to/input/video.mp4",
  "type": "video",
  "ref_img_infos": [
    {
      "ref_imgs": [
        {
          "img_path": "/path/to/ref_imgs/ref-img-0.png",
        },
        {
          "img_path": "/path/to/ref_imgs/ref-img-1.png",
        },
      ]
    }
  ]
}
```

其中 `control_path` 指向的是控制视频（例如分镜草图 sketch），`path` 指向的是对应的真实目标视频。
`caption` 与 `dense_caption` 可同时存在，`caption_mode` 训练参数可配置优先使用哪一版文本描述
（详见训练脚本中的 `--caption_mode` 说明，支持 `dense_caption`、`caption` 或带权重的混合模式）。



## CogControlBench
[`CogControlBench`](https://huggingface.co/datasets/yang1232009/CogControlBench)是一个用于评估专业影视制作的数据集。涵盖真实的工作室影视制作Storyboard/白模，世界渲染大赛白模，部分来源于其他benchmark的数据。
共有200条样本，为贴近真实工作室制作，一般在720P的分辨率下进行评估。

## 训练代码

高噪模型启动代码
```bash
bash /exps/260307_CogOmniControl_train/train_high_cogomni_icl_vlm_connector_joint_viusal_cot_prompting.sh
```

低噪模型启动代码
```bash
bash /exps/260307_CogOmniControl_train/train_low_cogomni_icl_vlm_connector_joint_viusal_cot_prompting.sh
```

Python文件：/scripts/wan2.2_cogomni_control/train_icl_vlm_connector_joint_viusal_cot_prompting_model_fast.py

## 推理

```bash
bash /exps/CogOmniControl_eval/scripts/eval_cogomni_control.sh
```

注意修改
/exps/CogOmniControl_eval/configs/flow_len81_cogomni_connector_720p_infer.yaml


## Harness for Best-of-N
CogVLM-v1版本不仅能推理多模态意图，还配置了一套harness，生成该条视频的动态评估器选择（例如人脸ID会要求调用ID一致性、涉及非卡通/动漫会要求调用美学等等）来抉择Best-of-N策略。
我们设计了11个基于VLM的评估器和3个基于一般模型的评估器（美学、跳变、运动）

可用VLM评估器 ID：

| 英文 ID　　　　　　　　　　　　　　　| 中文名称　　　　　　　　 |
| --------------------------------------| --------------------------|
| `prompt_adherence`　　　　　　　　　 | 文本遵循验证器　　　　　 |
| `character_consistency`　　　　　　　| 主体ID保持验证器　　　　 |
| `image_reference_visual_consistency` | 参考图像视觉特征验证器　 |
| `image_reference_pixel_consistency`　| 参考图像像素对齐验证器　 |
| `control_video_following`　　　　　　| 控制视频遵循验证器　　　 |
| `physical_effects`　　　　　　　　　 | 物理动态特效验证器　　　 |
| `cross_modal_causality`　　　　　　　| 多模态隐含因果验证器　　 |
| `temporal_spatial_smoothness`　　　　| 时空平滑度验证器　　　　 |
| `interaction_logic`　　　　　　　　　| 交互逻辑性验证器　　　　 |
| `artifact_detection`　　　　　　　　 | 负面伪影检测验证器　　　 |
| `storyboard_annotation_following`　　| Storyboard标注遵循验证器 |

基于VLM的评估器是利用Qwen3-VL-30B-A3B-Thinking-FP8作为VLM，结合设计的Prompt进行评估。
具体参考[`HARNESS.md`](./qwen3vl_harness/HARNESS.md)



