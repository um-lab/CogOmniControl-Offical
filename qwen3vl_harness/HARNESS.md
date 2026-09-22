# CogOmniControl —— Qwen3-VL 视频生成评估与择优框架（独立版）

基于 **Qwen3-VL-30B-A3B-Thinking-FP8**（通过 vLLM 部署）的 11 个视频生成评估器，
用于对生成结果自动打分，并从 **N 个候选结果中挑出每条样本的最优者**。


## 1. 整体流程

1. Sample多次推理结果（N 个候选），通过
```bash
bash ./exps/CogOmniControl_eval/scripts/eval_cogomni_control.sh
```
获得各次sample的输出结果 pred_infos.jsonl

2. CogVLM推理获得评估器jsonl
```bash
bash ./qwen3vl_harness/CogVLM/cogomni_thinking.sh
```

3. 准备候选集，融合 pred_infos.jsonl 以及 tool.json（由 cogomni_thinking.sh + convert_tool_json.py 生成的思考/工具结果）
```bash
python ./qwen3vl_harness/fuse_jsonl_files.py --tool tool.json --input pred_infos.jsonl --output input/fused_results_sample1.jsonl
```

4. 评估打分：用 `qwen3vl_harness.py evaluate` 对每个候选 JSONL 跑评估器，输出带分数的 jsonl。

5. 挑最优：用 `qwen3vl_harness.py select-best` 按 `sample_id` 对齐 N 份打分，选出总分最高的候选。

---

## 2. 评估器（11 个，0–5 分制）

所有评估器 Prompt 见 [`evaluator.md`](./evaluator.md)，定义严格不变（由 `qwen3vl_harness.py` 逐字引用）。

| 英文 ID                                | 中文名称             | 何时使用                         |
|----------------------------------------|----------------------|----------------------------------|
| `prompt_adherence`                     | 文本遵循验证器       | 始终调用（基础）                 |
| `character_consistency`                | 主体ID保持验证器     | 参考图存在可识别主体             |
| `image_reference_visual_consistency`   | 参考图像视觉特征验证器 | 需外观/身份一致（不要求像素对齐）|
| `image_reference_pixel_consistency`    | 参考图像像素对齐验证器 | 需关键帧/构图像素级对齐          |
| `control_video_following`              | 控制视频遵循验证器   | 以控制视频为主信号               |
| `physical_effects`                     | 物理动态特效验证器   | 含火/水/烟/雾/光等特效           |
| `cross_modal_causality`                | 多模态隐含因果验证器 | 跨模态暗示的因果关系需"脑补"     |
| `temporal_spatial_smoothness`          | 时空平滑度验证器     | 始终调用（基础）                 |
| `interaction_logic`                    | 交互逻辑性验证器     | 存在物体交互/接触/移动           |
| `artifact_detection`                   | 负面伪影检测验证器   | 始终调用（基础）                 |
| `storyboard_annotation_following`      | Storyboard标注遵循验证器 | 控制视频/故事板带文字标注       |

特殊值：缺省（输入 JSONL 未写 `evaluators` 字段）时**默认运行全部 11 个评估器**。

### 4.1 单份候选集评估（打分）

```bash
CUDA_VISIBLE_DEVICES=0,1 python qwen3vl_harness.py evaluate \
    --input_jsonl  inputs/fused_results_sample1.json \
    --output_jsonl outputs/fused_results_sample1_harness_all_metrics.json \
    --model_path Qwen/Qwen3-VL-30B-A3B-Thinking-FP8 \
    --tensor_parallel_size 2
```

关键参数：

| 参数 | 说明 |
|------|------|
| `--model_path` | 模型路径，**生产用 `Qwen/Qwen3-VL-30B-A3B-Thinking-FP8`**；也可传本地路径 |
| `--model_variant` | `thinking` / `instruct`（仅当 `--model_path` 为空时用于解析默认 HF 名）|
| `--tensor_parallel_size` | 张量并行 GPU 数 |
| `--max_model_len` / `--gpu_memory_utilization` | vLLM 显存与上下文配置 |
| `--max_new_tokens` / `--video_max_frames` / `--video_fps` | 生成长度与视频抽帧配置 |
| `--default_evaluators` | 输入 JSONL 缺少 `evaluators` 字段时的默认评估器列表；缺省跑全部 11 个 |
| `--start_idx` / `--end_idx` | 评估样本范围（用于断点续跑）|

输入 JSONL 每条记录的可用字段（自动兼容）：

| 字段 | 说明 |
|------|------|
| `pred_lose_path` / `lose_path` / `pred_video` / `path` | 待评估生成视频（按优先级取）|
| `control_path` | 控制视频 |
| `ref_img_infos` | 参考图信息 `[{ref_imgs:[{img_path}]}]` |
| `dense_caption` | 文本 Prompt `[{content}]` |
| `evaluators` | 该条要跑的评估器列表（缺省用 `--default_evaluators`）|

输出 JSONL 每行结构：
```json
{
  "sample_id": "...",
  "evaluators": ["prompt_adherence", ...],
  "generated_video": "/path/to/pred.mp4",
  "results": {
    "prompt_adherence": {"success": true, "evaluator": "文本遵循验证器",
                          "score": 4.0, "findings": ["..."], "summary": "...",
                          "raw_response": "...", "elapsed_sec": 12.3}
  }
}
```

### 4.2 多 GPU 并行评估（4 份候选）

参考 `run_eval.sh`，每份候选占一组 GPU（详见脚本注释）。

### 4.3 从 N 份打分中挑最优

```bash
python qwen3vl_harness.py select-best \
    --inputs outputs/fused_results_sample1_harness_all_metrics.json \
             outputs/fused_results_sample2_harness_all_metrics.json \
             outputs/fused_results_sample3_harness_all_metrics.json \
             outputs/fused_results_sample4_harness_all_metrics.json \
    --output outputs/best_by_score.json \
    --rank-by total          # total=按总分；mean=按平均分
```

`best_by_score.json` 结构：
```json
{
  "meta": {"num_sources": 4, "num_samples": N, "rank_by": "total"},
  "source_summary": [{"source": "...", "win_count": 12, "avg_total_score": 41.2}],
  "samples": [
    {
      "sample_id": "...",
      "per_source": [{"source": "sample1", "total_score": 40.0, "mean_score": 4.44, "generated_video": "..."}],
      "best_index": 0,
      "best_source": "sample1",
      "best_total_score": 42.0,
      "best_generated_video": "/path/to/best.mp4"
    }
  ]
}
```

`source_summary.win_count` 统计每个候选来源胜出的样本数，便于横向比较 N 个候选集的整体质量。
