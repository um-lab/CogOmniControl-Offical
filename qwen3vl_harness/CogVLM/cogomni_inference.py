#!/usr/bin/env python3
import os
import sys
import re
import json
import time
import logging
import argparse
from typing import Optional, List, Dict, Any

import sys


os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


#!/usr/bin/env python3


import json
from typing import Dict, List, Any


def build_cogomni_prompt(item: Dict) -> str:
    """
    构建视频生成多模态理解与推理的完整prompt。
    
    输入格式：
        {
            "control_path": "path/to/control_video.mp4",
            "ref_img_infos": ["path/to/ref_image1.jpg", "path/to/ref_image2.jpg"],
            "prompt": "用户输入的文本提示"
        }
    
    返回：完整的推理prompt字符串
    """
    prompt_text = item.get("dense_caption", "")[0]["content"]
    control_path = item.get("control_path", None)
    ref_img_infos = item.get("ref_img_infos", [])
    
    # 构建用户输入部分
    user_input = f"文本描述：{prompt_text}\n\n"
    
    full_prompt = f"""# Role\n\n你是一位多模态视频生成条件协调专家，同时也是视频质量评估方案规划师。\n分析给定的素材（控制视频(带时间戳的图像)、参考图像(不带时间戳的图像)和文本描述）你需要：\n1. 输出一段连贯的推理协调方案，指导视频生成模型如何从这些条件生成最终视频\n2. 基于方案内容，从评估器库中选择合适的评估器\n\n# Inputs\n\n1. 控制视频，形式不固定（3D白模/线稿/深度图/骨架/特效预览/分镜storyboard等），提供的信息因场景而异。\n2. 静态参考图像，通常定义目标视频的视觉世界观。\n3. 文本描述，表达创作意图。\n\n控制视频和参考图像可能不会每次都提供，如果没有则忽略。\n\n\n# 生成方案Rules\n\n1. 输出必须是一段连贯的文字，不要列表、不要表格，不少于500字。\n2. 控制视频的作用不是预设的，根据实际内容判断。\n3. 参考图像通常定义视觉世界观。\n4. 所有决策自然嵌入叙述中并给出理由。\n5. 主动推理条件暗示但未明说的效果，将笼统描述展开为具体物理过程。\n6. 不同条件的信息要主动组合，推理组合后的新效果。\n7. 具体可执行，不说空话。\n8. 对条件中缺失或断裂的关键信息，必须主动强调补全。\n9. 当实体需要在画面中出现或消失时，推理出合理的进场和退场动作，保证叙事连贯，不允许实体凭空出现或消失。\n10. 最终方案应当是一段信息稠密、逻辑连贯的文字，读完后能清晰知道目标视频的每个元素应该如何呈现、如何运动、如何交互。\n\n文本描述：{prompt_text}\n\n# Evaluator Registry（评估器库）\n\n固定名称清单（**必须逐字精确匹配，严禁修改、缩写、翻译**）：\n\n- `文本遵循验证器` - 视频是否忠实遵循文本Prompt的核心内容。**当方案确定严格遵循文本时，或冲突情况确定遵循文本时调用**。\n- `主体ID保持验证器` - 视频中主体身份是否与参考图像一致。**当参考图像中存在可识别的角色/主体时调用**。\n- `参考图像视觉特征验证器` - 视频是否以图像为视觉参考基准。**当方案指定\"参考图像的视觉特征/形象\"时调用，注意并不是要求视频中的某一帧必须是该残稿图像。\n- `参考图像像素对齐验证器` - 视频是否严格遵循参考图像作为关键帧，保持像素级别的对齐。**当方案指定\"以参考图像为基础，应用控制视频暗示的动态信息\"时调用。与\"控制视频遵循验证器\"互斥**。\n- `控制视频遵循验证器` - 视频是否严格遵循控制视频的空间布局/姿态/深度等具体信号。**当方案指定\"以控制视频为基础，应用参考图像的视觉特征\"时调用。与\"参考图像像素对齐验证器\"互斥**。\n- `物理动态特效验证器` - 火焰/水流/烟雾/爆炸等特效是否动态合理。**当输入中存在物理特效元素时调用**。\n- `多模态隐含因果验证器` - 跨模态输入暗示的因果关系是否被\"脑补\"进视频。**当多模态输入之间存在需要推断的因果联动时调用**。\n- `时空平滑度验证器` - 视频是否存在闪烁、跳变、撕裂、卡顿（语义层面）。**始终调用**。\n- `交互逻辑性验证器` - 物体交互是否符合物理和日常逻辑。**当视频中存在物体交互、接触、移动场景时调用**。\n- `负面伪影检测器` - 多头、多肢、形变、漂浮等AI伪影检测。**始终调用**。\n- `Storyboard标注遵循验证器` - 视频是否遵循Storyboard上的文字标注指令。**当控制视频是Storyboard分镜、附带文字标注、且方案决定遵循时调用**。\n- `美学评分器` - 画面美学质量评分。**当内容属于艺术性/风景性/设计性题材时调用**（适合：风景/艺术/时尚/建筑/精致渲染；不适合：动漫/卡通/UGC/监控/游戏截图）。\n- `动态程度评估器` - 视频动态幅度（基于光流）。**当方案判断视频应有较大动态变化时调用**。\n- `运动平滑度评估器` - 运动轨迹的平滑性和连续性（基于光流）。**当视频中存在明显运动时调用**。\n\n# Output Format\n\n严格按照以下一行式扁平格式输出：\n\n<方案文本>\n[tools][评估器名称1][评估器名称2][评估器名称3]...\n"""
    
    
    return full_prompt



def _extract_ref_images(item: Dict) -> Optional[List[str]]:
    #   ref_img_infos: [{ref_imgs: [{img_path: "..."}]}]
    #   ref_img_infos: [{ref_imgs: ["...", "..."]}]
    #   ref_img_infos: [{img_path: "..."}]
    #   ref_img_infos: ["...", "...", "..."]
    #   ref_img_infos: [{"subject_id": 0, "subject_type": "frame", "ref_imgs": [{"frame_idx": -2, "img_path": "..."}]}]
    out: List[str] = []
    rii = item.get("ref_img_infos")
    if isinstance(rii, list):
        for info in rii:
            # 1) 直接是字符串路径
            if isinstance(info, str):
                if os.path.exists(info):
                    out.append(info)
                continue
            # 2) 是 dict，尝试从 ref_imgs / img_path 中取
            if isinstance(info, dict):
                imgs = info.get("ref_imgs")
                if isinstance(imgs, list):
                    for ri in imgs:
                        if isinstance(ri, dict):
                            p = ri.get("img_path")
                            if isinstance(p, str) and os.path.exists(p):
                                out.append(p)
                        elif isinstance(ri, str):
                            if os.path.exists(ri):
                                out.append(ri)
                elif isinstance(imgs, str):
                    if os.path.exists(imgs):
                        out.append(imgs)
                # 顺带兼容 dict 顶层直接带 img_path 的情况
                p = info.get("img_path")
                if isinstance(p, str) and os.path.exists(p):
                    out.append(p)

    if out:
        # 去重但保留顺序
        seen = set()
        uniq = []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq
    # 通用兜底
    for k in ("ref_images", "reference_image", "ref_image", "ref_img", "image"):
        v = item.get(k)
        if isinstance(v, str) and os.path.exists(v):
            return [v]
        if isinstance(v, list):
            collected = [p for p in v if isinstance(p, str) and os.path.exists(p)]
            if collected:
                return collected
    return None


def build_messages_with_custom_prompt(item: Dict) -> List[Dict[str, Any]]:
    """
    使用自定义prompt构建消息格式，同时保留多模态数据。
    """
    custom_prompt = build_cogomni_prompt(item)
    control_path = item.get("control_path", None)
    ref_img_infos = _extract_ref_images(item)
    
    content: List[Dict[str, Any]] = []
    
    # 1. 添加控制视频 content block
    if control_path and isinstance(control_path, str) and control_path.strip():
        content.append({"type": "video", "video": control_path})
    
    # 2. 添加参考图像 content blocks
    if ref_img_infos:
        if isinstance(ref_img_infos, list):
            for img_path in ref_img_infos:
                if isinstance(img_path, str) and img_path.strip():
                    content.append({"type": "image", "image": img_path})
        elif isinstance(ref_img_infos, str) and ref_img_infos.strip():
            content.append({"type": "image", "image": ref_img_infos})
    
    # 3. 添加自定义文本prompt
    content.append({"type": "text", "text": custom_prompt})
    
    messages = [{"role": "user", "content": content}]
    return messages



# ============================================================
# 文本处理工具
# ============================================================

def strip_thinking(text: str) -> str:
    """去掉 thinking 模式的...部分，只保留最终回答。"""
    # 去掉 thinking 模式的内容（从...开始到结束）
    cleaned = re.sub(r"\.\.\..*", "", text, flags=re.DOTALL)
    # 如果还有...标记，只保留之前的内容
    if "..." in cleaned:
        cleaned = cleaned[: cleaned.index("...")]
    return cleaned.strip()


# ============================================================
# 数据读写
# ============================================================

def parse_txt_data(file_path: str) -> List[Dict]:
    """
    解析 .txt 格式的输入文件。
    txt 中每行记录一个文件夹路径，文件夹中包含：
    - control video（文件名含 "control"）
    - 0或多张 reference image（文件名含 "ref"）
    - prompt/caption 文本文件（文件名含 "caption" 或 "prompt"）
    """
    from glob import glob

    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = [line.strip() for line in f if line.strip()]

    data_infos = []
    for data_dir in data_list:
        if not os.path.isdir(data_dir):
            logging.warning(f"目录不存在，跳过: {data_dir}")
            continue

        data_input_paths = sorted(glob(os.path.join(data_dir, "*")))
        control_path = None
        ref_img_paths = []
        caption = ""

        for data_input_path in data_input_paths:
            basename = os.path.basename(data_input_path)
            if "control" in basename:
                control_path = data_input_path
            if "ref" in basename:
                ref_img_paths.append(os.path.realpath(data_input_path))
            if "caption" in basename or "prompt" in basename:
                with open(data_input_path, "r", encoding='utf-8') as f:
                    caption = f.read().strip()

        if control_path is None:
            logging.warning(f"未找到 control 文件，跳过目录: {data_dir}")
            continue

        data_info = {
            "control_path": os.path.realpath(control_path),
            "ref_img_infos": ref_img_paths,
            "dense_caption": [
                {"content": caption, "dense_caption_type": "manual"}
            ],
        }
        data_infos.append(data_info)

    logging.info(f"从 txt 文件解析到 {len(data_infos)} 条数据")
    return data_infos


def read_input_data(file_path: str) -> List[Dict]:
    """读取输入数据，支持 JSON 数组、JSONL 和 TXT 格式。"""

    # 如果是 .txt 格式，使用专门的解析逻辑
    if file_path.endswith('.txt'):
        return parse_txt_data(file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # 尝试 JSON 数组
    if content.startswith('['):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                logging.info(f"读取到 JSON 数组，共 {len(data)} 条数据")
                return data
        except json.JSONDecodeError:
            pass

    # 尝试 JSONL
    items = []
    for line_num, line in enumerate(content.split('\n'), 1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            logging.warning(f"跳过第 {line_num} 行（JSON 解析错误）: {e}")

    logging.info(f"读取到 JSONL 数据，共 {len(items)} 条")
    return items


def write_jsonl(file_path: str, items: List[Dict]):
    """将字典列表写入 JSONL 文件（覆盖模式）。"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def append_jsonl(file_path: str, items: List[Dict]):
    """将字典列表追加写入 JSONL 文件（实时写入模式）。"""
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


# ============================================================
# 消息构造（针对 CogOmniControl 格式）
# ============================================================

def build_messages(item: Dict) -> List[Dict[str, Any]]:
    """
    从 CogOmniControl 数据条目构造标准的 Qwen3-VL messages 格式。
    
    输入格式：
        {
            "control_path": "path/to/control_video.mp4",
            "ref_img_infos": ["path/to/img1.jpg", "path/to/img2.jpg"],
            "prompt": "用户提示"
        }
    """
    try:
        # 尝试使用新的prompt模板
        from .cogomni_prompt_template import build_messages_with_custom_prompt
        return build_messages_with_custom_prompt(item)
    except ImportError:
        # 回退到原来的简单格式
        # prompt_text = item.get("prompt", "")
        prompt_text = build_cogomni_prompt(item)
        control_path = item.get("control_path", None)
        ref_img_infos = _extract_ref_images(item)

        content: List[Dict[str, Any]] = []

        # 1. 添加控制视频 content block
        if control_path and isinstance(control_path, str) and control_path.strip():
            content.append({"type": "video", "video": control_path})

        # 2. 添加参考图像 content blocks
        if ref_img_infos:
            print("获取到参考图像")
            if isinstance(ref_img_infos, list):
                for img_path in ref_img_infos:
                    if isinstance(img_path, str) and img_path.strip():
                        content.append({"type": "image", "image": img_path})
            elif isinstance(ref_img_infos, str) and ref_img_infos.strip():
                content.append({"type": "image", "image": ref_img_infos})

        # 3. 添加文本 content block
        if prompt_text.strip():
            content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]
        return messages


# ============================================================
# vLLM 后端推理
# ============================================================

def inference_vllm(
    items: List[Dict],
    base_model: str,
    lora_path: Optional[str],
    output_path: Optional[str] = None,
    batch_size: int = 16,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = -1,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: Optional[int] = None,
    max_model_len: Optional[int] = None,
) -> List[str]:
    """使用 vLLM 进行批量推理。"""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    if tensor_parallel_size is None:
        tensor_parallel_size = torch.cuda.device_count()

    logging.info(f"[vLLM] 加载基座模型: {base_model}")
    logging.info(f"[vLLM] LoRA 路径: {lora_path}")
    logging.info(f"[vLLM] tensor_parallel_size={tensor_parallel_size}")

    # 加载 processor
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    # 设置视频处理参数
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        if hasattr(vp, "min_pixels"):
            vp.min_pixels = 256 * 28 * 28
        if hasattr(vp, "max_pixels"):
            vp.max_pixels = 1664 * 28 * 28
        if hasattr(vp, "min_frames"):
            vp.min_frames = 4
        if hasattr(vp, "max_frames"):
            vp.max_frames = 81
        if hasattr(vp, "fps"):
            vp.fps = 12

    ip = processor.image_processor
    if hasattr(ip, "min_pixels"):
        ip.min_pixels = 256 * 28 * 28
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = 1280 * 28 * 28

    # 构建 vLLM 引擎
    llm_kwargs = dict(
        model=base_model,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=False,
        seed=42,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64

    llm = LLM(**llm_kwargs)

    # 构建 LoRA 请求
    lora_request = None
    if lora_path:
        lora_request = LoRARequest("lora_adapter", 1, lora_path)
        logging.info(f"[vLLM] LoRA adapter 已加载")

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        stop_token_ids=[],
    )

    # 批量处理
    all_outputs: List[str] = []
    total_batches = (len(items) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(items))
        batch_items = items[start:end]

        logging.info(f"[vLLM] 处理批次 {batch_idx + 1}/{total_batches} "
                     f"(条目 {start + 1}-{end}/{len(items)})")

        # 构造 vLLM 输入
        prompts = []
        for item in batch_items:
            messages = build_messages(item)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 处理多模态数据
            mm_data = {}
            video_kwargs = {}
            try:
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=processor.image_processor.patch_size,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
                if image_inputs is not None:
                    mm_data['image'] = image_inputs
                if video_inputs is not None:
                    mm_data['video'] = video_inputs
            except TypeError:
                # 旧版 qwen_vl_utils
                try:
                    image_inputs, video_inputs = process_vision_info(messages)
                    if image_inputs is not None:
                        mm_data['image'] = image_inputs
                    if video_inputs is not None:
                        mm_data['video'] = video_inputs
                    video_kwargs = {}
                except Exception as e:
                    logging.warning(f"process_vision_info 失败: {e}")
            except Exception as e:
                logging.warning(f"process_vision_info 失败: {e}")

            prompt_dict = {"prompt": text}
            if mm_data:
                prompt_dict["multi_modal_data"] = mm_data
            if video_kwargs:
                prompt_dict["mm_processor_kwargs"] = video_kwargs
            prompts.append(prompt_dict)

        # 推理
        t0 = time.time()
        if lora_request:
            outputs = llm.generate(prompts, sampling_params=sampling_params,
                                   lora_request=lora_request)
        else:
            outputs = llm.generate(prompts, sampling_params=sampling_params)
        elapsed = time.time() - t0

        batch_completions = []
        for out in outputs:
            generated_text = out.outputs[0].text
            all_outputs.append(generated_text)
            batch_completions.append(generated_text)

        # 实时写入本批次结果
        if output_path:
            batch_results = []
            for item, completion in zip(batch_items, batch_completions):
                new_item = item.copy()
                new_item['completion'] = completion
                new_item['final_answer'] = strip_thinking(completion)
                new_item['completion_length'] = len(completion)
                new_item['final_answer_length'] = len(new_item['final_answer'])
                batch_results.append(new_item)
            append_jsonl(output_path, batch_results)
            logging.info(f"[vLLM] 批次 {batch_idx + 1} 结果已实时写入 {output_path}")

        logging.info(f"[vLLM] 批次 {batch_idx + 1} 完成，耗时 {elapsed:.1f}s，"
                     f"平均 {elapsed / len(batch_items):.1f}s/条")

    return all_outputs


# ============================================================
# transformers 后端推理
# ============================================================

def inference_transformers(
    items: List[Dict],
    base_model: str,
    lora_path: Optional[str],
    output_path: Optional[str] = None,
    batch_size: int = 1,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = -1,
) -> List[str]:
    """使用 transformers + PeftModel 进行推理。"""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    logging.info(f"[transformers] 加载基座模型: {base_model}")

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    # 设置视频处理参数
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        if hasattr(vp, "min_pixels"):
            vp.min_pixels = 256 * 28 * 28
        if hasattr(vp, "max_pixels"):
            vp.max_pixels = 1664 * 28 * 28
        if hasattr(vp, "min_frames"):
            vp.min_frames = 4
        if hasattr(vp, "max_frames"):
            vp.max_frames = 81
        if hasattr(vp, "fps"):
            vp.fps = 12

    ip = processor.image_processor
    if hasattr(ip, "min_pixels"):
        ip.min_pixels = 256 * 28 * 28
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = 1280 * 28 * 28

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    if lora_path:
        from peft import PeftModel
        logging.info(f"[transformers] 加载 LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # 合并 LoRA 权重以加速推理
        logging.info(f"[transformers] LoRA 已合并到基座模型")

    model.eval()

    try:
        from qwen_vl_utils import process_vision_info
        has_qwen_utils = True
    except ImportError:
        has_qwen_utils = False
        logging.warning("qwen_vl_utils 未安装，多模态输入可能无法正确处理")

    device = next(model.parameters()).device
    all_outputs: List[str] = []

    for i, item in enumerate(items):
        logging.info(f"[transformers] 推理条目 {i + 1}/{len(items)}")

        messages = build_messages(item)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = None, None
        if has_qwen_utils:
            try:
                image_inputs, video_inputs = process_vision_info(messages)
            except Exception as e:
                logging.warning(f"process_vision_info 失败: {e}")

        try:
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except Exception as e:
            logging.warning(f"processor 处理失败，仅用文本: {e}")
            inputs = processor(text=[text], padding=True, return_tensors="pt")

        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():

            output_ids = model.generate(**inputs, max_new_tokens=8196)

        generated = output_ids[0][input_len:]
        text_out = processor.batch_decode(
            [generated], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        elapsed = time.time() - t0

        all_outputs.append(text_out)

        # 实时写入当前条目结果
        if output_path:
            new_item = item.copy()
            new_item['completion'] = text_out
            new_item['final_answer'] = strip_thinking(text_out)
            new_item['completion_length'] = len(text_out)
            new_item['final_answer_length'] = len(new_item['final_answer'])
            append_jsonl(output_path, [new_item])

        logging.info(f"[transformers] 条目 {i + 1} 完成，耗时 {elapsed:.1f}s，"
                     f"输出长度 {len(text_out)} 字符")

    return all_outputs


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CogOmniControl 推理脚本：处理包含 control video 和 reference images 的 JSONL 输入"
    )
    # 输入输出
    parser.add_argument("--input", required=True,
                        help="输入 JSONL 文件路径（包含 control_path 和 ref_img_infos）")
    parser.add_argument("--output", required=True,
                        help="输出 JSONL 文件路径")

    # 模型配置
    parser.add_argument("--base_model", default="Qwen/Qwen3-VL-8B-Thinking",
                        help="基座模型路径或 HuggingFace ID")
    parser.add_argument("--lora_path", default=None,
                        help="LoRA adapter checkpoint 路径（不指定则使用纯基座模型）")

    # 推理后端
    parser.add_argument("--backend", default="vllm", choices=["vllm", "transformers"],
                        help="推理后端（默认 vllm）")

    # 生成参数
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批量大小（vLLM 后端有效）")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="采样温度（0 表示贪心解码）")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p 采样")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k 采样（-1 表示不限制）")

    # vLLM 专用参数
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="vLLM GPU 显存利用率")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="vLLM tensor 并行数（默认使用所有 GPU）")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="vLLM 最大模型长度")

    # 数据分片参数（用于多卡并行推理）
    parser.add_argument("--start_index", type=int, default=None,
                        help="数据起始行索引（从0开始，包含），用于多卡并行时指定数据范围")
    parser.add_argument("--end_index", type=int, default=None,
                        help="数据终止行索引（不包含），用于多卡并行时指定数据范围")

    # 其他
    parser.add_argument("--skip_existing", action="store_true",
                        help="如果输出文件已存在，跳过已有 completion 的条目")
    parser.add_argument("--limit", type=int, default=None,
                        help="仅处理前 N 条数据（用于调试）")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别")

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 读取输入数据
    logging.info(f"读取输入文件: {args.input}")
    items = read_input_data(args.input)
    total_count = len(items)

    # 按起始/终止行索引切片（优先于 --limit）
    if args.start_index is not None or args.end_index is not None:
        si = args.start_index if args.start_index is not None else 0
        ei = args.end_index if args.end_index is not None else len(items)
        si = max(0, si)
        ei = min(len(items), ei)
        items = items[si:ei]
        logging.info(f"按索引范围切片: [{si}, {ei})，共 {len(items)} 条（总数据 {total_count} 条）")

    if args.limit is not None:
        items = items[:args.limit]
        logging.info(f"限制处理前 {args.limit} 条数据")

    # 跳过已有 completion 的条目
    items_to_process = items
    existing_results = {}
    if args.skip_existing and os.path.exists(args.output):
        logging.info(f"检查已有结果: {args.output}")
        existing = read_input_data(args.output)
        for ex_item in existing:
            if 'completion' in ex_item and ex_item['completion']:
                # 用 prompt 作为 key 去重
                key = ex_item.get('prompt', '')[:200]
                existing_results[key] = ex_item
        items_to_process = []
        skipped = 0
        for item in items:
            key = item.get('prompt', '')[:200]
            if key in existing_results:
                skipped += 1
            else:
                items_to_process.append(item)
        logging.info(f"跳过 {skipped} 条已有结果，剩余 {len(items_to_process)} 条待处理")

    if not items_to_process:
        logging.info("所有条目已处理完毕，无需推理")
        return

    # 准备输出文件（如果有已有结果，先写入已有结果作为文件头部）
    if existing_results:
        # 覆盖写入已有结果
        existing_items = []
        for item in items:
            key = item.get('prompt', '')[:200]
            if key in existing_results:
                existing_items.append(existing_results[key])
        write_jsonl(args.output, existing_items)
        logging.info(f"已写入 {len(existing_items)} 条已有结果到 {args.output}")
    else:
        # 清空输出文件，准备追加写入
        with open(args.output, 'w', encoding='utf-8') as f:
            pass

    # 推理（结果在推理过程中实时追加写入）
    logging.info(f"开始推理，后端: {args.backend}，共 {len(items_to_process)} 条")
    logging.info(f"结果将实时写入: {args.output}")
    t_start = time.time()

    if args.backend == "vllm":
        completions = inference_vllm(
            items=items_to_process,
            base_model=args.base_model,
            lora_path=args.lora_path,
            output_path=args.output,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
    else:
        completions = inference_transformers(
            items=items_to_process,
            base_model=args.base_model,
            lora_path=args.lora_path,
            output_path=args.output,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

    t_total = time.time() - t_start
    logging.info(f"推理完成，总耗时 {t_total:.1f}s，"
                 f"平均 {t_total / len(items_to_process):.1f}s/条")

    total_output = len(existing_results) + len(completions)
    logging.info(f"处理完成，共输出 {total_output} 条数据到 {args.output}")

    # 打印统计信息
    completion_lengths = [len(c) for c in completions]
    final_lengths = [len(strip_thinking(c)) for c in completions]
    print("\n" + "=" * 60)
    print("  CogOmniControl Inference Summary")
    print("=" * 60)
    print(f"  总条目数:        {total_output}")
    print(f"  新推理条目数:    {len(completions)}")
    print(f"  总耗时:          {t_total:.1f}s")
    print(f"  平均耗时:        {t_total / len(completions):.1f}s/条")
    print(f"  Completion 长度: min={min(completion_lengths)}, "
          f"max={max(completion_lengths)}, "
          f"avg={sum(completion_lengths) / len(completion_lengths):.0f}")
    print(f"  Final Answer 长度: min={min(final_lengths)}, "
          f"max={max(final_lengths)}, "
          f"avg={sum(final_lengths) / len(final_lengths):.0f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()