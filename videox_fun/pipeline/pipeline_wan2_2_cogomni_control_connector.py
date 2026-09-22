import inspect
import math
from dataclasses import dataclass
import os
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import VaeImageProcessor
import torch.nn.functional as F
from einops import rearrange
from ..models import (AutoencoderKLWan, AutoTokenizer,
                              WanT5EncoderModel, CogOmniControlWanModel_Connector)
from ..utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                get_sampling_sigmas)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ..utils.fm_solvers_lcm import FlowMatchLCMScheduler
from ..utils import VideoReader_contextmanager
from ..utils.llm_utils import build_llm_embeds

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    videos: torch.Tensor

def smart_resize(frames, sample_h, sample_w):
    # 保持横总比不变缩放
    resized_frames = []
    max_pixel_count = sample_h * sample_w
    for frame in frames:
        h, w = frame.shape[:2]
        if h * w > max_pixel_count:
            scale_factor = (max_pixel_count / (h * w)) ** 0.5
            new_h = int(h * scale_factor / 16) * 16
            new_w = int(w * scale_factor / 16) * 16
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized_frame = frame
            new_w, new_h = w, h
        resized_frames.append(resized_frame)
    return resized_frames, new_h, new_w
    

class CogOmniControlConnectorPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _optional_components = ["transformer_2"]
    model_cpu_offload_seq = "text_encoder->transformer_2->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: CogOmniControlWanModel_Connector,
        transformer_2: CogOmniControlWanModel_Connector = None,
        scheduler: FlowMatchEulerDiscreteScheduler = None,
        llm=None,
        processor=None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, 
            transformer_2=transformer_2, scheduler=scheduler, llm=llm, processor=processor
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def _get_llm_embeds(
        self, 
        control_pixel_values, 
        ref_pixel_values, 
        text,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        ):
        device = device or self.llm.device
        dtype = dtype or self.llm.dtype

        # # user_instruction = ["Instruction: 为文本到视频生成模型分析并推理如何基于给定的控制视频（带有时间戳的图像）、参考图像（不带时间戳的图像）以及文本描述三个条件下生成最终视频。"]
        # user_instruction = ["Role: 你是一位精通多模态理解与视频生成的专家。\nInsturction:你的任务是协调给定的控制视频（带有时间戳的图像）、参考图像（不带时间戳的图像）以及文本描述三者，分析各模态中哪些核心信息需要被提取、对齐或保留。\n确保生成的视频：1. 符合高度抽象化的控制视频所暗示的潜在运动轨迹和空间结构；2. 符合参考图像视觉特征、纹理细节、光影基调及构图风格；3. 符合文本描述。\n思考过程与方案输出必须精炼，总长度不能超过 2048 Tokens"]
        # system_prompt = "<|im_start|>user\n"
        # for i in range(1, ref_pixel_values.shape[1] + 1):
        #     system_prompt = system_prompt + f"Picture {i}:<|vision_start|><|image_pad|><|vision_end|>\n"

        # system_prompt = system_prompt + f"\nVideo 1:<|vision_start|><|video_pad|><|vision_end|>\n文本描述: {text}. {user_instruction[0]}<|im_end|>\n<|im_start|>assistant\n<think>"
                
        # user_instruction = ["你是一个用于视频生成的多模态理解与推理系统。\n你需要通过分析给定的素材（控制视频(带时间戳的图像)、参考图像(不带时间戳的图像)和文本描述），解决素材间可能的语义不对齐问题，推导出一套完整详细的视频生成逻辑，思考最终应该生成怎样的视频\n\n还需要关注除了文本和控制视频外可能的物理逻辑和运动轨迹或规律，包括文本描述中未给定，但其他素材可能包含的视觉线索，进行隐式信息推理\n以中文形式输出一段详细的综合控制逻辑，不少于500字"]
        # system_prompt = "<|im_start|>user\n"
        # if ref_pixel_values is not None:
        #     for i in range(1, ref_pixel_values.shape[1] + 1):
        #         system_prompt = system_prompt + f"Picture {i}:<|vision_start|><|image_pad|><|vision_end|>\n"
        # system_prompt = system_prompt + f"\nVideo 1:<|vision_start|><|video_pad|><|vision_end|>\n{user_instruction[0]}\n文本描述: {text}<|im_end|>\n<|im_start|>assistant\n<think>"
        
        # CogVLM-v1 prompt
        user_instruction = ["# Role\n\n你是一位多模态视频生成条件协调专家，同时也是视频质量评估方案规划师。\n分析给定的素材（控制视频(带时间戳的图像)、参考图像(不带时间戳的图像)和文本描述）你需要：\n1. 输出一段连贯的推理协调方案，指导视频生成模型如何从这些条件生成最终视频\n2. 基于方案内容，从评估器库中选择合适的评估器\n\n# Inputs\n\n1. 控制视频，形式不固定（3D白模/线稿/深度图/骨架/特效预览/分镜storyboard等），提供的信息因场景而异。\n2. 静态参考图像，通常定义目标视频的视觉世界观。\n3. 文本描述，表达创作意图。\n\n控制视频和参考图像可能不会每次都提供，如果没有则忽略。\n\n\n# 生成方案Rules\n\n1. 输出必须是一段连贯的文字，不要列表、不要表格，不少于500字。\n2. 控制视频的作用不是预设的，根据实际内容判断。\n3. 参考图像通常定义视觉世界观。\n4. 所有决策自然嵌入叙述中并给出理由。\n5. 主动推理条件暗示但未明说的效果，将笼统描述展开为具体物理过程。\n6. 不同条件的信息要主动组合，推理组合后的新效果。\n7. 具体可执行，不说空话。\n8. 对条件中缺失或断裂的关键信息，必须主动强调补全。\n9. 当实体需要在画面中出现或消失时，推理出合理的进场和退场动作，保证叙事连贯，不允许实体凭空出现或消失。\n10. 最终方案应当是一段信息稠密、逻辑连贯的文字，读完后能清晰知道目标视频的每个元素应该如何呈现、如何运动、如何交互。"]
        system_prompt = "<|im_start|>user\n"
        for i in range(1, ref_pixel_values.shape[1] + 1):
            system_prompt = system_prompt + f"Picture {i}:<|vision_start|><|image_pad|><|vision_end|>\n"
        system_prompt = system_prompt + f"\nVideo 1:<|vision_start|><|video_pad|><|vision_end|>\n{user_instruction[0]}\n\n文本描述: {text}# Evaluator Registry（评估器库）\n\n固定名称清单（**必须逐字精确匹配，严禁修改、缩写、翻译**）：\n\n- `文本遵循验证器` - 视频是否忠实遵循文本Prompt的核心内容。**当方案确定严格遵循文本时，或冲突情况确定遵循文本时调用**。\n- `主体ID保持验证器` - 视频中主体身份是否与参考图像一致。**当参考图像中存在可识别的角色/主体时调用**。\n- `参考图像视觉特征验证器` - 视频是否以图像为视觉参考基准。**当方案指定\"参考图像的视觉特征/形象\"时调用，注意并不是要求视频中的某一帧必须是该残稿图像。\n- `参考图像像素对齐验证器` - 视频是否严格遵循参考图像作为关键帧，保持像素级别的对齐。**当方案指定\"以参考图像为基础，应用控制视频暗示的动态信息\"时调用。与\"控制视频遵循验证器\"互斥**。\n- `控制视频遵循验证器` - 视频是否严格遵循控制视频的空间布局/姿态/深度等具体信号。**当方案指定\"以控制视频为基础，应用参考图像的视觉特征\"时调用。与\"参考图像像素对齐验证器\"互斥**。\n- `物理动态特效验证器` - 火焰/水流/烟雾/爆炸等特效是否动态合理。**当输入中存在物理特效元素时调用**。\n- `多模态隐含因果验证器` - 跨模态输入暗示的因果关系是否被\"脑补\"进视频。**当多模态输入之间存在需要推断的因果联动时调用**。\n- `时空平滑度验证器` - 视频是否存在闪烁、跳变、撕裂、卡顿（语义层面）。**始终调用**。\n- `交互逻辑性验证器` - 物体交互是否符合物理和日常逻辑。**当视频中存在物体交互、接触、移动场景时调用**。\n- `负面伪影检测器` - 多头、多肢、形变、漂浮等AI伪影检测。**始终调用**。\n- `Storyboard标注遵循验证器` - 视频是否遵循Storyboard上的文字标注指令。**当控制视频是Storyboard分镜、附带文字标注、且方案决定遵循时调用**。\n- `美学评分器` - 画面美学质量评分。**当内容属于艺术性/风景性/设计性题材时调用**（适合：风景/艺术/时尚/建筑/精致渲染；不适合：动漫/卡通/UGC/监控/游戏截图）。\n- `动态程度评估器` - 视频动态幅度（基于光流）。**当方案判断视频应有较大动态变化时调用**。\n- `运动平滑度评估器` - 运动轨迹的平滑性和连续性（基于光流）。**当视频中存在明显运动时调用**。\n\n# Output Format\n\n严格按照以下一行式扁平格式输出：\n\n<方案文本>\n[tools][评估器名称1][评估器名称2][评估器名称3]...\n<|im_end|>\n\n<|im_start|>assistant\n<think>"
                

        with torch.no_grad():
            from transformers import set_seed
            set_seed(42)
            # B, F, C, H, W
            llm_inputs = self.processor(
                text = system_prompt,
                videos = (((control_pixel_values[0] + 1) / 2.0) * 255),
                images = [(((ref_pixel_values[0, i:i+1, :, :, :]+ 1) / 2.0) * 255) for i in range(ref_pixel_values.shape[1])] if ref_pixel_values is not None and ref_pixel_values.shape[1] != 0 else None, # [B, C, F, H, W] -> [[B, C, H, W], [B, C, H, W]]
                video_metadata = {
                    "fps": 12,
                    "total_num_frames": control_pixel_values.shape[1],
                    },
                return_tensors="pt",
            ).to(dtype=dtype, device=device)

            self.llm.to(dtype=dtype, device=device)
            outputs = self.llm.generate(
                        **llm_inputs, 
                        max_new_tokens=1024,
                        return_dict_in_generate=True,
                        output_hidden_states=True)
            # cot_steps = [step[-2] for step in outputs.hidden_states] 
            # cot_steps = [step[-1] for step in outputs.hidden_states]
            # llm_embeds = torch.cat(cot_steps, dim=1) # 1, seq, C
            # llm_embeds, cut_idx = build_llm_embeds(
            #             outputs, llm_inputs['input_ids'], self.processor.tokenizer,
            #             marker_text="# Evaluator Registry（评估器库）", 
            #             keep_marker=False
            #             )
            llm_embeds, cut_idx = build_llm_embeds(
                        outputs, llm_inputs['input_ids'], self.processor.tokenizer,
                        marker_text="# Evaluator Registry（评估器库）", 
                        keep_marker=False,
                        # layer_indices=llm_embed_layer_indices,
                        layer_indices=[16, 24, -1],
                        )
            # llm_embeds = [u[:len(llm_inputs.input_ids)] for u in cot_embeds]
            # llm_embeds = torch.stack(llm_embeds).to(dtype=dtype, device=device)
            print(control_pixel_values[0].shape, ref_pixel_values[0].shape, llm_embeds.shape)

            
            # generated_ids_trimmed = [
            #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(llm_inputs.input_ids, outputs.sequences)
            # ]
            generated_ids_trimmed = outputs.sequences
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(output_text)


        return llm_embeds, output_text

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.cpu().float().numpy()
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        # for control conditions start
        control_video: Optional[str] = None,
        ref_images: Optional[Union[str, List[str]]] = None,
        # for control conditions end
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        boundary: float = 0.875,
        comfyui_progressbar: bool = False,
        shift: int = 5,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.
        Args:

        Examples:

        Returns:

        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1
        
        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        # Prepare control conditions
        with VideoReader_contextmanager(control_video, num_threads=8) as video_reader:
            total_frame_num = len(video_reader)
            num_frames = min(num_frames, total_frame_num)
            control_frames = video_reader.get_batch(range(num_frames)).asnumpy() / 255.
            control_h, control_w = control_frames.shape[1:3]
            if control_h != height or control_w != width:
                # smart resize control_frames
                control_frames, height, width = smart_resize(control_frames, height, width)
            control_pixel_values = self.video_processor.preprocess_video(control_frames, height=height, width=width)
            control_pixel_values = control_pixel_values.to(device).to(weight_dtype)
            # self.vae = self.vae.to(device).to(weight_dtype)

            with torch.autocast(device_type="cuda", dtype=weight_dtype):
                control_latents = self.vae.encode(control_pixel_values)[0].mode()
            # control_latents = self.vae.encode(control_pixel_values)[0].mode()
            control_latents = control_latents.to(device).to(torch.bfloat16)
        
        ref_image_latents = None
        ref_pixel_values = None
        if ref_images is not None and len(ref_images) > 0:
            ref_frames = []
            for ref_image in ref_images:
                ref_pixel_item = self.process_reference_images(ref_image, target_size=(height, width)) / 255.
                ref_frames.append(ref_pixel_item)
            ref_pixel_values = self.video_processor.preprocess_video(ref_frames, height=height, width=width)
            ref_pixel_values = ref_pixel_values.to(device).to(weight_dtype)
            # self.vae = self.vae.to(device).to(weight_dtype)
            # ref_image_latents = self.vae.encode(ref_pixel_values)[0].mode()
            with torch.autocast(device_type="cuda", dtype=weight_dtype):
                ref_image_latents = self.vae.encode(ref_pixel_values)[0].mode()
            ref_image_latents = ref_image_latents.to(device).to(torch.bfloat16)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0


        # 4. Encode llm embeds
        print(control_pixel_values.shape, ref_pixel_values.shape) # b c f h w
        llm_embeds, output_text = self._get_llm_embeds(
            rearrange(control_pixel_values, "b c f h w -> b f c h w"),
            rearrange(ref_pixel_values, "b c f h w -> b f c h w") if ref_pixel_values is not None else None,
            # control_pixel_values,
            # ref_pixel_values if ref_pixel_values is not None else None,
            prompt,
            device=device,
        )

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,# + output_text[0],
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else:
            in_prompt_embeds = prompt_embeds

        

        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler,
                device=device,
                sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 1)

        # 5. Prepare latents
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )
        # inject low freqs of ref_image_latents to latents
        if ref_image_latents is not None:
            ref_image_latents_mean = ref_image_latents.mean(dim=[-2, -1], keepdim=True)
            # latents = (1-0.15) * latents + ref_image_latents_mean * 0.15
            # latents = self.apply_ref_low_freq_blending(latents, ref_image_latents, filter_size=7)
            
        if comfyui_progressbar:
            pbar.update(1)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (self.vae.latent_channels, (num_frames - 1) // self.vae.temporal_compression_ratio + 1, height // self.vae.spatial_compression_ratio, width // self.vae.spatial_compression_ratio)
        noisy_seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1]) 
        control_seq_len = noisy_seq_len
        ref_seq_len = math.ceil(
            (ref_image_latents.shape[3] * ref_image_latents.shape[4]) /
            (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) *
            ref_image_latents.shape[2]
        ) if ref_image_latents is not None else 0

        llm_seq_len = llm_embeds.shape[1]
        seq_len = noisy_seq_len + control_seq_len + ref_seq_len + llm_seq_len
        # for context parrallel
        seq_len = math.ceil(seq_len / 8) * 8
        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.transformer.current_steps = i

                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                control_latents_input = torch.cat([control_latents] * 2) if do_classifier_free_guidance else control_latents
                llm_embeds_input = torch.cat([llm_embeds] * 2) if do_classifier_free_guidance else llm_embeds
                ref_image_latents_input = None
                if ref_image_latents is not None:
                    ref_image_latents_input = torch.cat([ref_image_latents] * 2) if do_classifier_free_guidance else ref_image_latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    control_latents_input = self.scheduler.scale_model_input(control_latents_input, t)
                    if ref_image_latents_input is not None:
                        ref_image_latents_input = self.scheduler.scale_model_input(ref_image_latents_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                
                if self.transformer_2 is not None:
                    if t >= boundary * self.scheduler.config.num_train_timesteps:
                        local_transformer = self.transformer_2
                    else:
                        local_transformer = self.transformer
                else:
                    local_transformer = self.transformer

                # predict noise model_output
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    noise_pred = local_transformer(
                        x=latent_model_input.to(dtype=weight_dtype),
                        control_latents=control_latents_input.to(dtype=weight_dtype),
                        ref_image_latents=ref_image_latents_input.to(dtype=weight_dtype) if ref_image_latents_input is not None else None,
                        context=in_prompt_embeds,
                        llm_embeds=llm_embeds_input.to(dtype=weight_dtype),
                        t=timestep,
                        seq_len=seq_len,
                    )

                # perform guidance
                if do_classifier_free_guidance:
                    if self.transformer_2 is not None and (isinstance(self.guidance_scale, (list, tuple))):
                        sample_guide_scale = self.guidance_scale[1] if t >= self.transformer_2.config.boundary * self.scheduler.config.num_train_timesteps else self.guidance_scale[0]
                    else:
                        sample_guide_scale = self.guidance_scale
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if comfyui_progressbar:
                    pbar.update(1)

        # if ref_image_latents is not None:
        #     latents = self.apply_latents_adain(latents, ref_image_latents, blend_weight=0.8)

        if output_type == "numpy":
            video = self.decode_latents(latents)
        elif not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            video = torch.from_numpy(video)

        return WanPipelineOutput(videos=video)

    def apply_ref_low_freq_blending(self, init_noise: torch.Tensor, ref_latents: torch.Tensor, filter_size: int = 7) -> torch.Tensor:
        """
        将参考图 (Reference Image) 的低频特征提取并融合到初始噪声中，
        以此提供全局的底色偏置，防止视频生成过程中的颜色漂移 (Color Drifting)。
        
        参数:
            init_noise (torch.Tensor): 外部传入的初始纯噪声，形状应为 (B, C, T, H, W)。
            ref_latents (torch.Tensor): 参考图像的潜变量特征，形状可为 (B, C, 1, H, W) 或 (B, C, H, W)。
            filter_size (int, optional): 低通滤波器 (Average Pool) 的核大小。
                                        必须为奇数。值越大注入的宏观色块越平滑，值越小注入的细节越多。默认为 7。
                                        
        返回:
            torch.Tensor: 经过低频融合并完成方差重归一化后的新噪声，形状与 init_noise 保持一致。
        """
        assert filter_size % 2 == 1, "filter_size 必须是奇数，以保证 padding 对称。"
        
        if ref_latents.ndim == 4:
            ref_latents = ref_latents.unsqueeze(2) # 补齐时间维度: (B, C, 1, H, W)
            
        pad = filter_size // 2
        
        # ==========================================
        # 1. 提取参考图的低频宏观色彩 (Spatial Low-Pass)
        # ==========================================
        # 压缩时间维度以使用 2D 池化: (B, C, H, W)
        ref_spatial = ref_latents.squeeze(2) 
        ref_low_freq = F.avg_pool2d(ref_spatial, kernel_size=filter_size, stride=1, padding=pad)
        # 恢复时间维度，后续融合时会自动沿着 T 维度广播: (B, C, 1, H, W)
        ref_low_freq = ref_low_freq.unsqueeze(2)
        
        # ==========================================
        # 2. 提取并剥离初始噪声的高频细节 (Spatial High-Pass)
        # ==========================================
        B, C, T, H, W = init_noise.shape
        # 将 T 维度折叠进 Batch，对每一帧的噪声独立做 2D 空间平滑
        noise_spatial = init_noise.transpose(1, 2).reshape(B * T, C, H, W)
        noise_low_freq_spatial = F.avg_pool2d(noise_spatial, kernel_size=filter_size, stride=1, padding=pad)
        
        # 恢复回 5D 形状: (B, C, T, H, W)
        noise_low_freq = noise_low_freq_spatial.view(B, T, C, H, W).transpose(1, 2)
        
        # 原始噪声减去自身的低频部分，剩下的就是纯粹的高频细节噪声
        noise_high_freq = init_noise - noise_low_freq
        
        # ==========================================
        # 3. 频率重组与方差对齐 (Recombination & Re-normalization)
        # ==========================================
        # 融合：参考图的低频色彩 + 标准噪声的高频细节
        blended_noise = ref_low_freq + noise_high_freq
        
        # 极其关键：将方差重新缩放回 init_noise 的水平，防止 Flow Matching 轨迹发散
        std_before = init_noise.std()
        std_after = blended_noise.std()
        blended_mean = blended_noise.mean()
        
        # (x - mean) * (target_std / current_std) + mean
        blended_noise = (blended_noise - blended_mean) * (std_before / (std_after + 1e-8)) + blended_mean
        
        return blended_noise


    def apply_latents_adain(self, video_latents, ref_latents, blend_weight=0.8):
        """
        使用参考图的统计特征对生成的视频 Latent 进行条件归一化。
        
        参数:
        video_latents: ODE 求解器输出的最终干净视频 Latent (B, C, T, H, W)
        ref_latents: 参考图的 Latent (B, C, 1, H, W)
        blend_weight: 混合权重。1.0 表示 100% 强制使用参考图的颜色, 0.0 表示不改变。
                    建议 0.7 ~ 0.9, 给模型保留一点自身生成的动态空间。
        """
        eps = 1e-5
        
        # 如果参考图缺少时间维度，给它补上
        if ref_latents.ndim == 4:
            ref_latents = ref_latents.unsqueeze(2)

        # 1. 计算参考图在空间维度 (H, W) 上的均值和标准差
        ref_mean = ref_latents.mean(dim=[-2, -1], keepdim=True)
        ref_std = ref_latents.std(dim=[-2, -1], keepdim=True) + eps
        
        # 2. 计算视频每一帧在空间维度 (H, W) 上的均值和标准差
        vid_mean = video_latents.mean(dim=[-2, -1], keepdim=True)
        vid_std = video_latents.std(dim=[-2, -1], keepdim=True) + eps
        
        # 3. 标准化：减去视频自己的均值，除以自己的标准差 (变成 N(0,1))
        norm_vid = (video_latents - vid_mean) / vid_std
        
        # 4. 条件反归一化：乘上参考图的标准差，加上参考图的均值
        adain_vid = norm_vid * ref_std + ref_mean
        
        # 5. 动静结合：与原始生成的 Latent 进行加权混合
        final_vid = blend_weight * adain_vid + (1.0 - blend_weight) * video_latents
        
        return final_vid



    def process_reference_images(self, ref_img_paths, target_size, data_root=None):
        """
        Process multiple reference images: 
        1. Resize each image to a common scale while maintaining aspect ratio
        2. Horizontally concatenate them
        3. Resize the concatenated image to target size while maintaining aspect ratio
        4. Place on white canvas to match target size
        
        Args:
            ref_img_paths: List of paths to reference images
            target_size: Tuple of (height, width) for target canvas size
            data_root: Optional data root directory
        
        Returns:
            Processed reference image (h, w, 3) resized to target_size
        """
        if not isinstance(ref_img_paths, list):
            ref_img_paths = [ref_img_paths]

        # Determine common scale based on target size and number of images
        canvas_height, canvas_width = target_size
        num_images = len(ref_img_paths)
        
        # Calculate available width per image (accounting for potential spacing)
        available_width_per_img = canvas_width // num_images
        
        # Process each image to the common scale
        processed_imgs = []
        max_height = 0
        total_width = 0
        
        for ref_img_path in ref_img_paths:
            if data_root is not None:
                full_path = os.path.join(data_root, ref_img_path)
            else:
                full_path = ref_img_path
                
            # Load image
            img = Image.open(full_path).convert("RGB")
            img = np.array(img)
            
            # Calculate scale to fit within available width while maintaining aspect ratio
            orig_height, orig_width = img.shape[:2]
            scale = min(available_width_per_img / orig_width, canvas_height / orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            # Resize image
            resized_img = cv2.resize(img, (new_width, new_height))
            processed_imgs.append(resized_img)
            
            # Update dimensions for concatenation
            max_height = max(max_height, new_height)
            total_width += new_width
        
        # Create canvas for concatenated image
        concatenated_img = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255
        
        # Place images on canvas
        x_offset = 0
        for img in processed_imgs:
            h, w = img.shape[:2]
            # Center vertically
            y_offset = (max_height - h) // 2
            concatenated_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
            x_offset += w
        
        # Resize the concatenated image to target size while maintaining aspect ratio
        ref_height, ref_width = concatenated_img.shape[:2]
        
        # Calculate scale to fit within target size
        scale = min(canvas_height / ref_height, canvas_width / ref_width)
        new_height = int(ref_height * scale)
        new_width = int(ref_width * scale)
        resized_image = cv2.resize(concatenated_img, (new_width, new_height))
        
        # Create white canvas for final output
        white_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # Center the resized image on the canvas
        top = (canvas_height - new_height) // 2
        left = (canvas_width - new_width) // 2
        white_canvas[top:top + new_height, left:left + new_width] = resized_image
        
        return white_canvas
