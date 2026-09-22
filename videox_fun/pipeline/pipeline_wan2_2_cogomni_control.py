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

from ..models import (AutoencoderKLWan, AutoTokenizer,
                              WanT5EncoderModel, CogOmniControlWanModel)
from ..utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                get_sampling_sigmas)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ..utils.fm_solvers_lcm import FlowMatchLCMScheduler
from ..utils import VideoReader_contextmanager

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
    

class CogOmniControlPipeline(DiffusionPipeline):
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
        transformer: CogOmniControlWanModel,
        transformer_2: CogOmniControlWanModel = None,
        scheduler: FlowMatchEulerDiscreteScheduler = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, 
            transformer_2=transformer_2, scheduler=scheduler
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
            control_latents = self.vae.encode(control_pixel_values)[0].mode()
        
        ref_image_latents = None
        if ref_images is not None:
            ref_frames = []
            for ref_image in ref_images:
                ref_pixel_item = self.process_reference_images(ref_image, target_size=(height, width)) / 255.
                ref_frames.append(ref_pixel_item)
            ref_pixel_values = self.video_processor.preprocess_video(ref_frames, height=height, width=width)
            ref_pixel_values = ref_pixel_values.to(device).to(weight_dtype)
            ref_image_latents = self.vae.encode(ref_pixel_values)[0].mode()

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

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
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
        seq_len = noisy_seq_len + control_seq_len + ref_seq_len
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
                        x=latent_model_input,
                        control_latents=control_latents_input,
                        ref_image_latents=ref_image_latents_input,
                        context=in_prompt_embeds,
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
