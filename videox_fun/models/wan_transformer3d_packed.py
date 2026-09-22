"""
FramePack for Wan-2.1-T2V. Reproduced from the Inverted anti-drifting sampling in
https://arxiv.org/pdf/2504.12626.

Author: xzavierpeng
Date: 2025/08/08
"""

# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py

import glob
import json
import math
import os
import types
from typing import Any, Dict
import re
import einops
import torch
import accelerate
from torch.cuda import amp
from torch import nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version
from diffusers.utils import WEIGHTS_NAME
from safetensors.torch import save_file, load_file
from safetensors import safe_open

from ..dist import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from ..dist.wan_xfuser import usp_attn_forward
from .cache_utils import TeaCache
from ..utils.hash_utils import (
    precalculate_safetensors_hashes,
)
from .wan_transformer3d import (
    attention,
    sinusoidal_embedding_1d,
    rope_params,
    get_1d_rotary_pos_embed_riflex,
    WanRMSNorm,
    WanLayerNorm,
    WanT2VCrossAttention,
    WanI2VCrossAttention,
    MLPProj,
)


def pad_for_3d_conv(x, kernel_size):
    """
    Pad the input tensor `x` to make its spatial dimensions divisible by the kernel size.

    Args:
        x (`torch.Tensor`): The input tensor with shape
            (batch_size, channels, time, height, width).
        kernel_size (`tuple[int, int, int]`): The kernel size for 3D convolution
            with shape (time, height, width).

    Returns:
        `torch.Tensor`: The padded tensor.
    """
    _, _, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


def center_down_sample_3d(x, kernel_size):
    """
    Perform 3D downsampling using average pooling with a given kernel size.

    Args:
        x (`torch.Tensor`): The input tensor with shape
            (batch_size, channels, depth, height, width).
        kernel_size (`tuple[int, int, int]`): The size of the pooling kernel
            with shape (depth, height, width).

    Returns:
        `torch.Tensor`: The downsampled tensor.
    """
    return torch.nn.functional.avg_pool3d(x, kernel_size, stride=kernel_size)


@amp.autocast(enabled=False)
def get_indiced_rope_freqs(indices, hs, ws, freqs, dim=128):
    """
    Retrieve the frequency tensor at specific positions for each sample in a batch.

    Args:
        indices (`list[torch.Tensor]`): A list of batch index tensors, each with
            shape [Li], where `Li` is the length of each sample in the F dimension.
        hs (`list[torch.Tensor]`): A list of batch tensors, each containing a single
            element representing the length of the H dimension.
        ws (`list[torch.Tensor]`): A list of batch tensors, each containing a single
            element representing the length of the W dimension.
        freqs (`torch.Tensor`): A tensor containing all frequency values within the
            maximum position, shape [max_length, D].
        dim (`int`, *optional*): The dimensionality `D` of the positional encoding
            (default is 128).

    Returns:
        `list[torch.Tensor]`: A list of frequency tensors at the specified indices,
            shape [[L1, H1, W1, D/2], [L2, H2, W2, D/2], ...].
    """
    c = dim // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output_freqs = []
    for indice, h, w in zip(indices.tolist(), hs.tolist(), ws.tolist()):
        f = len(indice)
        freqs_i = torch.cat(
            [
                freqs[0][indice].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        output_freqs.append(freqs_i)
    return output_freqs


@amp.autocast(enabled=False)
def apply_indiced_rope_freqs(x, latent_freqs):
    """
    Apply rotational position encoding (RoPE) before attention calculation.

    Args:
        x (`torch.Tensor`): The query or key tensor in the attention mechanism,
            shape [bs, L, num_heads, head_dim].
        latent_freqs (`torch.Tensor`): The frequency tensor corresponding to positions,
            shape [bs, L, 1, head_dim/2].

    Returns:
        `torch.Tensor:` The updated query or key tensor with RoPE applied, shape
            [bs, L, num_heads, head_dim].
    """
    bs, seq_len, n_heads, _ = x.shape
    x = torch.view_as_complex(x.to(torch.float32).reshape(bs, seq_len, n_heads, -1, 2))
    x = torch.view_as_real(x * latent_freqs).flatten(3)
    return x.float()


class WanSelfAttention(nn.Module):
    """
    Self-attention mechanism with optional frequency encoding and normalization.
    """

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        """
        Initializes the WanSelfAttention layer.

        Args:
            dim (`int`): The dimensionality of the input and output features.
            num_heads (`int`): The number of attention heads.
            window_size (`tuple`, *optional*): The attention window size. Defaults
                to (-1, -1).
            qk_norm (`bool`, *optional*): Whether to normalize the query and key.
                Defaults to `True`.
            eps (`float`, *optional*): Small constant for numerical stability
                in normalization.
        """
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, latent_freqs, seq_lens, dtype):
        """
        Args:
            x (`torch.Tensor`): Shape [B, L, num_heads, C / num_heads]
            seq_lens (`torch.Tensor`): Shape [B]
            grid_sizes (`torch.Tensor`): Shape [B, 3], the second dimension contains (F, H, W)
            freqs (`torch.Tensor`): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
            k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
            v = self.v(x.to(dtype)).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q = apply_indiced_rope_freqs(q, latent_freqs).to(dtype)
        k = apply_indiced_rope_freqs(k, latent_freqs).to(dtype)

        x = attention(
            q=q,
            k=k,
            v=v.to(dtype),
            q_lens=seq_lens,
            k_lens=seq_lens,
            window_size=self.window_size,
        )
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        x = self.o(x)

        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):
    """
    A block of self-attention and cross-attention with feedforward network (FFN)
    for use in transformer-based models.
    """

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        """
        Initializes the WanAttentionBlock.

        Args:
            cross_attn_type (`str`): Type of cross-attention mechanism.
            dim (`int`): Dimensionality of input/output vectors.
            ffn_dim (`int`): Dimensionality of feedforward network's hidden layer.
            num_heads (`int`): Number of attention heads.
            window_size (`tuple`, *optional*): Window size for attention. Default is (-1, -1).
            qk_norm (`bool`, *optional*): Whether to normalize queries and keys in attention.
            cross_attn_norm (`bool`, *optional*): Whether to normalize after cross-attention.
            eps (`float`, *optional*): Epsilon for numerical stability during normalization.
        """
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self, x, e, latent_freqs, seq_lens, context, context_lens, dtype=torch.float32
    ):
        """
        Args:
            x (`torch.Tensor`): Shape [B, L, C]
            e (`torch.Tensor`): Shape [B, 6, C]
            seq_lens (`torch.Tensor`): Shape [B], length of each sequence in batch
            grid_sizes (`torch.Tensor`): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(`torch.Tensor`): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)

        y = self.self_attn(temp_x, latent_freqs, seq_lens, dtype)
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            # cross-attention
            x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype)

            # ffn function
            temp_x = self.norm2(x) * (1 + e[4]) + e[3]
            temp_x = temp_x.to(dtype)

            y = self.ffn(temp_x)
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):
    """
    A head module that applies normalization, modulation, and a linear transformation.
    """

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        """
        Initializes the Head.

        Args:
            dim (`int`): Input feature dimension.
            out_dim (`int`): Output feature dimension.
            patch_size (`tuple`): Shape of the patch.
            eps (`float`): Small value to avoid division by zero in normalization.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e, context_length):
        """
        Args:
            x (`torch.Tensor`): Shape [B, L1, C]
            e (`torch.Tensor`): Shape [B, C]
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        norm_out = self.norm(x) * (1 + e[1]) + e[0]
        norm_out = norm_out[:, :context_length, :]
        x = self.head(norm_out)
        return x


class WanPatchEmbedForCleanLatents(nn.Module):
    """
    A module for 3D patch embedding with multiple compression ratio.
    """

    def __init__(self, in_dim, dim):
        """
        Initializes the WanPatchEmbedForCleanLatents.

        Args:
            in_dim (`int`): Input feature dimension.
            dim (`int`): Output feature dimension.
        """
        super().__init__()
        self.proj = nn.Conv3d(in_dim, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(in_dim, dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(in_dim, dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))

    def initialize_weight_from_another_conv3d(self, another_layer):
        """
        Initialize the weights and biases of the current layer using the weights from
        another Conv3D layer.

        Args:
            another_layer (`torch.nn.Conv3d`): The Conv3D layer from which to copy the
                weights and biases.
        """
        weight = another_layer.weight.detach().clone()
        bias = another_layer.bias.detach().clone()

        sd = {
            "proj.weight": weight.clone(),
            "proj.bias": bias.clone(),
            "proj_2x.weight": einops.repeat(
                weight, "b c t h w -> b c (t tk) (h hk) (w wk)", tk=2, hk=2, wk=2
            )
            / 8.0,
            "proj_2x.bias": bias.clone(),
            "proj_4x.weight": einops.repeat(
                weight, "b c t h w -> b c (t tk) (h hk) (w wk)", tk=4, hk=4, wk=4
            )
            / 64.0,
            "proj_4x.bias": bias.clone(),
        }

        sd = {k: v.clone() for k, v in sd.items()}

        self.load_state_dict(sd)

    def prepare_optimizer_params(self, default_lr, lr_1x=None, lr_2x=None, lr_4x=None):
        """
        Prepare optimizer parameters with specified learning rates for different layers.

        Args:
            default_lr (`float`): The default learning rate to use if a specific rate is
                not provided.
            lr_1x (`float`, *optional*): The learning rate for the 'proj' layer. Defaults
                to None, in which case `default_lr` is used.
            lr_2x (`float`, *optional*): The learning rate for the 'proj_2x' layer. Defaults
                to None, in which case `default_lr` is used.
            lr_4x (`float`, *optional*): The learning rate for the 'proj_4x' layer. Defaults
                to None, in which case `default_lr` is used.

        Returns:
            `list[dict]`: A list of dictionaries, each containing the parameters of a layer
                and its associated learning rate.
        """
        self.requires_grad_(True)
        all_params = []

        param_data_1x = {
            "params": self.proj.parameters(),
            "lr": lr_1x if lr_1x is not None else default_lr,
        }
        all_params.append(param_data_1x)
        param_data_2x = {
            "params": self.proj_2x.parameters(),
            "lr": lr_2x if lr_2x is not None else default_lr,
        }
        all_params.append(param_data_2x)
        param_data_4x = {
            "params": self.proj_4x.parameters(),
            "lr": lr_4x if lr_4x is not None else default_lr,
        }
        all_params.append(param_data_4x)

        return all_params

    def save_weights(self, file, dtype, metadata):
        """
        Save the model weights to a specified file with optional dtype conversion and
        metadata.

        Args:
            file (`str`): The path to the file where weights will be saved.
            dtype (`torch.dtype`, *optional*): The data type to which the weights should
                be converted. If None, no conversion is performed.
            metadata (`Dict[str, Any]`, *optional*): Additional metadata to be saved along
                with the model weights. If None, no metadata is saved.
        """
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = precalculate_safetensors_hashes(
                state_dict, metadata
            )
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def load_weights(self, file, device=None, dtype=None):
        """
        Load model weights from a specified file with optional dtype and device settings.

        Args:
            file (`str`): The path to the file containing the model weights.
            device (`torch.device`, *optional*): The device to load the weights onto. If
                None, it uses the device of the model's parameters.
            dtype (`torch.dtype`, *optional*): The data type to convert the weights to.
                If None, no conversion is done.

        Returns:
            `Tuple[list, list]`: A tuple containing two lists:
                - `missing_keys`: List of keys in the model that were not found in the
                    state_dict.
                - `unexpected_keys`: List of keys in the state_dict that were not used
                    by the model.
        """
        if device is None:
            device = next(self.parameters()).device

        ext = os.path.splitext(file)[1].lower()

        if ext == ".safetensors":

            state_dict = {}
            with safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            state_dict = torch.load(file, map_location="cpu")

        if dtype is not None:
            for key in list(state_dict.keys()):
                state_dict[key] = state_dict[key].to(dtype)

        for key in list(state_dict.keys()):
            state_dict[key] = state_dict[key].to(device)

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        print(f"Loaded weights from: {file}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        return missing_keys, unexpected_keys

    def forward(self, inputs):
        """Installed as a plugin into WanTransformer3DModelPacked."""
        pass


class WanTransformer3DModelPacked(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """
    Wan FramePack diffusion backbone supporting only text-to-video.
    """

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        has_clean_patch_embedder=False,
    ):
        """
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)

        self.teacache = None
        self.gradient_checkpointing = False
        self.sp_world_size = 1
        self.sp_world_rank = 0

        if has_clean_patch_embedder:
            self.install_clean_patch_embedder()

    def enable_teacache(
        self,
        coefficients,
        num_steps: int,
        rel_l1_thresh: float,
        num_skip_start_steps: int = 0,
        offload: bool = True,
    ):
        """Enable TeaCache for efficient memory usage."""
        self.teacache = TeaCache(
            coefficients,
            num_steps,
            rel_l1_thresh=rel_l1_thresh,
            num_skip_start_steps=num_skip_start_steps,
            offload=offload,
        )

    def disable_teacache(self):
        """Disable TeaCache."""
        self.teacache = None

    def enable_riflex(
        self,
        k=6,
        l_test=66,
        l_test_scale=4.886,
    ):
        """Enable Riflex for positional embeddings."""
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                get_1d_rotary_pos_embed_riflex(
                    1024,
                    self.d - 4 * (self.d // 6),
                    use_real=False,
                    k=k,
                    L_test=l_test,
                    L_test_scale=l_test_scale,
                ),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
            ],
            dim=1,
        ).to(device)

    def disable_riflex(self):
        """Disable Riflex and revert positional embedding."""
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                rope_params(1024, self.d - 4 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
            ],
            dim=1,
        ).to(device)

    def enable_multi_gpus_inference(
        self,
    ):
        """Enable multi-GPU support for inference."""
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn
            )

    def _set_gradient_checkpointing(self, module, value=False):
        """Set gradient checkpointing for memory efficiency."""
        _ = module
        self.gradient_checkpointing = value

    def install_clean_patch_embedder(self):
        """Install a clean patch embedder for processing packed frames."""
        self.clean_patch_embedder = WanPatchEmbedForCleanLatents(self.in_dim, self.dim)
        self.config["has_clean_patch_embedder"] = True
        self.clean_patch_embedder.initialize_weight_from_another_conv3d(
            self.patch_embedding
        )
        print("Install Clean Patch Embedder For FramePack-Wan.")

    # Reproduced from `Inverted anti-drifting` in https://arxiv.org/abs/2504.12626
    def forward(
        self,
        x,
        t,
        context,
        latent_indices=None,
        clean_latents=None,
        clean_latent_indices=None,
        clean_latents_2x=None,
        clean_latent_2x_indices=None,
        clean_latents_4x=None,
        clean_latent_4x_indices=None,
        clip_fea=None,
        y=None,
        cond_flag=True,
    ):
        """
        Forward pass through the diffusion model.

        Args:
            x (`List[torch.Tensor]`): List of input video tensors, each with shape
                [C_in, F, H, W].
            t (`torch.Tensor`): Diffusion timesteps tensor of shape [B].
            context (`List[torch.Tensor]`): List of text embeddings each with shape
                [L, C].
            seq_len (`int`): Maximum sequence length for positional encoding.
            latent_indices (`List[torch.Tensor]`, *optional*): Latent indices for
                frequency encoding, defaults to None.
            clean_latents (`List[torch.Tensor]`, *optional*): Clean latent variables,
                each with shape [C_in, F_post, H, W], defaults to None.
            clean_latent_indices (`List[torch.Tensor]`, *optional*): Latent indices
                for clean latents, defaults to None.
            clean_latents_2x (`List[torch.Tensor]`, *optional*): Clean latent_2x variables,
                each with shape [C_in, F_2x, H, W], defaults to None.
            clean_latent_2x_indices (`List[torch.Tensor]`, *optional*): Latent indices for
                clean latents_2x, defaults to None.
            clean_latents_4x (`List[torch.Tensor]`, *optional*): Clean latent_4x variables,
                each with shape [C_in, F_4x, H, W], defaults to None.
            clean_latent_4x_indices (`List[torch.Tensor]`, *optional*): Latent indices for
                clean latents_4x, defaults to None.
            clip_fea (`torch.Tensor`, *optional*): CLIP image features for image-to-video mode.
            y (List[Tensor], *optional*): Conditional video inputs for image-to-video mode,
                same shape as x.
            cond_flag (`bool`, *optional*, defaults to True): Flag to indicate whether to
                forward the condition input.

        Returns:
            `List[torch.Tensor]`: List of denoised video tensors with original input shapes
                [C_out, F, H / 8, W / 8].
        """
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None

        # params
        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        hs = torch.stack([torch.tensor(u.shape[3], dtype=torch.long) for u in x])
        ws = torch.stack([torch.tensor(u.shape[4], dtype=torch.long) for u in x])

        x = [u.flatten(2).transpose(1, 2) for u in x]
        original_context_length = x[0].size(1)

        latent_freqs = get_indiced_rope_freqs(
            latent_indices, hs, ws, self.freqs, dim=self.d
        )
        latent_freqs = [u.permute(3, 0, 1, 2) for u in latent_freqs]
        latent_freqs = [u.flatten(1).unsqueeze(1).transpose(0, 2) for u in latent_freqs]

        if self.config["has_clean_patch_embedder"]:
            x, latent_freqs = self.process_clean_hidden_states(
                self.freqs,
                hs,
                ws,
                x,
                latent_freqs,
                clean_latents,
                clean_latent_indices,
                clean_latents_2x,
                clean_latent_2x_indices,
                clean_latents_4x,
                clean_latent_4x_indices,
            )

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

        x = torch.cat(x, dim=0)
        latent_freqs = torch.stack(latent_freqs)

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            # to bfloat16 for saving memeory
            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            e0 = e0.to(dtype)
            e = e.to(dtype)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]

        # TeaCache
        if self.teacache is not None:
            if cond_flag:
                modulated_inp = e0
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if (
                    self.teacache.cnt == 0
                    or self.teacache.cnt == self.teacache.num_steps - 1
                    or skip_flag
                ):
                    should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(
                            self.teacache.previous_modulated_input, modulated_inp
                        )
                        self.teacache.accumulated_rel_l1_distance += (
                            self.teacache.rescale_func(rel_l1_distance)
                        )
                    if (
                        self.teacache.accumulated_rel_l1_distance
                        < self.teacache.rel_l1_thresh
                    ):
                        should_calc = False
                    else:
                        should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.cnt += 1
                if self.teacache.cnt == self.teacache.num_steps:
                    self.teacache.reset()
                self.teacache.should_calc = should_calc
            else:
                should_calc = self.teacache.should_calc

        # TeaCache
        if self.teacache is not None:
            if not should_calc:
                previous_residual = (
                    self.teacache.previous_residual_cond
                    if cond_flag
                    else self.teacache.previous_residual_uncond
                )
                x = x + previous_residual.to(x.device)
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = (
                            {"use_reentrant": False}
                            if is_torch_version(">=", "1.11.0")
                            else {}
                        )
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            e0,
                            seq_lens,
                            grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            dtype,
                            **ckpt_kwargs,
                        )
                    else:
                        # arguments
                        kwargs = {
                            "e": e0,
                            "seq_lens": seq_lens,
                            "grid_sizes": grid_sizes,
                            "freqs": self.freqs,
                            "context": context,
                            "context_lens": context_lens,
                            "dtype": dtype,
                        }
                        x = block(x, **kwargs)

                    if cond_flag:
                        self.teacache.previous_residual_cond = (
                            x.cpu() - ori_x if self.teacache.offload else x - ori_x
                        )
                    else:
                        self.teacache.previous_residual_uncond = (
                            x.cpu() - ori_x if self.teacache.offload else x - ori_x
                        )
        else:
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = (
                        {"use_reentrant": False}
                        if is_torch_version(">=", "1.11.0")
                        else {}
                    )
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        e0,
                        latent_freqs,
                        seq_lens,
                        context,
                        context_lens,
                        dtype,
                        **ckpt_kwargs,
                    )
                else:
                    # arguments
                    kwargs = {
                        "e": e0,
                        "seq_lens": seq_lens,
                        "context": context,
                        "context_lens": context_lens,
                        "dtype": dtype,
                        "latent_freqs": latent_freqs,
                    }
                    x = block(x, **kwargs)

        if self.sp_world_size > 1:
            x = get_sp_group().all_gather(x, dim=1)

        # head
        x = self.head(x, e, original_context_length)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        return x

    def unpatchify(self, x, grid_sizes):
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x (`List[torch.Tensor]`): List of patchified features, each with shape
                [L, C_out * prod(patch_size)]
            grid_sizes (`torch.Tensor`): Original spatial-temporal grid dimensions
                before patching, shape [B, 3].

        Returns:
            `List[torch.Tensor]`: Reconstructed video tensors with shape
                [C_out, F, H / 8, W / 8].
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        """
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        subfolder=None,
        transformer_additional_kwargs=None,
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16,
    ):
        """
        Load a pretrained model from the given path.

        Args:
            pretrained_model_name_or_path (`str`): Path to the folder containing the
                pretrained model.
            subfolder (`str`, *optional*): Subfolder to look into for the model
                weights and configuration.
            transformer_additional_kwargs (`dict`, *optional*): Additional keyword
                arguments for transformer model.
            low_cpu_mem_usage (`bool`, *optional*): Whether to load model with lower
                memory usage on CPU.
            torch_dtype (`torch.dtype`, *optional*): The torch data type to use for
                the model's parameters.

        Returns:
            `WanTransformer3DModelPacked`: The model instance with loaded weights.
        """
        if transformer_additional_kwargs is None:
            transformer_additional_kwargs = {}
        if subfolder is not None:
            pretrained_model_name_or_path = os.path.join(
                pretrained_model_name_or_path, subfolder
            )
        print(
            f"Loaded 3D transformer's pretrained weights from {pretrained_model_name_or_path} ..."
        )

        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        model_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[
                    transformer_additional_kwargs["dict_mapping"][key]
                ] = config[key]

        if low_cpu_mem_usage:
            try:
                from diffusers.models.modeling_utils import load_model_dict_into_meta

                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    state_dict = load_file(model_file_safetensors)
                else:
                    model_files_safetensors = glob.glob(
                        os.path.join(pretrained_model_name_or_path, "*.safetensors")
                    )
                    state_dict = {}
                    print(model_files_safetensors)
                    for _model_file_safetensors in model_files_safetensors:
                        _state_dict = load_file(_model_file_safetensors)
                        for key, value in _state_dict.items():
                            state_dict[key] = value
                model._convert_deprecated_attention_blocks(state_dict)
                # move the params from meta device to cpu
                missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                if len(missing_keys) > 0:
                    raise ValueError(
                        f"Cannot load {cls} from {pretrained_model_name_or_path}, "
                        f"because the following missing keys: \n {', '.join(missing_keys)}"
                        f"Please make sure `low_cpu_mem_usage=False` and `device_map=None`."
                    )

                unexpected_keys = load_model_dict_into_meta(
                    model,
                    state_dict,
                    device=param_device,
                    dtype=torch_dtype,
                    model_name_or_path=pretrained_model_name_or_path,
                )

                if cls._keys_to_ignore_on_load_unexpected is not None:
                    for pat in cls._keys_to_ignore_on_load_unexpected:
                        unexpected_keys = [
                            k for k in unexpected_keys if re.search(pat, k) is None
                        ]

                if len(unexpected_keys) > 0:
                    print(f"Unexpected keys: \n {[', '.join(unexpected_keys)]}")
                return model
            except Exception as e:
                print(
                    f"The low_cpu_mem_usage mode is not work because {e}."
                    f"Use low_cpu_mem_usage=False instead."
                )

        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            state_dict = load_file(model_file_safetensors)
        else:
            model_files_safetensors = glob.glob(
                os.path.join(pretrained_model_name_or_path, "*.safetensors")
            )
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key, value in _state_dict.items():
                    state_dict[key] = value

        if (
            model.state_dict()["patch_embedding.weight"].size()
            != state_dict["patch_embedding.weight"].size()
        ):
            model.state_dict()["patch_embedding.weight"][
                :, : state_dict["patch_embedding.weight"].size()[1], :, :
            ] = state_dict["patch_embedding.weight"]
            model.state_dict()["patch_embedding.weight"][
                :, state_dict["patch_embedding.weight"].size()[1] :, :, :
            ] = 0
            state_dict["patch_embedding.weight"] = model.state_dict()[
                "patch_embedding.weight"
            ]

        tmp_state_dict = {}
        for key in state_dict:
            if (
                key in model.state_dict().keys()
                and model.state_dict()[key].size() == state_dict[key].size()
            ):
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")

        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)

        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [
            p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()
        ]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")

        model = model.to(torch_dtype)
        return model

    # Reproduced from `Inverted anti-drifting` in https://arxiv.org/abs/2504.12626
    def process_clean_hidden_states(
        self,
        freqs,
        hs,
        ws,
        latents,
        latent_freqs,
        clean_latents=None,
        clean_latent_indices=None,
        clean_latents_2x=None,
        clean_latent_2x_indices=None,
        clean_latents_4x=None,
        clean_latent_4x_indices=None,
    ):
        """
        Processes clean hidden states and returns updated latents and latent frequencies.

        Args:
            freqs (`List[torch.Tensor]`): List of frequency tensors for RoPE encoding,
                each with shape [L, H, W, D].
            hs (`torch.Tensor`): Height tensors, shape [B].
            ws (`torch.Tensor`): Width tensors, shape [B].
            latents (`List[torch.Tensor]`): List of latent tensors, each with shape
                [C_in, F, H, W].
            latent_freqs (`List[torch.Tensor]`): List of latent frequency tensors,
                each with shape [L_in, 1, D].
            clean_latents (`List[torch.Tensor]`, *optional*): List of clean latent tensors,
                each with shape [C_in, F_post, H, W], defaults to None.
            clean_latent_indices (`List[torch.Tensor]`, *optional*): Indices for clean latents,
                default is None.
            clean_latents_2x (`List[torch.Tensor]`, *optional*): List of clean latent_2x tensors,
                each with shape [C_in, F_2x, H, W], defaults to None.
            clean_latent_2x_indices (`List[torch.Tensor]`, *optional*): Indices for clean
                latents_2x, default is None.
            clean_latents_4x (`List[torch.Tensor]`, *optional*): List of clean latent_4x tensors,
                each with shape [C_in, F_4x, H, W], defaults to None.
            clean_latent_4x_indices (`List[torch.Tensor]`, *optional*): Indices for clean
                latents_4x, default is None.

        Returns:
            `Tuple[List[torch.Tensor], List[torch.Tensor]]`: List of updated latents,
                each shape with [1, L_total, C_in], and list of updated latent frequencies,
                each with shape [L_total, 1, D].
        """

        def get_complex_with_cos_sin(tensor_list, dim=64):
            complex_tensor_list = []
            for tensor in tensor_list:
                cos_part = tensor[:, :, :dim]
                sin_part = tensor[:, :, dim:]
                complex_tensor = torch.complex(cos_part, sin_part)
                complex_tensor_list.append(complex_tensor)
            return complex_tensor_list

        if clean_latents is not None and clean_latent_indices is not None:
            clean_latents = [
                self.clean_patch_embedder.proj(u.unsqueeze(0)) for u in clean_latents
            ]
            clean_latents = [u.flatten(2).transpose(1, 2) for u in clean_latents]

            clean_freqs = get_indiced_rope_freqs(
                clean_latent_indices, hs, ws, freqs, dim=self.d
            )
            clean_freqs = [u.permute(3, 0, 1, 2) for u in clean_freqs]
            clean_freqs = [
                u.flatten(1).unsqueeze(1).transpose(0, 2) for u in clean_freqs
            ]

            latents = [torch.cat([u, v], dim=1) for u, v in zip(latents, clean_latents)]
            latent_freqs = [
                torch.cat([u, v], dim=0) for u, v in zip(latent_freqs, clean_freqs)
            ]

        if clean_latents_2x is not None and clean_latent_2x_indices is not None:
            clean_latents_2x = [
                pad_for_3d_conv(u.unsqueeze(0), (2, 4, 4)) for u in clean_latents_2x
            ]
            clean_latents_2x = [
                self.clean_patch_embedder.proj_2x(u) for u in clean_latents_2x
            ]
            clean_latents_2x = [u.flatten(2).transpose(1, 2) for u in clean_latents_2x]

            clean_freqs_2x = get_indiced_rope_freqs(
                clean_latent_2x_indices, hs, ws, freqs, dim=self.d
            )
            clean_freqs_2x = [u.permute(3, 0, 1, 2) for u in clean_freqs_2x]
            clean_freqs_2x = [
                pad_for_3d_conv(u.unsqueeze(0), (2, 2, 2)) for u in clean_freqs_2x
            ]
            clean_freqs_2x = [
                torch.cat([u.real, u.imag], dim=1) for u in clean_freqs_2x
            ]
            clean_freqs_2x = [
                center_down_sample_3d(u, (2, 2, 2)) for u in clean_freqs_2x
            ]
            clean_freqs_2x = [u.flatten(2).transpose(1, 2) for u in clean_freqs_2x]
            clean_freqs_2x = get_complex_with_cos_sin(clean_freqs_2x, dim=64)
            clean_freqs_2x = [u.permute(1, 0, 2) for u in clean_freqs_2x]

            latents = [
                torch.cat([u, v], dim=1) for u, v in zip(latents, clean_latents_2x)
            ]
            latent_freqs = [
                torch.cat([u, v], dim=0) for u, v in zip(latent_freqs, clean_freqs_2x)
            ]

        if clean_latents_4x is not None and clean_latent_4x_indices is not None:
            clean_latents_4x = [
                pad_for_3d_conv(u.unsqueeze(0), (4, 8, 8)) for u in clean_latents_4x
            ]
            clean_latents_4x = [
                self.clean_patch_embedder.proj_4x(u) for u in clean_latents_4x
            ]
            clean_latents_4x = [u.flatten(2).transpose(1, 2) for u in clean_latents_4x]

            clean_freqs_4x = get_indiced_rope_freqs(
                clean_latent_4x_indices, hs, ws, freqs, dim=self.d
            )
            clean_freqs_4x = [u.permute(3, 0, 1, 2) for u in clean_freqs_4x]
            clean_freqs_4x = [
                pad_for_3d_conv(u.unsqueeze(0), (4, 4, 4)) for u in clean_freqs_4x
            ]
            clean_freqs_4x = [
                torch.cat([u.real, u.imag], dim=1) for u in clean_freqs_4x
            ]
            clean_freqs_4x = [
                center_down_sample_3d(u, (4, 4, 4)) for u in clean_freqs_4x
            ]
            clean_freqs_4x = [u.flatten(2).transpose(1, 2) for u in clean_freqs_4x]
            clean_freqs_4x = get_complex_with_cos_sin(clean_freqs_4x, dim=64)
            clean_freqs_4x = [u.permute(1, 0, 2) for u in clean_freqs_4x]

            latents = [
                torch.cat([u, v], dim=1) for u, v in zip(latents, clean_latents_4x)
            ]
            latent_freqs = [
                torch.cat([u, v], dim=0) for u, v in zip(latent_freqs, clean_freqs_4x)
            ]

        return latents, latent_freqs
