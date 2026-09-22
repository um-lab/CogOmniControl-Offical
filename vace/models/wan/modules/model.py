# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.utils import is_torch_version
from wan.modules.model import WanModel, WanAttentionBlock, sinusoidal_embedding_1d

from fastvideo.utils.communications import all_gather
from fastvideo.utils.parallel_states import nccl_info


class VaceWanAttentionBlock(WanAttentionBlock):
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
            block_id=0
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)
        
    def forward(
        self,
        c,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):

        if self.block_id == 0:
            c = self.before_proj(c) + x

        c = super().forward(c, e, seq_lens, grid_sizes, freqs, context, context_lens)
        c_skip = self.after_proj(c)

        return c, c_skip
    
    
class BaseWanAttentionBlock(WanAttentionBlock):
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
        block_id=None
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(
        self,
        x,
        hint,
        context_scale,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        x = super().forward(x, e, seq_lens, grid_sizes, freqs, context, context_lens)

        if hint is not None:
            x = x + hint * context_scale
        return x
    
    
class VaceWanModel(WanModel):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 vace_layers=None,
                 vace_in_dim=None,
                 model_type='t2v',
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
                 eps=1e-6):
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim, freq_dim, text_dim, out_dim,
                         num_heads, num_layers, window_size, qk_norm, cross_attn_norm, eps)

        self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # blocks
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                  self.cross_attn_norm, self.eps,
                                  block_id=self.vace_layers_mapping[i] if i in self.vace_layers else None)
            for i in range(self.num_layers)
        ])

        # vace blocks
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                     self.cross_attn_norm, self.eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward_vace(
        self,
        x,
        vace_context,
        seq_len,
        kwargs
    ):
        # embeddings
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        # squence parrallel chunk
        c = torch.chunk(c, nccl_info.sp_size, dim=1)[nccl_info.rank_within_group]
        
        hints = []
        for block in self.vace_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                x = new_kwargs["x"]
                e0 = new_kwargs["e"]
                seq_lens = new_kwargs["seq_lens"]
                grid_sizes = new_kwargs["grid_sizes"]
                freqs = new_kwargs["freqs"]
                context = new_kwargs["context"]
                context_lens = new_kwargs["context_lens"]

                c, c_skip = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    c,
                    x,
                    e0,
                    seq_lens,
                    grid_sizes,
                    freqs,
                    context,
                    context_lens,
                    **ckpt_kwargs,
                )
            else:
                c, c_skip = block(c, **new_kwargs)
                
            hints.append(c_skip)
        return hints

    def forward(
        self,
        x,
        t,
        vace_context,
        context,
        seq_len,
        vace_context_scale=1.0,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        # squence parrallel chunk
        x = torch.chunk(x, nccl_info.sp_size, dim=1)[nccl_info.rank_within_group]

        hints = self.forward_vace(x, vace_context, seq_len, kwargs)

        for block in self.blocks:
            block_id = block.block_id
            if block_id is not None:
                hint = hints[block_id]
            else:
                hint = None

            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward
                ckpt_kwargs = {"use_reentrant": True} if is_torch_version(">=", "1.11.0") else {}

                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    hint,
                    vace_context_scale,
                    e0,
                    seq_lens,
                    grid_sizes,
                    self.freqs,
                    context,
                    context_lens,
                    **ckpt_kwargs,
                )
            else:
                x = block(x, hint, vace_context_scale, e0, seq_lens, grid_sizes, self.freqs, context, context_lens)

        # squence parralel all_gather
        x = all_gather(x, dim=1).contiguous()

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        # return [u.float() for u in x]
        x = torch.stack(x)
        return x


    def load_state_dict_from_single_file(self, ckpt_path, shield_incompatible=False, load_from_wan_base=False):
        import json
        import os

        from safetensors.torch import load_file
        from tqdm import tqdm

        print(f"From checkpoint: {ckpt_path}")
        if os.path.isdir(ckpt_path):
            if os.path.exists(os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors")):
                state_dict = load_file(os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors"))
            else:
                index_file_path = os.path.join(ckpt_path, "diffusion_pytorch_model.safetensors.index.json")
                assert os.path.exists(index_file_path)
                with open(index_file_path, "r") as fin:
                    weight_map = json.load(fin)["weight_map"]
                shard_names = sorted(list(set(weight_map.values())))
                shard_weight_maps = {}
                for shard_name in tqdm(shard_names, total=len(shard_names)):
                    shard_path = os.path.join(ckpt_path, shard_name)
                    shard_state_dict = load_file(shard_path)
                    shard_weight_maps[shard_name] = shard_state_dict
                state_dict = {}
                for weight_name, weight_shard_name in weight_map.items():
                    state_dict[weight_name] = shard_weight_maps[weight_shard_name][weight_name]
        elif ckpt_path.endswith("safetensors"):
            state_dict = load_file(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = self.load_state_dict(state_dict, strict=False)
        if not shield_incompatible:
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
            assert len(u) == 0 
        if load_from_wan_base:
            print(f"Load vace blocks state dict from {self.vace_layers}")
            blocks = [self.blocks[idx] for idx in self.vace_layers]
            for idx in range(len(blocks)):
                self.vace_blocks[idx].load_state_dict(blocks[idx].state_dict(), strict=False)
