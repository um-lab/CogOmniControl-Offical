# Copyright 2024-2025. Add-on module (does NOT modify any existing source file).
"""
Control/Ref latent attention dropout for `CogOmniControlWanModel_Connector`.

Motivation
----------
In `WanSelfAttention.forward` (see `modeling_cogomni_control_connector.py`), the
self-attention sequence for every sample is laid out as:

    x = [ Noisy | Control | Ref | LLM_embeds | Padding ]

This module implements an *attention-level* dropout: with some probability `p`,
we make every query (noisy latent / control / ref / llm) unable to attend to
the Control (and optionally Ref) key/value tokens of that same sample, i.e. we
remove those tokens from the Key/Value sequence used by flash-attention. This
is functionally equivalent to zeroing the corresponding rows/cols of the
attention mask for the Control/Ref block, while keeping the tensor layout
(and therefore RoPE, residual-add, shapes, ...) completely untouched.

Because the underlying attention op is `flash_attn_varlen_func` (see
`modular_cogomni_control.flash_attention`), there is no explicit dense
attention-mask argument -- masking is expressed purely through
`k_lens` (a *contiguous prefix* length per sample). So to "drop" a
*middle* segment (Control/Ref) we:
    1. gather the kept segments (Noisy [+ Control] [+ Ref] + LLM) for the
       samples that are selected for dropout, writing them contiguously to
       the front of the K/V tensor (this does NOT touch Q, so query rows -
       i.e. the model's output positions - are completely unaffected);
    2. shrink `k_lens` for those samples accordingly.

Sequence-Parallel (SP) correctness
-----------------------------------
This model shards the sequence dimension across SP ranks. Inside
`WanSelfAttention.forward`, right before calling `flash_attention`, an
`all_to_all_4D` has already been performed so every SP rank locally holds the
**full** sequence (with only a shard of attention heads) -- this is exactly
the point where our dropout logic hooks in, so it is naturally SP-shape
compatible.

However, the *decision* of which samples get dropped must be **identical
across every rank inside the same SP group**, otherwise different ranks would
truncate the K/V sequence at different lengths for the "same" sample and the
attention output re-assembled via all_to_all would be corrupted. We therefore:
    * generate the random mask once per top-level `forward()` call (via a
      `forward_pre_hook` on the transformer, so it is shared by *all* blocks/
      layers in that pass, and reused verbatim if gradient checkpointing
      recomputes the forward during backward);
    * only sample on the SP-group's rank 0 and `dist.broadcast` the decision
      to the rest of the group, so every rank agrees on the same mask.

Usage (no source files are modified)
-------------------------------------
    from videox_fun.models.cogomni_control.control_ref_attention_dropout import (
        enable_control_ref_attention_dropout,
    )

    transformer3d = CogOmniControlWanModel_Connector(...)
    ...
    enable_control_ref_attention_dropout(
        transformer3d,
        drop_control_prob=0.1,   # probability of dropping the Control latent
        drop_ref_prob=0.1,       # probability of dropping the Ref latent
        drop_together=True,      # True: control & ref share ONE coin flip
                                  # False: independent coin flips
    )

    # ... run training as usual, e.g.:
    # model_pred = transformer3d(x, t, context, seq_len, control_latents, ...)

    # to turn if off again (e.g. for eval/inference):
    # disable_control_ref_attention_dropout(transformer3d)

The dropout is automatically skipped when `transformer3d.training` is False,
so you do not need to manually disable it before `.eval()` inference, but you
may still call `disable_control_ref_attention_dropout` to fully unpatch.
"""

import types
import weakref

import torch
import torch.distributed as dist

from fastvideo.utils.communications import all_to_all_4D
from fastvideo.utils.parallel_states import get_sequence_parallel_state, nccl_info

from .modeling_cogomni_control_connector import rope_apply, rope_apply_dist
from .modular_cogomni_control import flash_attention

__all__ = [
    "enable_control_ref_attention_dropout",
    "disable_control_ref_attention_dropout",
]

_CFG_ATTR = "_control_ref_dropout_cfg"
_HOOK_ATTR = "_control_ref_dropout_hook_handle"
_MASKS_ATTR = "_control_ref_dropout_current_masks"
_ORIG_FORWARD_ATTR = "_control_ref_dropout_orig_forward"
_PARENT_REF_ATTR = "_control_ref_dropout_parent_ref"


def _sample_broadcast_mask(batch_size, drop_prob, device):
    """Sample a bool[batch_size] dropout mask, identical across all ranks in
    the current SP group (if any)."""
    if drop_prob <= 0:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    if get_sequence_parallel_state() and dist.is_available() and dist.is_initialized():
        if nccl_info.rank_within_group == 0:
            mask_u8 = (torch.rand(batch_size, device=device) < drop_prob).to(torch.uint8)
        else:
            mask_u8 = torch.zeros(batch_size, dtype=torch.uint8, device=device)
        src = nccl_info.group_id * nccl_info.sp_size
        dist.broadcast(mask_u8, src=src, group=nccl_info.group)
        return mask_u8.bool()

    return torch.rand(batch_size, device=device) < drop_prob


def _make_pre_hook(cfg):
    """Forward-pre-hook registered on the top-level transformer. Samples the
    per-sample drop masks ONCE per forward() call and stashes them on the
    module, so every block/layer inside this single forward pass (and any
    gradient-checkpointing recompute of it) reuses the exact same decision.
    """

    def _hook(module, args, kwargs=None):
        x = args[0] if len(args) > 0 else kwargs["x"]
        batch_size = len(x)
        device = x[0].device

        if cfg["drop_together"]:
            shared_mask = _sample_broadcast_mask(batch_size, cfg["drop_control_prob"], device)
            drop_control_mask = shared_mask
            drop_ref_mask = shared_mask
        else:
            drop_control_mask = _sample_broadcast_mask(batch_size, cfg["drop_control_prob"], device)
            drop_ref_mask = _sample_broadcast_mask(batch_size, cfg["drop_ref_prob"], device)

        # ---- debug: print ONCE on the main rank whenever dropout is applied ----
        dbg_step = getattr(module, "_control_ref_dropout_dbg_step", 0) + 1
        setattr(module, "_control_ref_dropout_dbg_step", dbg_step)
        if bool(drop_control_mask.any()) or bool(drop_ref_mask.any()):
            is_main = (
                (not dist.is_available())
                or (not dist.is_initialized())
                or (dist.get_rank() == 0)
            )
            if is_main:
                print(
                    f"[control_ref_dropout] forward_step={dbg_step} ",
                    flush=True,
                )
        # ------------------------------------------------------------------------

        setattr(module, _MASKS_ATTR, (drop_control_mask, drop_ref_mask))
        return None

    return _hook


def _drop_control_ref_from_kv(k, v, seq_lens, grid_sizes, control_grid_sizes,
                              ref_grid_sizes, drop_control_mask, drop_ref_mask):
    """
    k, v: [B, S, H_local, D] -- FULL sequence length (already all_to_all'ed in
          the SP case), local shard of attention heads.
    Returns possibly-modified (k, v, seq_lens) where, for the selected
    samples, the Control/Ref key-value tokens have been removed (the kept
    tokens are moved to the contiguous front of that sample's row, matching
    what `flash_attention`'s `k_lens`-based prefix truncation expects), and
    `seq_lens` shrunk accordingly. Samples that are not selected for dropout
    are left completely untouched.
    """
    batch_size = k.shape[0]

    if not bool(drop_control_mask.any()) and not (
        drop_ref_mask is not None and bool(drop_ref_mask.any())
    ):
        return k, v, seq_lens

    new_k = k.clone()
    new_v = v.clone()
    new_seq_lens = seq_lens.clone()

    for i in range(batch_size):
        drop_c = bool(drop_control_mask[i])
        drop_r = bool(drop_ref_mask[i]) if drop_ref_mask is not None else False
        if not drop_c and not drop_r:
            continue

        f, h, w = grid_sizes[i].tolist()
        noisy_len = f * h * w

        cf, ch, cw = control_grid_sizes[i].tolist()
        control_len = cf * ch * cw

        ref_len = 0
        if ref_grid_sizes is not None:
            rf, rh, rw = ref_grid_sizes[i].tolist()
            ref_len = rf * rh * rw
        else:
            drop_r = False

        total_real = int(seq_lens[i].item())
        llm_len = max(total_real - noisy_len - control_len - ref_len, 0)

        cursor = noisy_len
        control_seg = (cursor, cursor + control_len)
        cursor += control_len
        ref_seg = (cursor, cursor + ref_len)
        cursor += ref_len
        llm_seg = (cursor, cursor + llm_len)

        segments = [(0, noisy_len)]
        if not drop_c:
            segments.append(control_seg)
        if not drop_r:
            segments.append(ref_seg)
        segments.append(llm_seg)

        keep_idx = torch.cat([
            torch.arange(s, e, device=k.device, dtype=torch.long)
            for s, e in segments if e > s
        ])
        new_len = keep_idx.numel()

        new_k[i, :new_len] = k[i, keep_idx]
        new_v[i, :new_len] = v[i, keep_idx]
        if new_len < k.shape[1]:
            new_k[i, new_len:] = 0
            new_v[i, new_len:] = 0
        new_seq_lens[i] = new_len

    return new_k, new_v, new_seq_lens


def _patched_self_attn_forward(
    self,
    x,
    seq_lens,
    grid_sizes,
    control_grid_sizes,
    ref_grid_sizes,
    freqs,
    dtype=torch.bfloat16,
    llm_embeds_grid_sizes=None,
):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
    k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
    v = self.v(x.to(dtype)).view(b, s, n, d)

    if get_sequence_parallel_state():
        q = rope_apply_dist(q, grid_sizes, control_grid_sizes, ref_grid_sizes, freqs,
                            llm_embeds_grid_sizes=llm_embeds_grid_sizes)
        k = rope_apply_dist(k, grid_sizes, control_grid_sizes, ref_grid_sizes, freqs,
                            llm_embeds_grid_sizes=llm_embeds_grid_sizes)
    else:
        q = rope_apply(q, grid_sizes, control_grid_sizes, ref_grid_sizes, freqs,
                       llm_embeds_grid_sizes=llm_embeds_grid_sizes)
        k = rope_apply(k, grid_sizes, control_grid_sizes, ref_grid_sizes, freqs,
                       llm_embeds_grid_sizes=llm_embeds_grid_sizes)

    if get_sequence_parallel_state():
        q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
        k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
        v = all_to_all_4D(v, scatter_dim=2, gather_dim=1)

    # ---- control/ref attention dropout (new logic, no source files touched) ----
    effective_seq_lens = seq_lens
    if self.training:
        parent_ref = getattr(self, _PARENT_REF_ATTR, None)
        parent = parent_ref() if parent_ref is not None else None
        masks = getattr(parent, _MASKS_ATTR, None) if parent is not None else None
        if masks is not None:
            drop_control_mask, drop_ref_mask = masks
            k, v, effective_seq_lens = _drop_control_ref_from_kv(
                k, v, seq_lens, grid_sizes, control_grid_sizes, ref_grid_sizes,
                drop_control_mask, drop_ref_mask,
            )
    # -------------------------------------------------------------------------

    x = flash_attention(
        q=q.to(dtype),
        k=k.to(dtype),
        v=v.to(dtype),
        k_lens=effective_seq_lens,
        window_size=self.window_size)

    if get_sequence_parallel_state():
        x = all_to_all_4D(x, scatter_dim=1, gather_dim=2)

    x = x.to(dtype)
    # output
    x = x.flatten(2)
    x = self.o(x)

    return x


def enable_control_ref_attention_dropout(
    transformer,
    drop_prob=0.0,
    drop_control_prob=None,
    drop_ref_prob=None,
    drop_together=True,
):
    """Patch `transformer` (an instance of `CogOmniControlWanModel_Connector`,
    or any model exposing a `.blocks` ModuleList of `WanAttentionBlock`) so
    that every `self_attn` inside it randomly drops the Control/Ref latent
    tokens from the Key/Value sequence during self-attention.

    Args:
        transformer: the top-level Wan transformer3d model instance.
        drop_prob: default probability used for both control/ref when the
            more specific args below are not given.
        drop_control_prob: probability of dropping the Control latent.
            Defaults to `drop_prob`.
        drop_ref_prob: probability of dropping the Ref latent.
            Defaults to `drop_prob`.
        drop_together: if True, Control and Ref share a single Bernoulli
            draw per sample (drop_control_prob is used, drop_ref_prob is
            ignored) -- i.e. "with probability p, ignore both Control and Ref
            latents together". If False, Control and Ref are dropped with
            independent coin flips using their own probabilities.

    This only patches the given `transformer` instance's `self_attn`
    submodules (via per-instance bound-method overriding), so other model
    instances (e.g. a separately-loaded high/low-noise transformer) are not
    affected.
    """
    if drop_control_prob is None:
        drop_control_prob = drop_prob
    if drop_ref_prob is None:
        drop_ref_prob = drop_prob

    cfg = dict(
        drop_control_prob=float(drop_control_prob),
        drop_ref_prob=float(drop_ref_prob),
        drop_together=bool(drop_together),
    )
    setattr(transformer, _CFG_ATTR, cfg)

    # Remove a previously-registered hook if enable() is called twice.
    old_handle = getattr(transformer, _HOOK_ATTR, None)
    if old_handle is not None:
        old_handle.remove()

    hook_handle = transformer.register_forward_pre_hook(
        _make_pre_hook(cfg), with_kwargs=True
    )
    setattr(transformer, _HOOK_ATTR, hook_handle)

    parent_ref = weakref.ref(transformer)
    for block in transformer.blocks:
        self_attn = block.self_attn
        if not hasattr(self_attn, _ORIG_FORWARD_ATTR):
            setattr(self_attn, _ORIG_FORWARD_ATTR, self_attn.forward)
        setattr(self_attn, _PARENT_REF_ATTR, parent_ref)
        self_attn.forward = types.MethodType(_patched_self_attn_forward, self_attn)

    return transformer


def disable_control_ref_attention_dropout(transformer):
    """Undo `enable_control_ref_attention_dropout`: restores every
    `self_attn.forward` to its original implementation and removes the
    forward_pre_hook."""
    hook_handle = getattr(transformer, _HOOK_ATTR, None)
    if hook_handle is not None:
        hook_handle.remove()
        delattr(transformer, _HOOK_ATTR)

    for block in transformer.blocks:
        self_attn = block.self_attn
        orig_forward = getattr(self_attn, _ORIG_FORWARD_ATTR, None)
        if orig_forward is not None:
            self_attn.forward = orig_forward
            delattr(self_attn, _ORIG_FORWARD_ATTR)
        if hasattr(self_attn, _PARENT_REF_ATTR):
            delattr(self_attn, _PARENT_REF_ATTR)

    if hasattr(transformer, _MASKS_ATTR):
        delattr(transformer, _MASKS_ATTR)
    if hasattr(transformer, _CFG_ATTR):
        delattr(transformer, _CFG_ATTR)

    return transformer
