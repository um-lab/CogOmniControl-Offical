"""
Encoding and packaging frames follow Invert anti-drifting in
https://arxiv.org/pdf/2504.12626
"""

import random
import torch


def encoding_and_packaging_frames(
    cache_latents,
    latent_window_size=9,
    clean_latents_post_size=1,
    clean_latents_2x_size=2,
    clean_latents_4x_size=16,
):
    """
    Encodes and packages cache latents following the Inverted anti-drifting sampling
    in https://arxiv.org/pdf/2504.12626.

    Args:
        cache_latents (`torch.Tensor`):
            Cached video latents with shape (B, C, T, H, W).
        latent_window_size (`int`, *optional*):
            Size of the latent window, defaults to 9.
        clean_latents_post_size (`int`, *optional*):
            Size of clean_latents_post, defaults to 1.
        clean_latents_2x_size (`int`, *optional*):
            Size of clean_latents_2x, defaults to 2.
        clean_latents_4x_size (`int`, *optional*):
            Size of clean_latents_4x, defaults to 16.

    Returns:
        Tuple:
            - latents (`torch.Tensor`):
                Target latent frames for training with shape (B, C, latent_window_size, H, W).
            - latent_indices (`torch.Tensor`):
                Indices for the latent frames with shape (B, latent_window_size).
            - clean_latents (`torch.Tensor`):
                Clean_latents_post frames with shape (B, C, clean_latents_post_size, H, W).
            - clean_latent_indices (`torch.Tensor`):
                Indices for the clean_latents_post with shape (B, clean_latents_post_size).
            - clean_latents_2x (`torch.Tensor`):
                Clean_latents_2x frames with shape (B, C, clean_latents_2x_size, H, W).
            - clean_latent_2x_indices (`torch.Tensor`):
                Indices for the clean_latents_2x with shape (B, clean_latents_2x_size).
            - clean_latents_4x (`torch.Tensor`):
                Clean_latents_4x frames with shape (B, C, clean_latents_4x_size, H, W).
            - clean_latent_4x_indices (`torch.Tensor`):
                Indices for the clean_latents_4x with shape (B, clean_latents_4x_size).
            - seq_len (`int`): The final sequence length after padding and processing.
    """
    clean_latents_pre_size = 1
    clean_size = clean_latents_4x_size + clean_latents_2x_size + clean_latents_post_size

    b, c, t, h, w = cache_latents.shape
    if t < latent_window_size:
        raise ValueError(
            f"`Cache_latents has {t} latent frames, "
            f"less than latent_window_size which is {latent_window_size}."
        )

    latent_start_indice = random.randint(0, t - latent_window_size)
    # pre - blank - latent - post - 2x - 4x - pad
    if latent_start_indice != 0:
        pad_length = latent_start_indice + latent_window_size + clean_size - t
        if pad_length > 0:
            cache_latents = torch.nn.functional.pad(
                cache_latents, (0, 0, 0, 0, 0, pad_length), mode="constant", value=0
            )
            seq_len = t + pad_length
            pad_length = 0
        else:
            pad_length = -pad_length
            seq_len = t
        assert seq_len == cache_latents.size(2)

        indices = torch.arange(1, seq_len + 1).unsqueeze(0).repeat(b, 1)
        blank_size = latent_start_indice - clean_latents_pre_size

        (
            clean_latent_pre_indices,
            _,
            latent_indices,
            clean_latent_post_indices,
            clean_latent_2x_indices,
            clean_latent_4x_indices,
            _,
        ) = torch.split(
            indices,
            [
                clean_latents_pre_size,
                blank_size,
                latent_window_size,
                clean_latents_post_size,
                clean_latents_2x_size,
                clean_latents_4x_size,
                pad_length,
            ],
            dim=-1,
        )

        (
            clean_latents_pre,
            _,
            latents,
            clean_latents_post,
            clean_latents_2x,
            clean_latents_4x,
            _,
        ) = torch.split(
            cache_latents,
            [
                clean_latents_pre_size,
                blank_size,
                latent_window_size,
                clean_latents_post_size,
                clean_latents_2x_size,
                clean_latents_4x_size,
                pad_length,
            ],
            dim=2,
        )
    # (pre) - latent_pre - blank - latent_post - post - 2x - 4x
    else:
        indices = torch.arange(1, t + clean_size + 1).unsqueeze(0).repeat(b, 1)
        blank_size = t - latent_window_size
        pad_length = clean_size
        cache_latents = torch.nn.functional.pad(
            cache_latents, (0, 0, 0, 0, 0, pad_length), mode="constant", value=0
        )

        (
            latent_indices_pre,
            _,
            latent_indices_post,
            clean_latent_post_indices,
            clean_latent_2x_indices,
            clean_latent_4x_indices,
        ) = torch.split(
            indices,
            [
                1,
                blank_size,
                latent_window_size - 1,
                clean_latents_post_size,
                clean_latents_2x_size,
                clean_latents_4x_size,
            ],
            dim=-1,
        )
        latent_indices = torch.cat([latent_indices_pre, latent_indices_post], dim=-1)

        (
            latents_pre,
            _,
            latents_post,
            clean_latents_post,
            clean_latents_2x,
            clean_latents_4x,
        ) = torch.split(
            cache_latents,
            [
                1,
                blank_size,
                latent_window_size - 1,
                clean_latents_post_size,
                clean_latents_2x_size,
                clean_latents_4x_size,
            ],
            dim=2,
        )
        latents = torch.cat([latents_pre, latents_post], dim=2)

        clean_latent_pre_indices = torch.tensor([0]).unsqueeze(0).repeat(b, 1)
        clean_latents_pre = torch.zeros(b, c, clean_latents_pre_size, h, w).to(
            device=cache_latents.device, dtype=cache_latents.dtype
        )

    clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
    clean_latent_indices = torch.cat(
        [clean_latent_pre_indices, clean_latent_post_indices], dim=-1
    )

    seq_len = clean_latents_pre_size + latent_window_size + clean_size
    return (
        latents,
        latent_indices,
        clean_latents,
        clean_latent_indices,
        clean_latents_2x,
        clean_latent_2x_indices,
        clean_latents_4x,
        clean_latent_4x_indices,
        seq_len,
    )
