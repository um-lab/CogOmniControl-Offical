#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, List, Optional

import numpy as np


# see https://github.com/continue-revolution/sd-webui-animatediff/blob/2573b2b51369e65d92b80a57874b41fd7344bae9/
# scripts/animatediff_infv2v.py#L32
# 0 / 2^0 = 1.0
# 1 / 2^1 = 0.5
# 2 / 2^3 = 0.25
# 3 / 2^2 = 0.75
# 4 / 2^4 = 0.125
# 5 / 2^3 = 0.625
# 6 / 2^4 = 0.375
# 7 / 2^3 = 0.875
# 8 / 2^7 = 0.0625
# 9 / 2^4 = 0.5625
# Returns fraction that has denominator that is a power of 2
def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)

    return as_int / (1 << 64)


def uniform(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_overlap: int = 4,
    context_stride: int = 1,
    closed_loop: bool = True,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [e % num_frames for e in range(j, j + context_size * context_step, context_step)]


def shuffle(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_overlap: int = 4,
    context_stride: int = 1,
    closed_loop: bool = True,
):
    import random

    c = list(range(num_frames))
    c = random.sample(c, len(c))

    if len(c) % context_size:
        c += c[0 : context_size - len(c) % context_size]

    c = random.sample(c, len(c))

    for i in range(0, len(c), context_size):
        yield c[i : i + context_size]


def composite(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_overlap: int = 4,
    context_stride: int = 1,
    closed_loop: bool = True,
):
    if (step / num_steps) < 0.1:
        return shuffle(step, num_steps, num_frames, context_size, context_stride, context_overlap, closed_loop)
    else:
        return uniform(step, num_steps, num_frames, context_size, context_stride, context_overlap, closed_loop)


def sequential(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_overlap: int = 4,
    context_stride: int = 1,
    closed_loop: bool = True,
):
    visited_context = set()
    for context in uniform(0, num_steps, num_frames, context_size, context_overlap, context_stride, closed_loop):
        anchor_idx = None
        for idx in range(0, len(context) - 1):
            if context[idx] > context[idx + 1]:
                anchor_idx = idx
                break
        if anchor_idx is not None:
            anchor_num = context[anchor_idx]
            context_step = float("inf")
            for idx in range(0, len(context) - 1):
                if 0 < context[idx + 1] - context[idx] < context_step:
                    context_step = context[idx + 1] - context[idx]
            for idx in range(len(context) - 1, -1, -1):
                context[idx] = anchor_num
                anchor_num -= context_step
                anchor_num = max(anchor_num, 0)

        if tuple(context) in visited_context:
            continue
        yield context
        visited_context.add(tuple(context))


def get_context_scheduler(name: str) -> Callable:
    if name == "uniform":
        return uniform
    elif name == "shuffle":
        return shuffle
    elif name == "composite":
        return composite
    elif name == "sequential":
        return sequential
    else:
        raise ValueError(f"Unknown context_overlap policy {name}")


def get_total_steps(
    scheduler,
    timesteps: List[int],
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_overlap: int = 4,
    context_stride: int = 1,
    closed_loop: bool = True,
):
    return sum(
        len(
            list(
                scheduler(
                    i,
                    num_steps,
                    num_frames,
                    context_size,
                    context_overlap,
                    context_stride,
                    closed_loop,
                )
            )
        )
        for i in range(len(timesteps))
    )
