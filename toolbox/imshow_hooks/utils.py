from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib.axes._axes import Axes

from .draw_box import DrawBox
from .line_profile import LineProfile
from .metrics import MetricScores
from .volume_metrics import VolumeMetricScores

HOOK_LIST = [
    "DrawBox",
    "MetricScores",
    "VolumeMetricScores",
    "LineProfile",
]


def set_hook_fns(
    hook_fns: Optional[List[Union[Tuple[str, Dict], str]]],
    volume_imgs: List[np.ndarray],
    imgs: List[np.ndarray],
    titles: List[str],
    fig_size: Tuple[int, int],
    root: Optional[Path] = None,
    fname: Optional[str] = None,
    pad_inch: int = 0.3,
    skip_hook_idx: int = 0,
):
    if hook_fns is None:
        return []

    # validate hook_fns
    validate_hook_fns(hook_fns)

    hooks = []
    base_params = {
        "volume_imgs": volume_imgs,
        "imgs": imgs,
        "titles": titles,
        "fig_size": fig_size,
        "root": root,
        "fname": fname,
        "pad_inch": pad_inch,
        "skip_hook_idx": skip_hook_idx,
    }
    for hook_fn in hook_fns:
        if isinstance(hook_fn, tuple):
            hook_fn, kwargs = hook_fn
            base_params.update(kwargs)

        if isinstance(hook_fn, str):
            hooks.append(globals()[hook_fn](**base_params))
        else:
            raise ValueError("hook_fn should be a string or a tuple")

    return hooks


def run_intermediate_hook(
    hook_fns: List[Union[Tuple[str, Dict], str]], ax: Axes, img_idx: int, **kwargs
):
    for hook_fn in hook_fns:
        hook_fn.run_intermediate_hook(ax, img_idx, **kwargs)


def run_final_hook(
    hook_fns: List[Union[Tuple[str, Dict], str]], axes: List[Axes], **kwargs
):
    for hook_fn in hook_fns:
        hook_fn.run_final_hook(axes, **kwargs)


def validate_hook_fns(hook_fns: List[Union[Tuple[str, Dict], str]]):
    for hook_fn in hook_fns:
        if isinstance(hook_fn, str):
            assert (
                hook_fn in HOOK_LIST
            ), f"hook_fn {hook_fn} not in {HOOK_LIST}\nAvailable Hook list: {HOOK_LIST}"
        elif isinstance(hook_fn, tuple):
            assert (
                hook_fn[0] in HOOK_LIST
            ), f"hook_fn {hook_fn[0]} not in {HOOK_LIST}\nAvailable Hook list: {HOOK_LIST}"
        else:
            raise ValueError("hook_fn should be a string or a tuple")


if __name__ == "__main__":
    pass
