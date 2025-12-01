from abc import ABC
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes


class HookBase(ABC):

    def __init__(
        self,
        name: str,
        volume_imgs: List[np.ndarray],
        imgs: List[np.ndarray],
        titles: List[str],
        fig_size: Optional[Tuple[int, int]] = None,
        font_size: int = 20,
        font_color: str = "yellow",
        font_weight: str = "normal",
        root: Optional[Path] = None,
        fname: Optional[str] = None,
        pad_inch: int = 0.3,
        skip_hook_idx: int = 0,
        **kwargs,
    ):
        self.name = name
        self.volume_imgs = volume_imgs
        self.imgs = imgs
        self.titles = titles
        self.font_size = font_size
        self.font_color = font_color
        self.font_weight = font_weight
        self.fig_size = fig_size
        self.root = root
        self.fname = fname
        self.pad_inch = pad_inch
        self.skip_hook_idx = skip_hook_idx

        self.kwargs = kwargs

        plt.rcParams["font.family"] = "monospace"

    def run_intermediate_hook(self, ax: Axes, img_idx: int, **kwargs):
        return

    def run_final_hook(self, axes: List[Axes], **kwargs):
        return

    def annotate(self, ax, text):
        ax.annotate(
            text,
            xy=(1, 1),
            xytext=(-2, -2),
            fontsize=self.font_size,
            color=self.font_color,
            xycoords="axes fraction",
            textcoords="offset points",
            horizontalalignment="right",
            verticalalignment="top",
            fontweight=self.font_weight,
        )
