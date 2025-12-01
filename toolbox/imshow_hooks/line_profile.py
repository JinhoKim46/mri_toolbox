from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from scipy.stats import pearsonr
from skimage.measure import profile_line

from .hook_base import HookBase


class LineProfile(HookBase):
    def __init__(
        self,
        name="LineProfile",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.set_params()
        self.is_gt = True

    def run_intermediate_hook(self, ax: Axes, img_idx: int, **kwargs):
        if img_idx < self.skip_hook_idx:
            return

        if self.is_gt:
            x0, y0, x1, y1 = self.profile_coords
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->", color=self.arrow_color, linewidth=self.linewidth
                ),
            )

            self.is_gt = False  # Only show once

    def run_final_hook(self, axes: List[Axes], **kwargs):
        x0, y0, x1, y1 = self.profile_coords

        f_profile, ax_profile = plt.subplots(1, 1, figsize=self.plot_fig_size)
        ax_profile.set_xlabel("Profile length (pixel)")
        ax_profile.set_ylabel("Signal intensity (a.u.)")
        profile_gt = None
        norm_factor = np.abs(self.imgs[0]).max()
        for i, (img, title) in enumerate(zip(self.imgs, self.titles)):
            if i >= self.skip_hook_idx:
                profile = profile_line(img / norm_factor, (y0, x0), (y1, x1))
                profile_gt = profile if profile_gt is None else profile_gt
                corrcoef = pearsonr(profile_gt, profile)[0]
                label = (
                    f"{title: <11} ({corrcoef:0.3f})" if self.metric_on else title
                )
                ax_profile.plot(profile, label=label)

            ax_profile.legend(loc=self.legend_loc)
        f_profile.show()

        # configuration for saving figures
        if self.fname is not None:
            if self.root is None:
                self.root = Path.cwd()
            if isinstance(self.root, str):
                self.root = Path(self.root)
            self.root = self.root / "Figures"
            if not self.root.exists() and self.fname:
                self.root.mkdir(parents=True, exist_ok=True)

            f_profile.savefig(
                self.root / f"{self.fname}_profile",
                bbox_inches="tight",
                pad_inches=self.pad_inch,
            )

    def set_params(self):
        self.profile_coords = self.kwargs.get("profile_coords", None)
        self.plot_fig_size = self.kwargs.get("plot_fig_size", (8, 6))
        self.legend_loc = self.kwargs.get("legend_loc", "upper right")
        self.metric_on = self.kwargs.get("metric_on", True)
        self.linewidth = self.kwargs.get("linewidth", 1.5)
        self.arrow_color = self.kwargs.get("arrow_color", "yellow")

        assert (
            self.profile_coords is not None
        ), "Missing line_coords (required) in the kwargs."
