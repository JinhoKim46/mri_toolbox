
from matplotlib.patches import Rectangle

from .hook_base import HookBase


class DrawBox(HookBase):
    def __init__(
        self,
        name="DrawBox",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.set_params()
        self.is_gt = True

    def run_intermediate_hook(self, ax, img_idx):
        if img_idx < self.skip_hook_idx:
            return

        if self.is_gt:
            x, y, width, height = self.box_coords
            rect = Rectangle(
                (x, y),
                width=width,
                height=height,
                linewidth=self.box_linewidth,
                edgecolor=self.box_edgecolor,
                facecolor="none",
            )
            ax.add_patch(rect)

            self.is_gt = False  # Only show once

    def set_params(self):
        self.box_coords = self.kwargs.get("box_coords", None)
        self.box_linewidth = self.kwargs.get("box_linewidth", 2)
        self.box_edgecolor = self.kwargs.get("box_edgecolor", "red")

        assert (
            self.box_coords is not None
        ), "Missing box_coords (required) in the kwargs."
