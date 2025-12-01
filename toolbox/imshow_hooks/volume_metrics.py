from ..utils import calc_psnr, calc_ssim
from .hook_base import HookBase


class VolumeMetricScores(HookBase):

    def __init__(
        self,
        name="VolumneMetricScores",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.is_gt = True
        self.set_params()
        self.metrics_fns = {"psnr": calc_psnr, "ssim": calc_ssim}

    def run_intermediate_hook(self, ax, img_idx):
        if img_idx < self.skip_hook_idx:
            return

        if self.is_gt:
            text = "\n".join(self.metrics).upper()
            self.is_gt = False  # Only show once
        else:
            trg = self.volume_imgs[0]
            src = self.volume_imgs[img_idx]

            text = ""
            for metric in self.metrics:
                metric_val = self.metrics_fns[metric](trg.copy(), src.copy(), percentile=self.percentile, p=self.p)

                if metric == "psnr":
                    text += f"{metric_val:2.2f}\n"
                elif metric in ["ssim", "lpips"]:
                    text += f"{metric_val*100:2.2f}\n"

        self.annotate(
            ax,
            text,
        )

    def set_params(self):
        self.percentile = self.kwargs.get("percentile", False)
        self.p = self.kwargs.get("p", 98)
        self.metrics = [i.lower() for i in self.kwargs.get("metrics", ["psnr", "ssim"])]
