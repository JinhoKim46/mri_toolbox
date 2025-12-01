"""
===============================================================================
Copyright (c) 2025 Jinho Kim (jinho.kim@fau.de)

Modifications and additional features by Jinho Kim are licensed under the MIT
license, as detailed in the accompanying LICENSE file.
===============================================================================

- MetricScores
    - percentile = False
    - p = 98
    - metrics = ["psnr", "ssim]
- VolumeMetricScores
    - percentile = False
    - p = 98
    - metrics = ["psnr", "ssim]
- DrawBox
    - box_coords = None
    - box_linewidth = 2
    - box_edgecolor = red
- LineProfile
    - profile_coords = None
    - plot_fig_size = (6, 6)
    - legend_loc = upper right
    - metric_on = True
    - linewidth = 1
    - arrow_color = green

** None should be replaced with the actual values.
"""

from . import utils
from .draw_box import DrawBox
from .line_profile import LineProfile
from .metrics import MetricScores
from .volume_metrics import VolumeMetricScores
