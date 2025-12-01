from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact

from . import utils
from .imshow_hooks import utils as hook_utils

__all__ = ["imshow", "inter_imshow"]

def imshow(
    imgs: list,
    gt: Optional[np.ndarray] = None,
    gt_title: Optional[str] = None,
    titles: Optional[list] = None,
    root: Optional[Path] = None,
    fname: Optional[str] = None,
    suptitle: Optional[str] = None,
    norm: float = 0.0,
    axis: bool = False,
    fig_size: Optional[tuple[int, int]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    save_indiv: bool = False,
    is_MIP: bool = False,
    MIP_axes: int = 0,
    slice_num: int = None,
    slice_axes: int = 0,
    num_rows: int = 1,
    pos: Optional[list] = None,
    crop_fn=None,
    pad_inch=0.3,
    font_size=20,
    hook_fns: Optional[List[Union[Tuple[str, Dict], str]]] = None,
    skip_hook_idx: int = 0,
    img_plot: bool = True,
    complex_phase: bool = False,
    tight_layout: bool = True,
):
    """
    This function displays multiple images in a single row.
    vmin and vmax are set to
        1) if vmin and vmax are None, the minimum and maximum value of the first image in imgs list
        2) given vmin and vmax as parameters


    Args:
        imgs:                       list of images to display
        gt:                         ground truth image, optional
        gt_title:                   title for the ground truth image, optional
        root:                       Root path to save, optional
        fname:                      name of the file to save the figure, optional
        suptitle:                   main title for the figure, optional
        norm:                       normalization factor, default is 0.0
        axis:                       flag to display axis, default is False
        fig_size:                   figure size, default is (15,10)
        vmin:                       vmin for visualizaion
        vmax:                       vmax for visualizaion
        save_indiv:                 Save individual images or not
        is_MIP:                     Plot MIP images or not
        MIP_axes:                   MIP axes
        slice_num:                  Slice number to be displayed
        slice_axes:                 Slice axes
        num_rows:                   The number of rows of layout (default: 1)
        pos:                        Position of images.
                                    ex) for 2x3 layout, [1,1,1,0,1,1] plots images like
                                                        ===============
                                                        img1 img2 img3
                                                             img4 img5
                                                        ===============
                                    ex) for 2x3 layout with gt given, [1,1,1,0,1,1] plots images like
                                                        ===============
                                                        gt img1 img2 img3
                                                                img4 img5
                                                        ===============
                                    Available options are ['normal', 'metrics']
        crop_fn:                    Crop function to be applied to the image
        pad_inch:                   Padding for individual image
        font_size:                  font size for metric display (default: 20)
        font_color:                 font color for metric display (default: yellow
                                    Available options are ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
        font_weight:                font weight for metric display (default: normal)
                                    Available options are ['normal', 'bold', 'heavy', 'light', 'ultralight', 'medium', 'semibold', 'demibold']
        hook_fns:                   list of functions to be applied to the image
                                        ex) List[Union[Tuple[str, Dict], str]]
                                        ex) [('metrics', {'metric': 'PSNR', 'gt': gt}), 'normal']
        skip_hook_idx:              Skip hook function index for run_intermediate_hook. Default is 0
        img_plot:                   Plot image or not. Default is True
        complex_phase:              Display complex phase image or not. Default is False
        tight_layout:               Tight layout or not. Default is True

    """
    if not isinstance(imgs, list):
        imgs = [imgs]
        
    utils.validate_images(imgs, gt, is_MIP)

    pos, num_cols = utils.get_pos(pos, num_rows=num_rows, num_imgs=len(imgs))

    # if gt is given,
    titles = [None] * len(imgs) if titles is None else titles
    if gt is not None:
        pos, num_cols = utils.gt_pos_extension(pos, num_rows, num_cols)
        imgs = [gt] + imgs
        titles = ["Ground Truth"] + titles if gt_title is None else [gt_title] + titles

    # If gt is given, add gt to the imgs list
    # Convert images to numpy array
    imgs = utils.anything_to_np(imgs, complex_phase=complex_phase)

    # if mrcp_crop is True, then crop the image to the size of MRCP image
    # or if crop_fn is given, apply the crop function to the image
    if crop_fn is not None:
        imgs = [crop_fn(i) for i in imgs]

    volume_imgs = [i.copy() for i in imgs]

    if is_MIP:
        imgs = [np.abs(i).max(MIP_axes) for i in imgs]
    else:
        if slice_num is not None:
            imgs = [np.take(i, slice_num, slice_axes) for i in imgs]

    if fig_size is None:
        col_size, row_size = utils.get_figsize(imgs[0], num_rows, num_cols, titles[0])
        if titles:
            row_size += (font_size * 0.02) * num_rows
    else:
        col_size, row_size = fig_size

    f = plt.figure(figsize=(col_size, row_size), facecolor="#2C2B2B")

    hook_fns = hook_utils.set_hook_fns(
        hook_fns,
        volume_imgs,
        imgs,
        titles,
        fig_size,
        root,
        fname,
        pad_inch,
        skip_hook_idx,
    )

    img_idx = 0
    axes_list = []
    for i, pos_indiv in enumerate(pos, start=1):
        ax = f.add_subplot(num_rows, num_cols, i)
        if axis is False:
            ax.axis("off")

        # Set img and title only when pos_indiv is valid.
        img = imgs[img_idx] if pos_indiv != 0 else np.ones_like(imgs[0])
        title = titles[img_idx] if pos_indiv != 0 else ""

        if hook_fns and pos_indiv:
            hook_utils.run_intermediate_hook(hook_fns, ax, img_idx)

        if norm == 0.0:
            vmax = imgs[0].max() if vmax is None else vmax
            vmin = imgs[0].min() if vmin is None else vmin
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        else:
            ax.imshow(img, cmap="gray", norm=clr.PowerNorm(gamma=norm))

        if pos_indiv:
            axes_list.append(ax)
        ax.set_title(title, color="#00FF08")

        img_idx += 1 if pos_indiv else 0

    hook_utils.run_final_hook(hook_fns, axes_list)

    f.suptitle(suptitle, color="#00FF08") if suptitle is not None else f.suptitle("")
    if tight_layout:
        f.subplots_adjust(wspace=0.000, hspace=0.000)

    if img_plot:
        utils.safe_show(f)
    else:
        plt.close(f)

    if fname is not None:
        utils.save_figures(f, root, fname, axes_list, titles, pad_inch, save_indiv)


def inter_imshow(
    imgs: list,
    gt: Optional[np.ndarray] = None,
    gt_title: Optional[str] = None,
    titles: Optional[list] = None,
    suptitle: Optional[str] = None,
    axis: bool = False,
    fig_size: Optional[tuple] = None,
    num_rows: int = 1,
    slice_axes: int = 0,
    pos: Optional[list] = None,
    crop_fn=None,
    pad_inch=0.3,
    hook_fns: Optional[List[Tuple[Callable, Dict]]] = None,
    **kwargs
):

    def inter_fn(slice_num=15, norm=0.7):
        imshow(
            imgs=imgs,
            gt=gt,
            gt_title=gt_title,
            titles=titles,
            suptitle=suptitle,
            norm=norm,
            axis=axis,
            fig_size=fig_size,
            slice_num=slice_num,
            slice_axes=slice_axes,
            num_rows=num_rows,
            pos=pos,
            crop_fn=crop_fn,
            pad_inch=pad_inch,
            hook_fns=hook_fns,
            **kwargs,
        )

    interact(inter_fn, slice_num=(0, imgs[0].shape[slice_axes] - 1, 1), norm=(0.0, 1.0, 0.1))




