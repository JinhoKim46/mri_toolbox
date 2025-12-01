from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import cupy as cp
import matplotlib
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM


def validate_images(imgs, gt, is_MIP):
    # Check for image size consistency
    if gt is not None:
        gt_shape = gt.shape
        for img in imgs:
            if img.shape != gt_shape:
                raise ValueError("All images must have the same size.")

    # Check for image dimensions
    if imgs[0].ndim not in [2, 3, 4]:
        message = """All images must be 2, 3, or 4 dimensions.
        - 2D: (row, col)
        - 3D: (slice, row, col) or (row, col, 2)
        - 4D: (slice, row, col, 2) with is_MIP=True"""
        raise ValueError(message)
    if imgs[0].ndim == 4 and not is_MIP:
        raise ValueError("4D images are only supported for MIP visualization.")


def gt_pos_extension(pos: List, num_rows: int, num_cols: int):
    pos_tmp = []
    for i in range(num_rows):
        pos_tmp += [1 if i == 0 else 0] + pos[
            (len(pos) // num_rows) * i : (len(pos) // num_rows) * (i + 1)
        ]
    num_cols = num_cols + 1

    return pos_tmp.copy(), num_cols


def run_hookers(
    hook_fn, ax, imgs, gt, gt_title, titles, suptitle, norm, axis, fig_size, vmin, vmax
):
    for hook in hook_fn:
        hook(ax, imgs, gt, gt_title, titles, suptitle, norm, axis, fig_size, vmin, vmax)


def save_show(f: matplotlib.figure.Figure):
    backend = matplotlib.get_backend()
    interactive_backend = ["TkAgg", "Qt5Agg", "GTK3Agg", "macOSX", "ipympl"]

    if backend in interactive_backend:
        f.show()

def save_figures(
    f: matplotlib.figure.Figure,
    root: Union[Path, str],
    filename: str,
    axes_list: list,
    titles: str,
    pad_inch: float,
    save_indiv: bool,
):
    if root is None:
        root = Path.cwd()
    if isinstance(root, str):
        root = Path(root)
    root = root / "Figures"
    root.mkdir(parents=True, exist_ok=True)

    f.savefig(root / filename, bbox_inches="tight", pad_inches=pad_inch)
    if save_indiv:
        for i, (ax, title) in enumerate(zip(axes_list, titles)):
            extent = ax.get_window_extent().transformed(f.dpi_scale_trans.inverted())
            f.savefig(
                root / f"{filename}_{i}_{title}",
                bbox_inches=extent,
                pad_inches=pad_inch,
            )


def safe_show(f: matplotlib.figure.Figure):
    backend = matplotlib.get_backend()
    interactive_backend = ["TkAgg", "Qt5Agg", "GTK3Agg", "macOSX", "ipympl"]

    if backend in interactive_backend:
        f.show()





def get_figsize(img, num_rows, num_cols, title):
    # Handling different dimensions
    if img.ndim == 2:
        row, col = img.shape
    elif img.ndim == 3:
        if img.shape[-1] == 2:
            row, col, _ = img.shape
        else:
            _, row, col = img.shape
    elif img.ndim == 4:
        _, row, col, _ = img.shape

    new_row = row / row
    new_col = col / row

    # Calculate the figsize based on the new_row and new_col
    row_size = num_rows * (new_row * 5)
    col_size = num_cols * (new_col * 5)

    return col_size, row_size


def get_pos(pos, num_rows, num_imgs):
    num_cols = (
        np.ceil(num_imgs / num_rows).astype(int)
        if pos is None
        else np.ceil(len(pos) / num_rows).astype(int)
    )
    len_pos = num_rows * num_cols

    if pos is None:
        pos = [1] * num_imgs + [0] * (len_pos - num_imgs)
    else:  # if pos is given
        assert (
            np.count_nonzero(pos) == num_imgs
        ), f"Givin pos (len:{len(pos)}) are not matched to the number of given images (len:{num_imgs}))"
        res = len_pos - len(pos)
        pos += [0] * res

    return pos, num_cols


def calc_psnr(trg, src, percentile=False, p=98):
    norm_factor = np.percentile(trg.flatten(), p) if percentile else trg.max()
    trg = trg / norm_factor
    src = src / norm_factor
    return PSNR(trg, src, data_range=trg.max())


def calc_ssim(trg, src, percentile=False, p=98, full=False):
    norm_factor = np.percentile(trg.flatten(), p) if percentile else trg.max()
    trg = trg / norm_factor
    src = src / norm_factor
    return SSIM(trg, src, data_range=trg.max(), full=full)



def anything_to_np(imgs, complex_phase=False):
    for i in range(len(imgs)):
        img = imgs[i].squeeze()

        if cp.get_array_module(img) == cp:
            img = img.get()

        if torch.is_tensor(img):
            img = img.cpu().detach().numpy()

        if img.shape[-1] == 2:  # real-valued img (..., 2)
            img = img[..., 0] + 1j * img[..., -1]

        imgs[i] = np.angle(img) if complex_phase else np.abs(img)

    return imgs



def to_numpy(data: torch.Tensor) -> np.ndarray:
    data = data.detach().cpu().numpy()
    return data[..., 0] + 1j * data[..., 1]


def print_time(
    text: str, level: int = 0, output: bool = False, time_begin: Optional[float] = None
):
    cur_time = datetime.now()
    text = " " * (level * 2) + text
    if time_begin is not None:
        text += f" (process time: {cur_time - time_begin}!)"
    print(f"[{cur_time.strftime('%H:%M:%S')}] {text}")

    if output:
        return cur_time



def ifft2c(k, dim=None, img_shape=None, accelerator="cpu", verbose=True) -> np.ndarray:
    """Computes the Fourier transform from k-space to image space
    along a given or all dimensions

    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    xp = get_xp(accelerator, verbose)

    shifted_k = xp.fft.ifftshift(k, axes=dim)
    shifted_img = xp.fft.ifftn(shifted_k, s=img_shape, axes=dim, norm="ortho")
    img = xp.fft.fftshift(shifted_img, axes=dim).astype(xp.complex64)

    if xp == cp:
        img = cp.asnumpy(img)
        cp.get_default_memory_pool().free_all_blocks()  # Release GPU memory

    return img


def fft2c(img, dim=None, k_shape=None, accelerator="cpu", verbose=True) -> np.ndarray:
    """Computes the Fourier transform from image space to k-space space
    along a given or all dimensions

    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    xp = get_xp(accelerator, verbose)

    shifted_img = xp.fft.ifftshift(img, axes=dim)
    shifted_k = xp.fft.fftn(shifted_img, s=k_shape, axes=dim, norm="ortho")
    k = xp.fft.fftshift(shifted_k,axes=dim).astype(xp.complex64)

    if xp == cp:
        k = cp.asnumpy(k)
        cp.get_default_memory_pool().free_all_blocks()  # Release GPU memory

    return k

def get_xp(accelerator, verbose):
    if accelerator == "cpu":
        if verbose:
            print(f"- Using CPU")
        xp = np
    elif accelerator == "gpu" and cp.cuda.is_available():
        if verbose:
            print(f"- Using GPU")
        xp = cp
    else:
        if verbose:
            print("GPU is not availble. Using CPU instead.")
        xp = np

    return xp

def check_dim(x, is_gt=False):
    """
    Check the dimension of the input array.
    The shape of x is supposed to be (slice, coil, RO, PE)
    :param x:       Input array
    :param is_gt:   If the input is ground truth
    :return:        Array with dimension of 4
    """
    if (x.ndim == 3 and not is_gt) or (x.ndim == 2 and is_gt):
        x = x[np.newaxis, ...]

    return x


def get_ref(kspace):
    mask = kspace[0, 0, :, 0].astype(bool)
    acs_start, acs_end = _get_acs_index(mask)
    acs = kspace[:, :, acs_start : acs_end + 1,]

    return acs


def _get_acs_index(mask):
    slices = np.ma.clump_masked(np.ma.masked_where(mask, mask))
    acs_ind = [(s.start, s.stop - 1) for s in slices if s.start < (s.stop - 1)]
    assert (
        acs_ind != [] or len(acs_ind) > 1
    ), "Couldn't extract center lines mask from k-space - is there pat2 undersampling?"
    acs_start = acs_ind[0][0]
    acs_end = acs_ind[0][1]

    return acs_start, acs_end
