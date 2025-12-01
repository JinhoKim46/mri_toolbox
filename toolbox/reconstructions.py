from typing import Tuple

import sigpy as sp
import cupy as cp
import numpy as np
import sigpy.mri as mr
from pygrappa import grappa
from tqdm import tqdm

from .utils import check_dim, get_ref, ifft2c, print_time

__all__ = ["run_sense", "run_cs", "run_grappa", "rss"]

def run_sense(
    kspace: np.ndarray,
    mps: np.ndarray,
    lamda: float = 0.01,
    max_iter: int = 10,
    verbose: bool = True,
    show_pbar: bool = True,
    leave_pbar: bool = True,
    device: int = 0,
):
    """
    SENSE reconstruction using Sigpy framework.
    :param kspace:      Undersampled kspace(shape: [nSlice, nCoil, PE, RO])
    :param mps:         Sensitivity maps(shape: [nSlice, nCoil, PE, RO])
    :param lamda:       Regularization parameter
    :param max_iter:    Maximum iteration
    :param verbose:     Show verbose
    :param show_pbar:   Toggle whether show progress bar
    :param leave_pbar:  Toggle whether to leave progress bar after finished
    :param device:      Device ID for computation
    :return:
            recon:      SENSE reconstruction(shape: [nSlice, rows, cols])
    """

    nSlice, nCoil, row, col = kspace.shape
    if verbose:
        print_time("- SENSE reconstruction", level=1)
        print_time(f"- lamda: {lamda}, max_iter: {max_iter}", level=2)

    if show_pbar:
        pbar = tqdm(total=nSlice, desc="SENSE reconstruction", leave=leave_pbar)

    # check dimension
    kspace = check_dim(kspace)
    mps = check_dim(mps)

    device = sp.Device(device)
    with device:
        recon = cp.zeros((nSlice, row, col), dtype=kspace.dtype)
        kspace = cp.asarray(kspace)
        mps = cp.asarray(mps)

    for i, (k, sm) in enumerate(zip(kspace, mps)):
        recon[i] = mr.app.SenseRecon(
            k, sm, device=device, lamda=lamda, max_iter=max_iter, show_pbar=False
        ).run()

        if show_pbar:
            pbar.update()
            pbar.refresh()

    if show_pbar:
        pbar.close()

    return recon.get()


def run_cs(
    kspace: np.ndarray,
    mps: np.ndarray,
    lamda: float = 0.01,
    max_iter: int = 10,
    verbose: bool = True,
    show_pbar: bool = True,
    leave_pbar: bool = True,
    device: int = 0,
):
    """
    Compressed sensing reconstruction using Sigpy framework.
    :param kspace:      Undersampled kspace(shape: [nSlice, nCoil, PE, RO])
    :param mps:         Sensitivity maps(shape: [nSlice, nCoil, PE, RO])
    :param lamda:       Regularization parameter
    :param max_iter:    Maximum iteration
    :param verbose:     Show verbose
    :param show_pbar:   Toggle whether show progress bar
    :param leave_pbar:  Toggle whether to leave progress bar after finished
    :param device:      Device ID for computation
    :return:
            recon:      CS reconstruction(shape: [nSlice, rows, cols])
    """
    nSlice, nCoil, row, col = kspace.shape

    if verbose:
        print_time("- CS reconstruction", level=1)
        print_time(f"- lamda: {lamda}, max_iter: {max_iter}", level=2)

    if show_pbar:
        pbar = tqdm(total=nSlice, desc="CS reconstruction", leave=leave_pbar)

    # check dimension
    kspace = check_dim(kspace)
    mps = check_dim(mps)
    
    device = sp.Device(device)
    with device:
        recon = cp.zeros((nSlice, row, col), dtype=kspace.dtype)
        kspace = cp.asarray(kspace)
        mps = cp.asarray(mps)

    for i, (k, sm) in enumerate(zip(kspace, mps)):
        recon[i] = mr.app.L1WaveletRecon(
            k, sm, device=device, lamda=lamda, max_iter=max_iter, show_pbar=False
        ).run()

        if show_pbar:
            pbar.update()
            pbar.refresh()

    if show_pbar:
        pbar.close()

    return recon.get()


def run_grappa(
    kspace: np.ndarray,
    mps: np.ndarray,
    kernel_size: int = 5,
    fft_axes: Tuple[int, int] = (-2, -1),
    coil_axes: int = 1,
    verbose: bool = True,
    show_pbar: bool = True,
    leave_pbar: bool = True,
):
    """
    GRAPPA reconstruction.
    :param kspace:              Undersampled kspace(shape: [nSlice, nCoil, PE, RO])
    :param mps:                 Sensitivity maps(shape: [nSlice, nCoil, PE, RO])
    :param kernel_size:         Kernel size
    :param acc:                 Acceleration rate
    :param fft_axes:            Axes to perform FFT
    :param coil_axes:           Coil axes
    :param is_coil_combine:     Combine coils or not
    :param verbose:             Show verbose
    :param show_pbar:           Toggle whether show progress bar
    :param leave_pbar:          Toggle whether to leave progress bar after finished

    :return:
            recon:      GRAPPA reconstruction(shape: [nSlice, rows, cols])
    """
    nSlice, nCoil, _, _ = kspace.shape

    if verbose:
        print_time("- GRAPPA reconstruction", level=1)

    if show_pbar:
        pbar = tqdm(total=nSlice, desc="GRAPPA reconstruction", leave=leave_pbar)

    calib = get_ref(kspace)
    kernel_size = (kernel_size, kernel_size)

    grappa_k = np.zeros_like(kspace)
    for slice_num in range(nSlice):
        grappa_k[slice_num] = grappa(
            kspace[slice_num], calib[slice_num], kernel_size, coil_axis=(coil_axes - 1)
        )

        if show_pbar:
            pbar.update()
            pbar.refresh()

    coil_imgs = ifft2c(grappa_k, fft_axes)

    recon = np.sum(coil_imgs * np.conj(mps), axis=coil_axes)

    if show_pbar:
        pbar.close()

    return recon, grappa_k

def rss(coil_imgs, coil_axis=0):
    return np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=coil_axis))

