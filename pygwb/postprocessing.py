import pickle

import h5py
import numpy as np
from loguru import logger
from tqdm import tqdm

from .util import calc_bias, window_factors


def postprocess_Y_sigma(Y_fs, var_fs, segment_duration, deltaF, new_sample_rate):
    size = np.size(Y_fs, axis=0)
    _, w1w2squaredbar, _, w1w2squaredovlbar = window_factors(
        segment_duration * new_sample_rate
    )
    k = w1w2squaredovlbar / w1w2squaredbar

    # even/odd indices
    evens = np.arange(0, size, 2)
    odds = np.arange(1, size, 2)

    X_even = np.nansum(Y_fs[evens] / var_fs[evens], axis=0)
    GAMMA_even = np.nansum(var_fs[evens] ** -1, axis=0)
    X_odd = np.nansum(Y_fs[odds] / var_fs[odds], axis=0)
    GAMMA_odd = np.nansum(var_fs[odds] ** -1, axis=0)
    sigma2_oo = 1 / np.nansum(GAMMA_odd)
    sigma2_ee = 1 / np.nansum(GAMMA_even)
    sigma2_1 = 1 / np.nansum(var_fs[0, :] ** -1)
    sigma2_N = 1 / np.nansum(var_fs[-1, :] ** -1)
    sigma2IJ = 1 / sigma2_oo + 1 / sigma2_ee - (1 / 2) * (1 / sigma2_1 + 1 / sigma2_N)

    Y_f_new = (
        X_odd * (1 - (k / 2) * sigma2_oo * sigma2IJ)
        + X_even * (1 - (k / 2) * sigma2_ee * sigma2IJ)
    ) / (
        GAMMA_even
        + GAMMA_odd
        - k
        * (GAMMA_even + GAMMA_odd - (1 / 2) * (1 / var_fs[0, :] + 1 / var_fs[-1, :]))
    )

    inv_var_f_new = (
        GAMMA_odd
        + GAMMA_even
        - k
        * (GAMMA_odd + GAMMA_even - (1 / 2) * (1 / var_fs[0, :] + 1 / var_fs[-1, :]))
    ) / (1 - (k**2 / 4) * sigma2_oo * sigma2_ee * sigma2IJ**2)
    bias = calc_bias(segment_duration, deltaF, 1 / new_sample_rate, N_avg_segs=2)
    logger.debug(f"Bias factor: {bias}")
    var_f_new = (1 / inv_var_f_new) * bias**2

    return Y_f_new, var_f_new
