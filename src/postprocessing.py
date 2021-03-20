import numpy as np

from .util import calc_bias, window_factors


def postprocessing(
    Ys, sigs, jobDuration, segmentDuration, deltaF, deltaT, bufferSecs=0
):
    M = int(np.floor((jobDuration - 2 * bufferSecs) / segmentDuration))
    numSegmentsTotal = 2 * M - 1

    N = int(segmentDuration / deltaT)
    _, w1w2squaredbar, _, w1w2squaredovlbar = window_factors(N)

    Y_odds = Ys[::2]
    Y_evens = Ys[1::2]
    sig_odds = sigs[::2]
    sig_evens = sigs[1::2]

    Y_o = np.sum(Y_odds / sig_odds ** 2) / np.sum(1 / sig_odds ** 2)
    s_o = np.sqrt(1 / np.sum(1 / sig_odds ** 2))

    Y_e = np.sum(Y_evens / sig_evens ** 2) / np.sum(1 / sig_evens ** 2)
    s_e = np.sqrt(1 / np.sum(1 / sig_evens ** 2))

    s_oe = np.sqrt(
        (w1w2squaredovlbar / w1w2squaredbar)
        * 0.5
        * (s_o ** 2 + s_e ** 2)
        * (1.0 - 1.0 / (2 * numSegmentsTotal))
    )

    C_oo = s_o ** 2
    C_oe = s_oe ** 2
    C_eo = C_oe
    C_ee = s_e ** 2
    detC = C_oo * C_ee - C_oe ** 2

    lambda_o = (s_e ** 2 - s_oe ** 2) / detC
    lambda_e = (s_o ** 2 - s_oe ** 2) / detC

    lambda_sum = lambda_o + lambda_e

    Y_opt = (lambda_o * Y_o + lambda_e * Y_e) / lambda_sum
    var_opt = 1 / lambda_sum
    sig_opt = np.sqrt(var_opt)

    bias = calc_bias(segmentDuration, deltaF, deltaT)
    sig_opt = sig_opt * bias

    return Y_opt, sig_opt


def postprocessing_spectra(
    Y_fs, var_fs, jobDuration, segmentDuration, deltaF, deltaT, bufferSecs=0
):
    M = int(np.floor((jobDuration - 2 * bufferSecs) / segmentDuration))
    numSegmentsTotal = 2 * M - 1

    N = int(segmentDuration / deltaT)
    _, w1w2squaredbar, _, w1w2squaredovlbar = window_factors(N)

    Y_fs_odds = Y_fs[:, ::2]
    Y_fs_evens = Y_fs[:, 1::2]
    var_fs_odds = var_fs[:, ::2]
    var_fs_evens = var_fs[:, 1::2]

    Y_f_o = np.sum(Y_fs_odds / var_fs_odds, axis=1) / np.sum(1 / var_fs_odds, axis=1)
    v_f_o = 1 / np.sum(1 / var_fs_odds, axis=1)

    Y_f_e = np.sum(Y_fs_evens / var_fs_evens, axis=1) / np.sum(1 / var_fs_evens, axis=1)
    v_f_e = 1 / np.sum(1 / var_fs_evens, axis=1)

    v_f_oe = (
        (w1w2squaredovlbar / w1w2squaredbar)
        * 0.5
        * (v_f_o + v_f_e)
        * (1.0 - 1.0 / (2 * numSegmentsTotal))
    )

    C_f_oo = v_f_o
    C_f_oe = v_f_oe
    C_f_eo = C_f_oe
    C_f_ee = v_f_e
    detC_f = C_f_oo * C_f_ee - C_f_oe ** 2

    lambda_f_o = (v_f_e - v_f_oe) / detC_f
    lambda_f_e = (v_f_o - v_f_oe) / detC_f

    lambda_f_sum = lambda_f_o + lambda_f_e

    Y_f = (lambda_f_o * Y_f_o + lambda_f_e * Y_f_e) / lambda_f_sum
    var_f = 1 / lambda_f_sum

    bias = calc_bias(segmentDuration, deltaF, deltaT)
    var_f = var_f * bias ** 2

    return Y_f, var_f
