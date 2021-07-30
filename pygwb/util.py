import os
import shutil

import numpy as np

from .spectral import coarse_grain


class TimeSeries:
    def __init__(self, times, data):
        self.times = times
        self.t0 = times[0]
        self.deltaT = times[1] - times[0]
        self.Fs = 1 / self.deltaT
        self.data = data

    def window(self):
        return TimeSeries(self.times, np.hanning(len(self.data)) * self.data)

    def zero_pad(self, zpf=2):
        NZeros = (zpf - 1) * len(self.data)
        new_data = np.append(self.data, np.zeros(NZeros))
        additional_times = np.arange(
            self.times[-1], self.times[-1] + NZeros * self.deltaT, self.deltaT
        )
        new_times = np.append(self.times, additional_times)
        return TimeSeries(new_times, new_data)

    def window_and_fft(self):
        # window
        ts_w = self.window()

        # zero pad
        ts_wz = ts_w.zero_pad(zpf=2)

        # fft
        data_tilde = np.fft.rfft(ts_wz.data) * self.deltaT
        data_tilde = data_tilde[1:]

        # construct the frequency array
        deltaF = 1 / (len(ts_wz.data) * ts_wz.deltaT)
        fmin = deltaF
        fmax = 1 / (2 * ts_wz.deltaT)
        epsilon = deltaF / 100
        freqs = np.arange(fmin, fmax + epsilon, deltaF)

        return FrequencySeries(freqs, data_tilde)


class FrequencySeries:
    def __init__(self, freqs, data):
        self.deltaF = freqs[1] - freqs[0]
        self.freqs = freqs
        self.data = data

    def __mul__(self, x):
        if type(x) is float or type(x) is int:
            return FrequencySeries(self.freqs, self.data * x)
        return FrequencySeries(self.freqs, self.data * x.data)

    def __truediv__(self, x):
        if type(x) is float or type(x) is int:
            return FrequencySeries(self.freqs, self.data / x)
        return FrequencySeries(self.freqs, self.data / x.data)

    def coarse_grain(self, newDeltaF, newFMin, newFMax):
        coarsening_factor = newDeltaF / self.deltaF

        y = coarse_grain(self.data, coarsening_factor)

        new_freqs = coarse_grain(self.freqs, coarsening_factor)
        keep = (new_freqs > newFMin) & (new_freqs <= newFMax)
        return y[keep]


def slice_time_series(timeseries, istart, iend):
    return TimeSeries(timeseries.times[istart:iend], timeseries.data[istart:iend])


def window_factors(N):
    """
    calculate window factors for a hann window
    """
    w = np.hanning(N)
    w1w2bar = np.mean(w ** 2)
    w1w2squaredbar = np.mean(w ** 4)

    w1 = w[int(N / 2) : N]
    w2 = w[0 : int(N / 2)]
    w1w2squaredovlbar = 1 / (N / 2.0) * np.sum(w1 ** 2 * w2 ** 2)

    w1w2ovlbar = 1 / (N / 2.0) * np.sum(w1 * w2)

    return w1w2bar, w1w2squaredbar, w1w2ovlbar, w1w2squaredovlbar


def calc_Y_sigma_from_Yf_varf(Y_f, var_f, freqs=None, alpha=0, fref=1):
    if freqs is not None:
        weights = (freqs / fref) ** alpha
    else:
        weights = np.ones(Y_f.shape)

    var = 1 / np.sum(var_f ** (-1) * weights ** 2)

    # Y = np.sum(Y_f * var_f**(-1)) / np.sum( var_f**(-1) )
    Y = np.sum(Y_f * weights * (var / var_f))

    sigma = np.sqrt(var)

    return Y, sigma


def calc_rho1(N):
    w1w2bar, _, w1w2ovlbar, _ = window_factors(100000)
    rho1 = (0.5 * w1w2ovlbar / w1w2bar) ** 2
    return rho1


def calc_bias(segmentDuration, deltaF, deltaT):
    N = int(segmentDuration / deltaT)
    rho1 = calc_rho1(N)
    Nsegs = 2 * segmentDuration * deltaF - 1
    wfactor = (1 + 2 * rho1) ** (-1)
    Neff = 2 * wfactor * (2 * segmentDuration * deltaF - 1)
    bias = Neff / (Neff - 1)
    return bias


def make_dir(dirname):
    try:
        os.mkdir(dirname)
    except:
        pass  # directory already exists


def cleanup_dir(outdir):
    # cleanup
    try:
        shutil.rmtree(outdir)
    except OSError as e:
        pass  # directory doesn't exist