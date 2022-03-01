import os
import shutil

import gwpy
import h5py
import numpy as np
from scipy.interpolate import interp1d

from pygwb.constants import H0

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


def calc_Y_sigma_from_Yf_varf(Y_f, var_f, freqs=None, alpha=0, fref=1, weight_spectrum=True):
    if weight_spectrum and freqs is None:
        raise ValueError("Must supply frequency array if you want to weight the spectrum when combining")
    if weight_spectrum:
        weights = (freqs / fref) ** alpha
    else:
        weights = np.ones(Y_f.shape)

    var = 1 / np.sum(var_f ** (-1) * weights ** 2)

    # Y = np.sum(Y_f * var_f**(-1)) / np.sum( var_f**(-1) )
    Y = np.nansum(Y_f * weights * (var / var_f))

    sigma = np.sqrt(var)

    return Y, sigma


def calc_rho1(N):
    w1w2bar, _, w1w2ovlbar, _ = window_factors(N)
    rho1 = (0.5 * w1w2ovlbar / w1w2bar) ** 2
    return rho1


def calc_bias(segmentDuration, deltaF, deltaT, N_avg_segs=1):
    N = int(segmentDuration / deltaT)
    rho1 = calc_rho1(N)
    Nsegs = 2 * segmentDuration * deltaF - 1
    wfactor = (1 + 2 * rho1) ** (-1)
    Neff = N_avg_segs * wfactor * Nsegs
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


def omega_to_power(omega_GWB, frequencies):
    """
    Function that computes the GW power spectrum starting from the omega_GWB
    spectrum.

    Parameters
    ==========

    Returns
    =======
    power: gwpy.frequencyseries.FrequencySeries
        A gwpy FrequencySeries containing the GW power spectrum
    """
    H_theor = (3 * H0 ** 2) / (10 * np.pi ** 2)

    power = H_theor * omega_GWB * frequencies ** (-3)
    power = gwpy.frequencyseries.FrequencySeries(power, frequencies=frequencies)

    return power





def make_freqs(Nsamples, deltaF):
    """
    Function that makes an array of frequencies given the sampling rate
    and the segment duration specified in the initial parameter file.

    Parameters
    =========

    Returns
    =======
    freqs: array_like
        Array of frequencies for which an isotropic stochastic background
        will be simulated.
    """
    if NSamples % 2 == 0:
        numFreqs = NSamples / 2 - 1
    else:
        numFreqs = (NSamples - 1) / 2

    freqs = np.array([deltaF * (i + 1) for i in range(int(numFreqs))])
    return freqs


def interpolate_frequency_series(fSeries, new_frequencies):
    """
    Parameters
    ==========
    fSeries: FrequencySeries object
    new_frequencies: array_like
    """
    spectrum = fSeries.value
    frequencies = fSeries.frequencies.value

    spectrum_func = interp1d(
        frequencies, spectrum, kind="cubic", fill_value="extrapolate"
    )

    return gwpy.frequencyseries.FrequencySeries(
        spectrum_func(new_frequencies), frequencies=new_frequencies
    )




def read_jobfiles(njobs, directory, segment_duration):
    """
    Method that reads in the job files to extract quantities such as the point estimate and sigmas.

    Parameters
    ==========
    njobs: int
        Number of jobs
    directory: str
        Directory where the mat files are stored
    segment_duration: int
        Duration of a segment in seconds

    Returns
    =======
    sliding_times_all: array
        Array containing all the GPS times for this particular run
    sliding_omega_all: array
        Array containing the omega point estimate for this particular run
    sliding_sigmas_all: array
        Array containing the sigmas for this particular run
    naive_sigma_all: array
        Array containing the naive sigmas for this particular run
    """
    sliding_omega_all = np.array([])
    sliding_sigmas_all = np.array([])
    naive_sigma_all = np.array([])
    sliding_times_all = np.array([])
    for nn in range(njobs):
        jn = nn + 1
        file1 = "%sH1L1.job%d.mat" % (directory, jn)
        with h5py.File(file1, "r") as FF:
            try:
                sliding_omega_all = np.append(
                    sliding_omega_all,
                    np.array(FF["ccStat"][0]).flatten() / segment_duration,
                )
                sliding_sigmas_all = np.append(
                    sliding_sigmas_all,
                    np.array(FF["ccSigma"][0]).flatten() / segment_duration,
                )
                naive_sigma_all = np.append(
                    naive_sigma_all,
                    np.array(FF["naiSigma"][0]).flatten() / segment_duration,
                )
                sliding_times_all = np.append(
                    sliding_times_all, np.array(FF["segmentStartTime"][0]).flatten()
                )
            except:
                print("No data for job %u" % jn)
                continue
    return sliding_times_all, sliding_omega_all, sliding_sigmas_all, naive_sigma_all


def StatKS(DKS):
    jmax = 500
    pvalue = 0.0
    for jj in np.arange(1, jmax + 1):
        pvalue += 2.0 * (-1) ** (jj + 1) * np.exp(-2.0 * jj ** 2 * DKS ** 2)
    return pvalue


def calculate_point_estimate_sigma_spectrogram(freqs, csd, avg_psd_1, avg_psd_2, orf,
                                               sample_rate, segment_duration, fref=1,
                                               alpha=0, weight_spectrogram=False):
    S_alpha = (
        3
        * H0 ** 2
        / (10 * np.pi ** 2)
        / freqs ** 3
    )
    if weight_spectrogram:
        S_alpha *= (freqs / fref) ** alpha
    Y_fs = np.real(csd) / (orf * S_alpha)
    var_fs = (
        1
        / (2 * segment_duration * (freqs[1] - freqs[0]))
        * avg_psd_1
        * avg_psd_2
        / (orf ** 2 * S_alpha ** 2)
    )

    w1w2bar, w1w2squaredbar, _, _ = window_factors(
        sample_rate * segment_duration
    )

    var_fs = var_fs * w1w2squaredbar / w1w2bar ** 2
    return Y_fs, var_fs
