import os
import shutil

import numpy as np


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
        fmin = self.freqs[0]
        fmax = self.freqs[-1]
        epsilon = newDeltaF / 100.0
        new_freqs = np.arange(newFMin, newFMax + epsilon, newDeltaF)
        y = coarseGrain_MatlabTranslation(self, newFMin, newDeltaF, len(new_freqs))
        return y


def coarseGrain_MatlabTranslation(x, flowy, deltaFy, Ny):  # JOES's version
    """
    coarseGrain --- coarse grain a frequency-series

    coarseGrain(x, flowy, deltaFy, Ny) accepts a frequency-series structure
    (usually a PSD) and returns a frequency-series structure which has been
    coarse-grained to the frequency values f = flowy + deltaFy*[0:Ny-1].

    coarseGrain also returns the frequency indices of the lowest and highest
    frequency bins of x that overlap with y (0 <= index1 <= length(x);
    1 <= index2 <= length(x)+1) and the fractional contributions from these
    frequency bins (note that these indices start from 0 and and are 1 less
    than the corresponding Matlab indices). The fractional contribution is
    the fraction of the fine bin that overlaps with any part of the coarse
    series.

    The method used is to treat the x-values

    xc(k) = x.flow + (k - 1)*x.deltaF

    as bin centres and the value x.data(k) as the average value over the
    the bin i.e.,

    (1/x.deltaF)*(integral of x.data from the lower edge of the bin to upper edge)

    The coarse graining can then be performed by finding the integral
    from the start of the fine series just below the coarse series (via
    the cumulative sum), interpolating to the coarse series, and taking
    differences to recover the average value of the function for each coarse bin.
    """

    # Set the metadata for the coarse-grained frequency series
    fhighy = flowy + (Ny - 1) * deltaFy
    freqs = np.arange(flowy, fhighy + 0.01 * deltaFy, deltaFy)
    y = FrequencySeries(freqs, np.zeros(Ny))

    # Length of fine series
    Nx = len(x.data)

    # Lower edge of the first bin of the fine series
    flowx = x.freqs[0]
    xLowerEdge = flowx - 0.5 * x.deltaF

    # Upper edge of the last bin of the fine series
    xUpperEdge = flowx + (Nx - 0.5) * x.deltaF

    # yi(k) is the lower edge of bin k for the coarse series
    yi = flowy + y.deltaF * np.arange(-0.5, Ny - 0.5 + 0.01, 1)

    # Lower edge of the first bin of the coarse series
    yLowerEdge = yi[0]

    # Upper edge of the last bin of the coarse series
    yUpperEdge = yi[-1]

    # Error checking
    if Nx <= 0 or Ny <= 0 or x.deltaF <= 0 or y.deltaF <= 0:
        print("error: negative value for Nx, Ny, x.deltaF or y.deltaF")

    if y.deltaF < x.deltaF:
        print("error: Frequency spacing of coarse series is smaller than fine series")

    if yLowerEdge < xLowerEdge:
        print("error: Start of coarse series is less than start of fine series")

    if yUpperEdge > xUpperEdge:
        print("error: End of coarse series is more than end of fine series")

    # xlow is the index of the last bin whose lower edge is <= the
    # lower edge of the coarse-grained sequence, that is
    # x(xlow) <= y(1) < x(xlow+1)
    xlow = int(np.floor((yLowerEdge - xLowerEdge) / x.deltaF))

    # xhi is the index of the last bin whose upper edge is >= the
    # lower edge of the coarse-grained sequence, that is
    # x(xhi-1) < y(end) <= x(xhi)
    xhi = int(np.ceil((yUpperEdge - xLowerEdge) / x.deltaF)) - 1

    # xi is the array of frequencies of the lower edge of each bin, that is,
    # xi(k) is the lower edge of bin k, which is
    # x.flow + (k-1)*x.deltaF - 0.5*x.deltaF = x.flow + (k-1.5)*x.deltaF
    # This is only calculated for the bins that the coarse series overlaps with,
    # that is, bins [xlow:xhi]
    # print('x.freqs[0] =', x.freqs[0])
    xi = x.freqs[0] + x.deltaF * np.arange(xlow - 0.5, xhi + 0.5 + 0.01, 1)

    # Integrate the original function so that the value fi(k) is
    # the integral from the lower edge of the fine series xi(1)
    # to xi(k). Since each bin represents the average value of the PSD
    # we just have to sum them to get the integral (with the appropriate df).
    # The 0 is inserted so that the fi(1) is 0, as it should be,
    # so that we can interpolate the integral to the coarse series
    # print('xlow =', xlow, 'xhi =', xhi)
    # print('x.data =', x.data[xlow:xhi+1])
    fi = x.deltaF * np.cumsum(x.data[xlow : xhi + 1])
    fi = np.insert(fi, 0, 0)

    # Interpolate the integrated function using the lower edges of each bin in
    # the coarse series as the ordinates
    # print('xi =', xi)
    # print('fi =', fi)
    y.data = np.interp(yi, xi, fi)

    # Take the difference to obtain the integrals over each individual bin of the
    # coarse series and divide by deltaF to get the average value. Then y.data(k)
    # is the average value of the PSD over the interval y(k) to y(k+1) as desired
    y.data = (1 / y.deltaF) * np.diff(y.data)

    return y


def slice_time_series(timeseries, istart, iend):
    return TimeSeries(timeseries.times[istart:iend], timeseries.data[istart:iend])


def welch_psd(data, window="hann", nperseg=None, fs=None):
    """
    Inputs
    * data (assumed to have a length that is a power of 2)
    * window (take to be hann)
    * nperseg = Length of each segment in the PSD (TAvg/deltaT=TAvg*Fs)
    * fs = sampling rate in Hz

    Output
    * f = frequency array [from 0 to f_nyquist by df = 1/TAvg]
    * psd = the psd
    """

    # useful quantities
    NSamples = len(data)
    NAvgs = 2 * int(NSamples / nperseg) - 1  # 50% overlapping
    deltaT = 1 / fs
    TAvg = nperseg * deltaT
    stride = int(nperseg / 2)  # 50% overlapping
    fNyquist = fs / 2

    # frequency array
    fmin = 0
    deltaF = 1 / TAvg
    fmax = fNyquist
    epsilon = deltaF / 100
    freqs = np.arange(fmin, fmax + epsilon, deltaF)

    # psd estimate
    psd = np.zeros(len(freqs))

    for nn in range(NAvgs):  ## nn goes from 0 to NAvgs-1
        # slice data
        istart = stride * nn
        iend = istart + nperseg
        subdata = data[istart:iend]

        # window
        subdata = np.hanning(nperseg) * subdata

        # fft
        subdataTilde = np.fft.rfft(subdata) * deltaT

        # psd = |fft|^2 * 2 / T
        psd = psd + 1 / NAvgs * (np.abs(subdataTilde) ** 2 * 2 / TAvg)

    windowFactor = 1 / nperseg * (np.sum(np.hanning(nperseg) ** 2))
    psd = psd / windowFactor

    freqs = freqs[:-1]
    psd = psd[:-1]

    return freqs, psd


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
