import json
import warnings
from os import listdir
from os.path import isfile, join
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

matplotlib.rcParams['figure.figsize'] = (8,6)
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linestyle'] = ':'
matplotlib.rcParams['grid.color'] = 'grey'
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['legend.handlelength'] = 3

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# FIXME - changed to get workflow running
rc('text', usetex=False)

import seaborn as sns

sea = sns.color_palette("tab10")

import numpy as np
from scipy import integrate, stats
from scipy.optimize import curve_fit

from pygwb.baseline import Baseline
from pygwb.parameters import Parameters
from pygwb.util import StatKS, calc_bias


class StatisticalChecks(object):
    def __init__(
        self,
        sliding_times_all,
        sliding_omega_all,
        sliding_sigmas_all,
        naive_sigmas_all,
        coherence_spectrum,
        point_estimate_spectrum,
        sigma_spectrum,
        freqs,
        badGPSTimes,
        delta_sigmas,
        plot_dir,
        baseline_name,
        param_file,
        gates_ifo1 = None,
        gates_ifo2 = None,
        file_tag = None,
        legend_fontsize = 16
    ):
        """
        The statistical checks class performs various tests by plotting different quantities and saving this plots. This allows the user to check for consistency with expected results. Concretely, the following tests and plots can be generated: running point estimate, running sigma, (cumulative) point estimate integrand, real and imaginary part of point estimate integrand, FFT of the point estimate integrand, (cumulative) sensitivity, evolution of omega and sigma as a function of time, omega and sigma distribution, KS test, and a linear trend analysis of omega in time. Furthermore, part of these plots compares the values of these quantities before and after the delta sigma cut. Each of these plots can be made by calling the relevant class method (e.g. `plot_running_point_estimate()`).

        Parameters
        ==========
        sliding_times_all: array
            Array of GPS times before the bad GPS times from the delta sigma cut are applied.
        sliding_omega_all: array
            Array of sliding omegas before the bad GPS times from the delta sigma cut are applied.
        sliding_sigmas_all: array
            Array of sliding sigmas before the bad GPS times from the delta sigma cut are applied.
        naive_sigmas_all: array
            Array of naive sigmas before the bad GPS times from the delta sigma cut are applied.
        coherence_spectrum: array
            Array containing a coherence spectrum. Each entry in this array corresponds to the 2-detector coherence spectrum evaluated at the corresponding frequency in the freqs array.

        point_estimate_spectrum: array
            Array containing the point estimate spectrum. Each entry in this array corresponds to the point estimate spectrum evaluated at the corresponding frequency in the freqs array.
        sigma_spectrum: array
            Array containing the sigma spectrum. Each entry in this array corresponds to the sigma spectrum evaluated at the corresponding frequency in the freqs array.
        freqs: array
            Array containing the frequencies.
        badGPStimes: array
            Array of bad GPS times, i.e. times that do not pass the delta sigma cut.
        delta_sigmas: array
            Array containing the value of delta sigma for all times in sliding_times_all.
        plot_dir: str
            String with the path to which the output of the statistical checks (various plots) will be saved.
        baseline_name: str
            Name of the baseline under consideration.
        param_file: str
            String with path to the file containing the parameters that were used for the analysis run.

        Returns
        =======
        Initializes an instance of the statistical checks class.
        """
        self.params = Parameters()
        self.params.update_from_file(param_file)

        self.sliding_times_all = sliding_times_all
        self.days_all = (sliding_times_all - sliding_times_all[0]) / 86400.0
        self.sliding_omega_all = sliding_omega_all
        self.sliding_sigmas_all = sliding_sigmas_all
        self.naive_sigmas_all = naive_sigmas_all
        self.badGPStimes = badGPSTimes
        self.delta_sigmas_all = delta_sigmas
        self.sliding_deviate_all = (
            self.sliding_omega_all - np.nanmean(self.sliding_omega_all)
        ) / self.sliding_sigmas_all
        self.gates_ifo1 = gates_ifo1
        self.gates_ifo2 = gates_ifo2

        self.coherence_spectrum = coherence_spectrum

        self.sigma_spectrum = sigma_spectrum
        self.point_estimate_spectrum = point_estimate_spectrum

        self.plot_dir = Path(plot_dir)

        self.baseline_name = baseline_name
        self.segment_duration = self.params.segment_duration
        self.deltaF = self.params.frequency_resolution
        self.new_sample_rate = self.params.new_sample_rate
        self.deltaT = 1.0 / self.new_sample_rate
        self.fref = self.params.fref
        self.flow = self.params.flow
        self.fhigh = self.params.fhigh

        self.freqs = freqs

        self.alpha = self.params.alpha
        (
            self.sliding_times_cut,
            self.days_cut,
            self.sliding_omega_cut,
            self.sliding_sigma_cut,
            self.naive_sigma_cut,
            self.delta_sigmas_cut,
            self.sliding_deviate_cut,
            self.sliding_deviate_KS,
        ) = self.get_data_after_dsc()
        self.dsc_percent = (len(self.sliding_times_all) - len(self.sliding_times_cut))/len(self.sliding_times_all) * 100
        self.dsc_statement = r"The $\Delta\sigma$ cut removed" + f"{float(f'{self.dsc_percent:.2g}'):g}% of the data."

        (
            self.running_pt_estimate,
            self.running_sigmas,
        ) = self.compute_running_quantities()

        self.time_tag = f"{int(self.sliding_times_all[0])}"+"$-$"+f"{int(self.params.tf)}"

        if file_tag:
            self.file_tag = file_tag
        else:
            self.file_tag = f"{self.sliding_times_all[0]}-{self.params.tf}"

        self.legend_fontsize = legend_fontsize
        self.axes_labelsize = legend_fontsize + 2
        self.title_fontsize = legend_fontsize + 4
        self.annotate_fontsize = legend_fontsize - 4

    def get_data_after_dsc(self):
        """
        Function that returns the GPS times, the sliding omegas, the sliding sigmas, the naive sigmas, the delta sigmas and the sliding deviates after the bad GPS times from the delta sigma cut were applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `sliding_times_all`).

        Returns
        =======
        sliding_times_cut: array
            Array of GPS times after the bad GPS times were applied.
        days_cut: array
            Array of days after the bad GPS times were applied.
        sliding_omega_cut: array
            Array of the sliding omega values after the bad GPS times were applied.
        sliding_sigma_cut: array
            Array of sliding sigmas after the bad GPS times were applied.
        naive_sigma_cut: array
            Array of naive sigmas after the bad GPS times were applied.
        delta_sigma_cut: array
            Array of the delta sigma values after the bad GPS times were applied.
        sliding_deviate_cut: array
            Array of the deviates after the bad GPS times were applied.
        """
        bad_gps_times = self.badGPStimes
        bad_gps_mask = [(t not in bad_gps_times) for t in self.sliding_times_all]

        sliding_times_cut = self.sliding_times_all.copy()
        days_cut = self.days_all.copy()
        sliding_omega_cut = self.sliding_omega_all.copy()
        sliding_sigma_cut = self.sliding_sigmas_all.copy()
        naive_sigma_cut = self.naive_sigmas_all.copy()
        delta_sigma_cut = self.delta_sigmas_all.copy()
        sliding_deviate_cut = self.sliding_deviate_all.copy()

        sliding_times_cut = sliding_times_cut[bad_gps_mask]
        days_cut = days_cut[bad_gps_mask]
        sliding_omega_cut = sliding_omega_cut[bad_gps_mask]
        sliding_sigma_cut = sliding_sigma_cut[bad_gps_mask]
        naive_sigma_cut = naive_sigma_cut[bad_gps_mask]
        delta_sigma_cut = delta_sigma_cut[bad_gps_mask]
        sliding_deviate_cut = (
            sliding_omega_cut - np.nanmean(self.sliding_omega_all)
        ) / sliding_sigma_cut

        sliding_deviate_KS = (
            sliding_omega_cut - np.nanmean(sliding_omega_cut)
        ) / sliding_sigma_cut

        return (
            sliding_times_cut,
            days_cut,
            sliding_omega_cut,
            sliding_sigma_cut,
            naive_sigma_cut,
            delta_sigma_cut,
            sliding_deviate_cut,
            sliding_deviate_KS,
        )

    def compute_running_quantities(self):
        """
        Function that computes the running point estimate and running sigmas from the sliding point estimate and sliding sigmas. This is done only for the values after the delta sigma cut. This method does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `sliding_sigma_cut`).

        Returns
        =======
        running_pt_estimate: array
            Array containing the values of the running point estimate.
        running_sigmas: array
            Array containing the values of the running sigmas.
        """
        running_pt_estimate = self.sliding_omega_cut.copy()
        running_sigmas = self.sliding_sigma_cut.copy()

        ii = 0
        while ii < self.sliding_times_cut.shape[0] - 1:
            ii += 1
            numerator = running_pt_estimate[ii - 1] / (
                running_sigmas[ii - 1] ** 2
            ) + self.sliding_omega_cut[ii] / (self.sliding_sigma_cut[ii] ** 2)
            denominator = 1.0 / (running_sigmas[ii - 1] ** 2) + 1 / (
                self.sliding_sigma_cut[ii] ** 2
            )
            running_pt_estimate[ii] = numerator / denominator
            running_sigmas[ii] = np.sqrt(1.0 / denominator)

        return running_pt_estimate, running_sigmas

    def compute_ifft_integrand(self):
        """
        Function that computes the inverse Fourier transform of the point estimate integrand. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `point_estimate_integrand`).

        Returns
        =======
        t_array: array
            Array containing the time lag values (in seconds).
        omega_t: array
            Array containing the

        """

        numFreqs = self.point_estimate_spectrum.shape[0]
        freqs = self.flow + self.deltaF * np.arange(0, numFreqs)
        fhigh = self.flow + self.deltaF * numFreqs

        fNyq = 1 / (2 * self.deltaT)

        numFreqs_pre = np.floor(self.flow / self.deltaF) - 1
        f_pre = self.deltaF * np.arange(1, numFreqs_pre + 1)
        numFreqs_post = np.floor((fNyq - fhigh) / self.deltaF)
        f_post = fhigh + self.deltaF * np.arange(0, numFreqs_post)
        fp = np.concatenate((f_pre, freqs, f_post))
        fn = -np.flipud(fp)
        f_tot = np.concatenate((fn, np.array([0]), fp))

        integrand_pre = np.zeros(int(numFreqs_pre))
        integrand_post = np.zeros(int(numFreqs_post))
        integrand_p = np.concatenate(
            (integrand_pre, 0.5 * self.point_estimate_spectrum, integrand_post)
        )

        integrand_n = np.flipud(np.conj(integrand_p))

        integrand_tot = np.concatenate((np.array([0]), integrand_p, integrand_n))

        fft_integrand = np.fft.fftshift(np.fft.fft(self.deltaF * integrand_tot))

        t_array = np.arange(
            -1.0 / (2 * self.deltaF) + self.deltaT, 1.0 / (2 * self.deltaF), self.deltaT
        )
        omega_t = np.flipud(fft_integrand)

        return t_array, omega_t

    def plot_running_point_estimate(self, ymin=None, ymax=None):
        """
        Generates and saves a plot of the running point estimate. The plotted values are the ones after the delta sigma cut. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `days_cut`).

        Parameters
        ==========
        ymin: float
            Minimum value on the y-axis.
        ymax: float
            Maximum value on the y-axis.

        """
        if self.days_cut.size==0:
            return
        fig = plt.figure(figsize=(10, 8))
        plt.plot(
            self.days_cut,
            self.running_pt_estimate,
            ".",
            color="black",
            markersize=2,
            label=self.baseline_name,
        )
        plt.plot(
            self.days_cut,
            self.running_pt_estimate + 1.65 * self.running_sigmas,
            ".",
            color=sea[2],
            markersize=2,
        )
        plt.plot(
            self.days_cut,
            self.running_pt_estimate - 1.65 * self.running_sigmas,
            ".",
            color=sea[0],
            markersize=2,
        )
        plt.grid(True)
        plt.xlim(self.days_cut[0], self.days_cut[-1])
        if ymin and ymax:
            plt.ylim(ymin, ymax)
        plt.xlabel("Days since start of run", size=self.axes_labelsize)
        plt.ylabel(r"Point estimate $\pm 1.65 \sigma$", size=self.axes_labelsize)
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(f'Running point estimate in {self.time_tag}', fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-running_point_estimate.png", bbox_inches='tight'
        )
        plt.close()

    def plot_running_sigma(self):
        """
        Generates and saves a plot of the running sigma. The plotted values are the ones after the delta sigma cut. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `days_cut`).

        """
        if self.days_cut.size==0:
            return
        fig = plt.figure(figsize=(10, 8))
        plt.plot(
            self.days_cut, self.running_sigmas, '.', markersize=2, color=sea[0], label=self.baseline_name
        )
        plt.grid(True)
        plt.yscale("log")
        plt.xlim(self.days_cut[0], self.days_cut[-1])
        plt.xlabel("Days since start of run", size=self.axes_labelsize)
        plt.ylabel(r"$\sigma$", size=self.axes_labelsize)
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(r'Running $\sigma$ ' + f'in {self.time_tag}', fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-running_sigma.png", bbox_inches = 'tight'
        )
        plt.close()

    def plot_IFFT_point_estimate_integrand(self):
        """
        Generates and saves a plot of the IFFT of the point estimate integrand. The IFFT of the point estimate integrand is computed using the method "compute_ifft_integrand". This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `point_estimate_integrand`).
        """
        t_array, omega_array = self.compute_ifft_integrand()
        if len(t_array) != len(omega_array):
            warnings.warn("Times and Omega arrays don't match in the IFFT. No plot could be generated. Investigation is highly recommended.")
            return

        fig = plt.figure(figsize=(10, 8))
        plt.plot(t_array, omega_array, color=sea[0], label=self.baseline_name)
        plt.grid(True)
        plt.xlim(t_array[0], t_array[-1])
        plt.xlabel("Lag (s)", size=self.axes_labelsize)
        plt.ylabel(r"$\Omega$ integrand IFFT", size=self.axes_labelsize)
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(r"$\Omega$ integrand IFFT" + f" in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-IFFT_point_estimate_integrand.png", bbox_inches='tight'
        )
        plt.close()

    def plot_SNR_spectrum(self):
        """
        Generates and saves a plot of the point estimate integrand. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `point_estimate_integrand`).

        """
        if np.isnan(self.point_estimate_spectrum).all() or not np.real(self.point_estimate_spectrum).any():
            return
        plt.figure(figsize=(10, 8))
        plt.semilogy(
            self.freqs,
            abs(self.point_estimate_spectrum / self.sigma_spectrum),
            color=sea[0],
        )
        plt.xlabel("Frequency (Hz)", size=self.axes_labelsize)
        plt.ylabel(r"$|{\rm SNR}(f)|$", size=self.axes_labelsize)
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.xscale("log")
        plt.title(f"Absolute SNR in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-abs_point_estimate_integrand.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_cumulative_SNR_spectrum(self):
        """
        Generates and saves a plot of the cumulative point estimate integrand. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `point_estimate_integrand`).
        """
        cum_pt_estimate = integrate.cumtrapz(
            np.abs(self.point_estimate_spectrum / self.sigma_spectrum), self.freqs
        )
        cum_pt_estimate = cum_pt_estimate / cum_pt_estimate[-1]
        plt.figure(figsize=(10, 8))
        plt.plot(self.freqs[:-1], cum_pt_estimate, color=sea[0])
        plt.xlabel("Frequency (Hz)", size=self.axes_labelsize)
        plt.ylabel(r"Cumulative $|{\rm SNR}(f)|$", size=self.axes_labelsize)
        plt.xscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(f"Cumulative SNR in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-cumulative_SNR_spectrum.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_real_SNR_spectrum(self):
        """
        Generates and saves a plot of the real part of the SNR spectrum. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `point_estimate_spectrum` and `sigma_spectrum`).
        """
        plt.figure(figsize=(10, 8))
        plt.plot(
            self.freqs,
            np.real(self.point_estimate_spectrum / self.sigma_spectrum),
            color=sea[0],
        )
        plt.xlabel("Frequency (Hz)", size=self.axes_labelsize)
        plt.ylabel(r"Re$({\rm SNR}(f))$", size=self.axes_labelsize)
        plt.xscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(f"Real SNR in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-real_SNR_spectrum.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_imag_SNR_spectrum(self):
        """
        Generates and saves a plot of the imaginary part of the SNR spectrum. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `point_estimate_spectrum` and `sigma_spectrum`).
        """
        plt.figure(figsize=(10, 8))
        plt.plot(
            self.freqs,
            np.imag(self.point_estimate_spectrum / self.sigma_spectrum),
            color=sea[0],
        )
        plt.xlabel("Frequency (Hz)", size=self.axes_labelsize)
        plt.ylabel(r"Im$({\rm SNR}(f))$", size=self.axes_labelsize)
        plt.xscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(f"Imaginary SNR in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-imag_SNR_spectrum.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_sigma_spectrum(self):
        """
        Generates and saves a plot of the sigma spectrum. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `sigma_spectrum`).
        """
        if np.isinf(self.sigma_spectrum).all() or not np.real(self.point_estimate_spectrum).any():
            return

        plt.figure(figsize=(10, 8))
        plt.plot(self.freqs, self.sigma_spectrum, color=sea[0])
        plt.xlabel("Frequency (Hz)", size=self.axes_labelsize)
        plt.ylabel(r"$\sigma(f)$", size=self.axes_labelsize)
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(r"Total $\sigma$ spectrum" + f" in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-sigma_spectrum.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_coherence_spectrum(self, flow=None, fhigh=None):
        """
        Generates and saves a plot of the coherence spectrum, if present. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `coherence_spectrum`).
        """
        if self.coherence_spectrum is None or self.coherence_spectrum.size==1:
            return

        flow = flow or self.flow
        fhigh = fhigh or self.fhigh

        resolution = self.freqs[1] - self.freqs[0]
        fftlength = int(1.0 / resolution)
        n_segs = len(self.sliding_omega_cut) * int(np.floor(self.params.segment_duration/(fftlength))-1) #fftlength/2.
        plt.figure(figsize=(10, 8))
        plt.plot(self.freqs, self.coherence_spectrum, color=sea[0])
        plt.axhline(y=1./n_segs,dashes=(4,3),color='black')
        plt.xlim(flow, fhigh)
        plt.xlabel("Frequency (Hz)", size=self.axes_labelsize)
        plt.ylabel(r"coherence spectrum", size=self.axes_labelsize)
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(r"Total coherence spectrum at $\Delta f$ = " + f"{resolution} Hz in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-coherence_spectrum.png",
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.plot(self.freqs, self.coherence_spectrum, color=sea[0])
        plt.axhline(y=1./n_segs,dashes=(4,3),color='black')
        plt.xlim(flow, 200)
        plt.xlabel("Frequency (Hz)", size=self.axes_labelsize)
        plt.ylabel(r"coherence spectrum", size=self.axes_labelsize)
        plt.yscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(r"Total coherence spectrum at $\Delta f$ = " + f"{resolution} Hz in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-coherence_spectrum_zoom.png",
            bbox_inches="tight",
        )
        plt.close()


    def plot_hist_coherence(self):
        r"""
        Generates and saves a histogram of the coherence distribution. The plot shows the data after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class.
        Furthermore, it also saves a text file which contains the frequencies at which outliers of the coherence distribution were identified, i.e. spectral artefacts.
        """
        if self.coherence_spectrum is None or self.coherence_spectrum.size==1:
            return

        coherence = self.coherence_spectrum
        frequencies = self.freqs
        total_bins = 1000
        bins =  np.linspace(0, max(coherence), total_bins)
        alpha = 1
        n_frequencies = len(frequencies)
        delta_coherence = bins[1]-bins[0]
        resolution = frequencies[1] - frequencies[0]
        fftlength = int(1.0 / resolution)
        n_segs = len(self.sliding_omega_cut) * int(np.floor(self.params.segment_duration/(fftlength))-1)
        predicted = alpha * n_frequencies * delta_coherence * n_segs * np.exp(-alpha * n_segs * coherence)
        threshold = np.log(alpha * n_segs * n_frequencies * delta_coherence) / (n_segs * alpha)

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
   
        axs.hist(
            coherence,
            bins,
            color=sea[3],
            ec="k",
            lw = 0.1,
            zorder=1,
        )
        axs.plot(
            coherence, 
            predicted,
            color=sea[0],
            zorder=2,
            alpha = 0.8,
            label="Predicted",
        )
        axs.axvline(
            np.abs(threshold),
            zorder=3,
            color=sea[1],
            linestyle='dashed',
            label="Threshold",
        )

        axs.set_xlabel(r"Coherence", size=self.axes_labelsize)
        axs.set_ylabel(r"Number of bins", size=self.axes_labelsize)
        axs.legend(fontsize=self.legend_fontsize)
        axs.set_yscale("log")
        axs.set_xlim(left= 0)
        axs.set_ylim(0.5,10*predicted[0])
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)
        plt.title(r"Coherence distribution at $\Delta f$ = " + f"{resolution:.5f} Hz in" f" {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-histogram_coherence.png", bbox_inches = 'tight'
        )
        plt.close()

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))      

        axs.hist(
            coherence,
            bins,
            color=sea[3],
            ec="k",
            lw = 0.1,
            zorder=1,
        )
        axs.plot(
            coherence, 
            predicted,
            color=sea[0],
            zorder=2,
            alpha = 0.8,
            label="Predicted",
        )
        axs.axvline(
            np.abs(threshold),
            zorder=3,
            color=sea[1],
            linestyle='dashed',
            label="Threshold",
        )

        axs.set_xlabel(r"Coherence", size=self.axes_labelsize)
        axs.set_ylabel(r"Number of bins", size=self.axes_labelsize)
        axs.legend(fontsize=self.legend_fontsize)
        axs.set_yscale("log")
        axs.set_xlim(0,4*np.abs(threshold))
        axs.set_ylim(0.5,10*predicted[0])
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)

        plt.title(r"Coherence distribution (zoomed) at $\Delta f$ = " + f"{resolution:.5f} Hz in" f" {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-histogram_coherence_zoom.png", bbox_inches = 'tight'
        )
        plt.close()


        outlier_coherence = [(frequencies[i], coherence[i]) for i in range(len(coherence)) if coherence[i] > np.abs(threshold)]
        n_outlier = len(outlier_coherence)
        file_name = f"{self.plot_dir / self.baseline_name}-{self.file_tag}-list_coherence_outlier.txt"
        with open(file_name, 'w') as f:
            f.write('Frequencies  \tCoherence\n')
            for tup in outlier_coherence:
                f.write(f'{tup[0]}\t{tup[1]}\n')
                
    def plot_cumulative_sensitivity(self):
        """
        Generates and saves a plot of the cumulative sensitivity. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `sigma_spectrum`).

        """
        if np.isinf(self.sigma_spectrum).all() or not np.real(self.point_estimate_spectrum).any():
            return

        cumul_sens = integrate.cumtrapz((1 / self.sigma_spectrum ** 2), self.freqs)
        cumul_sens = cumul_sens / cumul_sens[-1]
        plt.figure(figsize=(10, 8))
        plt.plot(self.freqs[:-1], cumul_sens, color=sea[0])
        plt.xlabel("Frequency (Hz)", size=self.axes_labelsize)
        plt.ylabel("Cumulative sensitivity", size=self.axes_labelsize)
        plt.xscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(r"Cumulative sensitivity $1/\sigma^2$ " + f"in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-cumulative_sigma_spectrum.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_omega_sigma_in_time(self):
        r"""
        Generates and saves a panel plot with a scatter plot of :math:`\sigma` vs :math:`\Delta{\rm SNR}_i`, as well as the evolution of :math:`\Omega`, :math:`\sigma`, and :math:`(\Omega-\langle\Omega\rangle)/\sigma` as a function of the days since the start of the run. All plots show the data before and after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `sliding_sigmas_all`).
        """
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 15), constrained_layout=True)
        fig.suptitle(r"$\Omega$, $\sigma$, and" + f" SNR variations in {self.time_tag} with/out " + r"$\Delta\sigma$ cut", fontsize=self.title_fontsize)

        axs[0].plot(self.days_all, self.sliding_omega_all, color=sea[3], linewidth=1, alpha=0.5, label="All data")
        axs[0].plot(
            self.days_cut,
            self.sliding_omega_cut,
            color=sea[0], linewidth=1, alpha=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
        )
        axs[0].plot(self.days_all, self.sliding_omega_all, '.', color=sea[3])
        axs[0].plot(
            self.days_cut,
            self.sliding_omega_cut, '.',
            color=sea[0],
        )
        axs[0].set_xlabel("Days since start of run", size=self.axes_labelsize)
        axs[0].set_ylabel(r"$\Omega$", size=self.axes_labelsize)
        axs[0].legend(loc="upper left", fontsize=self.legend_fontsize)
        axs[0].set_xlim(0, self.days_all[-1])
        axs[0].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[0].tick_params(axis="y", labelsize=self.legend_fontsize)
        axs[0].yaxis.offsetText.set_fontsize(self.legend_fontsize)

        axs[1].plot(self.days_all, self.sliding_sigmas_all, color=sea[3], linewidth=1, alpha=0.5, label="All data")
        axs[1].plot(
            self.days_cut,
            self.sliding_sigma_cut,
            color=sea[0],
            linewidth=1,
            alpha=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
        )
        axs[1].plot(self.days_all, self.sliding_sigmas_all,'.', color=sea[3])
        axs[1].plot(
            self.days_cut,
            self.sliding_sigma_cut,'.',
            color=sea[0]
        )
        axs[1].set_xlabel("Days since start of run", size=self.axes_labelsize)
        axs[1].set_ylabel(r"$\sigma$", size=self.axes_labelsize)
        axs[1].legend(loc="upper left", fontsize=self.legend_fontsize)
        axs[1].set_xlim(0, self.days_all[-1])
        axs[1].set_yscale('log')
        axs[1].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[1].tick_params(axis="y", labelsize=self.legend_fontsize)
        axs[1].yaxis.offsetText.set_fontsize(self.legend_fontsize)

        axs[2].plot(self.days_all, self.sliding_deviate_all, color=sea[3], linewidth=1, alpha=0.5, label="All data")
        axs[2].plot(
            self.days_cut,
            self.sliding_deviate_cut,
            color=sea[0], linewidth=1, alpha=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
        )
        axs[2].plot(self.days_all, self.sliding_deviate_all, '.', color=sea[3])
        axs[2].plot(
            self.days_cut,
            self.sliding_deviate_cut, '.',
            color=sea[0],
        )
        axs[2].set_xlabel("Days since start of run", size=self.axes_labelsize)
        axs[2].set_ylabel(r"$\Delta{\rm SNR}_i$", size=self.axes_labelsize)
        axs[2].legend(loc="upper left", fontsize=self.legend_fontsize)
        axs[2].set_xlim(0, self.days_all[-1])
        axs[2].set_yscale("symlog")
        axs[2].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[2].tick_params(axis="y", labelsize=self.legend_fontsize)

        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-omega_sigma_time.png", bbox_inches='tight'
        )
        plt.close()

    def plot_hist_sigma_dsc(self):
        r"""
        Generates and saves a panel plot with a histogram of :math:`|\Delta\sigma|/\sigma`, as well as a histogram of :math:`\sigma`. Both plots show the data before and after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `delta_sigmas_all`).

        """
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 14), constrained_layout=True)
        fig.suptitle(r"$\Delta\sigma$ and $\sigma$ distributions in" f" {self.time_tag} with/out " + r"$\Delta\sigma$ cut", fontsize=self.title_fontsize)

        axs[0].hist(
            self.delta_sigmas_all,
            bins=80,
            color=sea[3],
            ec="k",
            lw=0.5,
            label="All data",
            range=(0.0001, 1),
        )
        axs[0].hist(
            self.delta_sigmas_cut,
            bins=80,
            color=sea[0],
            alpha = 0.6,
            ec="k",
            lw=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
            range=(0.0001, 1),
        )
        axs[0].set_xlabel(r"$|\Delta\sigma|/\sigma$", size=self.axes_labelsize)
        axs[0].set_ylabel(r"count", size=self.axes_labelsize)
        axs[0].legend(fontsize=self.legend_fontsize)
        axs[0].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[0].tick_params(axis="y", labelsize=self.legend_fontsize)
        axs[0].yaxis.offsetText.set_fontsize(self.legend_fontsize)

        if self.sliding_sigma_cut.size==0:
            minx1 = min(self.sliding_sigmas_all)
            maxx1 = max(self.sliding_sigmas_all)
        else:
            minx1 = min(self.sliding_sigma_cut)
            maxx1 = max(self.sliding_sigma_cut)
        nx = 50

        axs[1].hist(
            self.sliding_sigmas_all,
            bins=nx,
            color=sea[3],
            ec="k",
            lw=0.5,
            label="All data",
            range=(minx1, maxx1),
        )
        axs[1].hist(
            self.sliding_sigma_cut,
            bins=nx,
            color=sea[0],
            alpha = 0.6,
            ec="k",
            lw=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
            range=(minx1, maxx1),
        )
        axs[1].set_xlabel(r"$\sigma$", size=self.axes_labelsize)
        axs[1].set_ylabel(r"count", size=self.axes_labelsize)
        axs[1].legend(fontsize=self.legend_fontsize)
        axs[1].set_yscale("log")
        axs[1].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[1].tick_params(axis="y", labelsize=self.legend_fontsize)
        axs[1].xaxis.offsetText.set_fontsize(self.legend_fontsize)

        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-histogram_sigma_dsc.png", bbox_inches='tight'
        )
        plt.close()

    def plot_scatter_sigma_dsc(self):
        """
        Generates and saves a scatter plot of :math:`|\Delta\sigma]/\sigma` vs :math:`\sigma`. The plot shows the data before and after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `delta_sigmas_all`).
        """
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

        axs.scatter(
            self.delta_sigmas_all,
            self.naive_sigmas_all,
            marker=".",
            color=sea[3],
            label="All data",
            s=5,
        )
        axs.scatter(
            self.delta_sigmas_cut,
            self.naive_sigma_cut,
            marker=".",
            color=sea[0],
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",

            s=5,
        )
        axs.set_xlabel(r"$|\Delta\sigma|/\sigma$", size=self.axes_labelsize)
        axs.set_ylabel(r"$\sigma$", size=self.axes_labelsize)
        axs.set_yscale("log")
        axs.set_xscale("log")
        axs.legend(fontsize=self.legend_fontsize)
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)

        axs.annotate(
            r"Data cut by $\Delta\sigma$ cut"+f": {float(f'{self.dsc_percent:.2g}'):g}%",
            xy=(0.05, 0.8),
            xycoords="axes fraction",
            size = self.annotate_fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=1),
        )
        plt.title(r"$\Delta\sigma$ distribution in" f" {self.time_tag} with/out " + r"$\Delta\sigma$ cut", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-scatter_sigma_dsc.png", bbox_inches = 'tight'
        )
        plt.close()

    def plot_scatter_omega_sigma_dsc(self):
        r"""
        Generates and saves a panel plot with scatter plots of :math:`|\Delta\sigma|/\sigma` vs :math:`\Delta{\rm SNR}_i`, as well as :math:`\sigma` vs :math:`(\Omega-\langle\Omega\rangle)/\sigma`. All plots show the data before and after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `delta_sigmas_all`).
        """
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 13), constrained_layout=True)
        fig.suptitle(r"$\Delta$SNR spread" + f" in {self.time_tag} with/out " + r"$\Delta\sigma$ cut", fontsize=self.title_fontsize)

        if self.delta_sigmas_cut.size==0:
            maxx0 = max(self.delta_sigmas_all)
            maxx0 += maxx0 / 10.0
            minx0 = min(self.delta_sigmas_all)
            minx0 -= minx0 / 10.0
            maxy0 = np.nanmax(self.sliding_deviate_all)
            maxy0 += maxy0 / 10.0
            miny0 = np.nanmin(self.sliding_deviate_all)
            miny0 -= miny0 / 10.0

            maxx1 = max(self.sliding_sigmas_all)
            maxx1 += maxx1 / 10.0
            minx1 = min(self.sliding_sigmas_all)
            minx1 -= minx1 / 10.0
            maxy1 = max(self.sliding_deviate_all)
            maxy1 += maxy1 / 10.0
            miny1 = min(self.sliding_deviate_all)
            miny1 -= miny1 / 10.0

        else:
            maxx0 = max(self.delta_sigmas_cut)
            maxx0 += maxx0 / 10.0
            minx0 = min(self.delta_sigmas_cut)
            minx0 -= minx0 / 10.0
            maxy0 = np.nanmax(self.sliding_deviate_cut)
            maxy0 += maxy0 / 10.0
            miny0 = np.nanmin(self.sliding_deviate_cut)
            miny0 -= miny0 / 10.0

            maxx1 = max(self.sliding_sigma_cut)
            maxx1 += maxx1 / 10.0
            minx1 = min(self.sliding_sigma_cut)
            minx1 -= minx1 / 10.0
            maxy1 = max(self.sliding_deviate_cut)
            maxy1 += maxy1 / 10.0
            miny1 = min(self.sliding_deviate_cut)
            miny1 -= miny1 / 10.0

        axs[0].scatter(
            self.delta_sigmas_all,
            self.sliding_deviate_all,
            marker=".",
            color=sea[3],
            label="All data",
            s=3,
        )
        axs[0].scatter(
            self.delta_sigmas_cut,
            self.sliding_deviate_cut,
            marker=".",
            color=sea[0],
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
            s=3,
        )
        axs[0].set_xlabel(r"$|\Delta\sigma|/\sigma$", size=self.axes_labelsize)
        axs[0].set_ylabel(r"$\Delta{\rm SNR}_i$", size=self.axes_labelsize)
        axs[0].set_xlim(minx0, maxx0)
        axs[0].set_ylim(miny0, maxy0)
        axs[0].legend(fontsize=self.legend_fontsize)
        axs[0].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[0].tick_params(axis="y", labelsize=self.legend_fontsize)


        axs[1].scatter(
            self.sliding_sigmas_all,
            self.sliding_deviate_all,
            marker=".",
            color=sea[3],
            label="All data",
            s=3,
        )
        axs[1].scatter(
            self.sliding_sigma_cut,
            self.sliding_deviate_cut,
            marker=".",
            color=sea[0],
            label=r"Data after $|\Delta\sigma/\sigma|$ outlier cut",
            s=3,
        )
        axs[1].set_xlabel(r"$\sigma$", size=self.axes_labelsize)
        axs[1].set_ylabel(r"$\Delta{\rm SNR}_i$", size=self.axes_labelsize)
        axs[1].legend(fontsize=self.legend_fontsize)
        axs[1].set_xlim(minx1, maxx1)
        axs[1].set_ylim(miny1, maxy1)
        axs[1].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[1].tick_params(axis="y", labelsize=self.legend_fontsize)
        axs[1].xaxis.offsetText.set_fontsize(self.legend_fontsize)

        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-scatter_omega_sigma_dsc.png", bbox_inches = 'tight')
        plt.close()

    def plot_hist_omega_pre_post_dsc(self):
        r"""
        Generates and saves a histogram of the :math:`\Delta{\rm SNR}_i` distribution. The plot shows the data before and after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `sliding_deviate_all`).
        """
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

        # nan-safing the histograms for good measure...
        bins=np.histogram(np.hstack((self.sliding_deviate_all[~np.isnan(self.sliding_deviate_all)], self.sliding_deviate_cut[~np.isnan(self.sliding_deviate_cut)])), bins=202)[1]
        
        axs.hist(
            self.sliding_deviate_all,
            bins,
            color=sea[3],
            ec="k",
            lw=0.5,
            label="All data",
        )
        axs.hist(
            self.sliding_deviate_cut,
            bins,
            color=sea[0],
            alpha = 0.6,
            ec="k",
            lw=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
        )
        
        axs.set_xlabel(r"$\Delta{\rm SNR}_i$", size=self.axes_labelsize)
        axs.set_ylabel(r"count", size=self.axes_labelsize)
        axs.legend(fontsize=self.legend_fontsize)
        axs.set_yscale("log")
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)

        plt.title(r"$\Delta$SNR distribution in" f" {self.time_tag} with/out " + r"$\Delta\sigma$ cut", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-histogram_omega_dsc.png", bbox_inches = 'tight'
        )
        plt.close()

    def plot_KS_test(self, bias_factor=None):
        """
        Generates and saves a panel plot with results of the Kolmogorov-Smirnov test for Gaussianity. The cumulative distribution of the data (after the delta-sigma (bad GPS times) cut) is compared to the one of Gaussian data, where the bias factor for the sigmas is taken into account. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `sliding_deviate_cut`).

        Parameters
        ==========
        bias_factor: float
            Bias factor to consider in the KS calculation.

        """
        if self.delta_sigmas_cut.size==0:
            return

        if bias_factor is None:
            bias_factor = calc_bias(self.segment_duration, self.deltaF, self.deltaT)
        dof_scale_factor = 1.0 / (1.0 + 3.0 / 35.0)
        lx = len(self.sliding_deviate_cut)

        sorted_deviates = np.sort(self.sliding_deviate_KS / bias_factor)

        nanmask = ~np.isnan(sorted_deviates)
        sorted_deviates_nansafe = sorted_deviates[nanmask]
        count, bins_count = np.histogram(sorted_deviates_nansafe, bins=500)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)

        normal = stats.norm(0, 1)
        normal_cdf = normal.cdf(bins_count[1:])

        dks_x = max(abs(cdf - normal_cdf))
        lx_eff = lx * dof_scale_factor

        lam = (np.sqrt(lx_eff) + 0.12 + 0.11 / np.sqrt(lx_eff)) * dks_x
        pval_KS = StatKS(lam)

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), constrained_layout=True)
        fig.suptitle(f"Kolmogorov-Smirnov test in {self.time_tag}", fontsize=self.title_fontsize)

        axs[0].plot(bins_count[1:], cdf, "k", label="Data")
        axs[0].plot(
            bins_count[1:],
            normal_cdf,
            color=sea[3],
            label=r"Erf with $\sigma$=" + str(round(bias_factor, 2)),
        )
        axs[0].text(
            bins_count[1],
            0.8,
            "KS-statistic: "
            + str(round(dks_x, 3))
            + "\n"
            + "p-value: "
            + str(round(pval_KS, 3)),
        )
        axs[0].legend(loc="lower right", fontsize=self.legend_fontsize)
        axs[0].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[0].tick_params(axis="y", labelsize=self.legend_fontsize)

        axs[1].plot(
            bins_count[1:],
            cdf - normal_cdf,
        )
        axs[1].annotate(
            "Maximum absolute difference: " + str(round(dks_x, 3)),
            xy=(0.025, 0.9),
            xycoords="axes fraction",
            size = self.annotate_fontsize
        )
        axs[1].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[1].tick_params(axis="y", labelsize=self.legend_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-KS_test.png", bbox_inches = 'tight'
        )
        plt.close()

    def plot_hist_sigma_squared(self):
        """
        Generates and saves a histogram of :math:`\sigma^2/\langle\sigma^2\rangle`. The plot shows data after the delta-sigma (bad GPS times) cut. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.sliding_sigma_cut).

        Parameters
        ==========

        Returns
        =======

        """
        if self.delta_sigmas_cut.size==0:
            return

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        axs.hist(
            1 / np.nanmean(self.sliding_sigma_cut ** 2) * self.sliding_sigma_cut ** 2,
            bins=101,
            color=sea[0],
            ec="k",
            lw=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
        )
        axs.set_xlabel(r"$\sigma^2/\langle\sigma^2\rangle$", size=self.axes_labelsize)
        axs.set_ylabel(r"count", size=self.axes_labelsize)
        axs.set_yscale("log")
        axs.set_xlim(0, 5)
        axs.legend(fontsize=self.legend_fontsize)
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)

        plt.title(f"Relative variance in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-histogram_sigma_squared.png", bbox_inches = 'tight'
        )
        plt.close()

    def plot_omega_time_fit(self):
        """
        Generates and saves a plot of :math:`\Omega` as a function of time and fits the data to perform a linear trend analysis. The plot shows data after the delta-sigma (bad GPS times) cut. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.sliding_omega_cut).

        Parameters
        ==========

        Returns
        =======

        """
        if self.days_cut.size==0:
            return
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

        t_obs = self.days_cut[-1]
        scale = np.sqrt(np.var(self.sliding_omega_cut))

        def func(x, a, b):
            return a * (x - t_obs / 2) * t_obs + b

        nanmask = ~np.isnan(self.sliding_omega_cut)
        sliding_omega_cut_nansafe = self.sliding_omega_cut[nanmask]
        sliding_sigma_cut_nansafe = self.sliding_sigma_cut[nanmask]

        popt, pcov = curve_fit(
            func,
            self.days_cut[nanmask],
            sliding_omega_cut_nansafe,
            sigma=sliding_sigma_cut_nansafe,
        )
        c1, c2 = popt[0], popt[1]
        axs.plot(self.days_cut, func(self.days_cut, c1, c2), color=sea[3])
        axs.plot(self.days_cut, self.sliding_omega_cut, '.', color=sea[0], markersize=1)
        axs.plot(self.days_cut, 3 * self.sliding_sigma_cut, color=sea[0], linewidth=1.5)
        axs.plot(self.days_cut, -3 * self.sliding_sigma_cut, color=sea[0], linewidth=1.5)
        axs.set_xlabel("Days since start of run", size=self.axes_labelsize)
        axs.set_ylabel(r"$\Omega_i$", size=self.axes_labelsize)
        axs.set_xlim(self.days_cut[0], self.days_cut[-1])
        axs.annotate(
            r"Linear trend analysis: $\Omega(t) = C_1 (t-T_{\rm obs}/2) T_{\rm obs} + C_2 C_1 = $"
            + str(f"{c1:.3e}")
            + "\nC$_2$ = "
            + str(f"{c2:.3e}"),
            xy=(0.05, 0.05),
            xycoords="axes fraction",
            size = self.annotate_fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=1),
        )
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)
        axs.xaxis.offsetText.set_fontsize(self.legend_fontsize)
        plt.title(r"Time evolution of $\Omega$ " + f"in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-omega_time_fit.png", bbox_inches = 'tight'
        )
        plt.close()

    def plot_sigma_time_fit(self):
        """
        Generates and saves a plot of :math:`\sigma` as a function of time and fits the data to perform a linear trend analysis. The plot shows data after the delta-sigma (bad GPS times) cut. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `sliding_sigma_cut`).
        """
        if self.days_cut.size==0:
            return

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

        t_obs = self.days_cut[-1]
        scale = np.sqrt(np.var(self.sliding_sigma_cut))

        def func(x, a, b):
            return a * (x - t_obs / 2) * t_obs + b

        nanmask = ~np.isnan(self.sliding_sigma_cut)
        sliding_sigma_cut_nansafe = self.sliding_sigma_cut[nanmask]
        mean_sigma = np.nanmean(self.sliding_sigma_cut)

        popt, pcov = curve_fit(
            func,
            self.days_cut[nanmask],
            sliding_sigma_cut_nansafe,
        )
        c1, c2 = popt[0], popt[1]
        axs.plot(self.days_cut, func(self.days_cut, c1, c2), color=sea[3])
        axs.plot(self.days_cut, self.sliding_sigma_cut, ".", color=sea[0], markersize=1)
        axs.set_xlabel("Days since start of run", size=self.axes_labelsize)
        axs.set_ylabel(r"$\sigma_i$", size=self.axes_labelsize)
        axs.set_xlim(self.days_cut[0], self.days_cut[-1])
        axs.set_ylim(mean_sigma - 1.2 * mean_sigma, mean_sigma + 2.2 * mean_sigma)
        axs.annotate(
            r"Linear trend analysis: $\sigma(t) = C_1 (t-T_{\rm obs}/2) T_{\rm obs} + C_2 C_1 = $"
            + str(f"{c1:.3e}")
            + "\nC$_2$ = "
            + str(f"{c2:.3e}"),
            xy=(0.05, 0.05),
            xycoords="axes fraction",
            size = self.annotate_fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=1),
        )
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)
        plt.title(r"Time evolution of $\sigma$ " + f"in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-sigma_time_fit.png", bbox_inches = 'tight'
        )
        plt.close()

    def plot_gates_in_time(self):
        if self.gates_ifo1 is None and self.gates_ifo2 is None:
            self.gates_ifo1_statement=None
            self.gates_ifo2_statement=None
            return
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        if self.gates_ifo1 is None:
            self.gates_ifo1_statement=None
        else:
            self.total_gated_time_ifo1 = np.sum(self.gates_ifo1[:,1]-self.gates_ifo1[:,0])
            self.total_gated_percent_ifo1 = self.total_gated_time_ifo1/(int(self.params.tf)- int(self.sliding_times_all[0]))*100
            gate_times_in_days_ifo1 = (np.array(self.gates_ifo1[:,0]) - self.sliding_times_all[0]) / 86400.0
            self.gates_ifo1_statement= f"Data gated out: {self.total_gated_time_ifo1} s\n" f"Percentage: {float(f'{self.total_gated_percent_ifo1:.2g}'):g}%"
            gatefig1 = ax.plot(gate_times_in_days_ifo1, self.gates_ifo1[:,1]-self.gates_ifo1[:,0], 's', color=sea[0], label="IFO1:\n" f"{self.gates_ifo1_statement}")
            first_legend = ax.legend(handles=gatefig1, loc=(0.05,0.75), fontsize = self.axes_labelsize)
            ax.add_artist(first_legend)
        if self.gates_ifo2 is None:
            self.gates_ifo2_statement=None
        else:
            self.total_gated_time_ifo2 = np.sum(self.gates_ifo2[:,1]-self.gates_ifo2[:,0])
            self.total_gated_percent_ifo2 = self.total_gated_time_ifo2/(int(self.params.tf)- int(self.sliding_times_all[0]))*100
            gate_times_in_days_ifo2 = (np.array(self.gates_ifo2[:,0]) - self.sliding_times_all[0]) / 86400.0
            self.gates_ifo2_statement= f"Data gated out: {self.total_gated_time_ifo2} s\n" f"Percentage: {float(f'{self.total_gated_percent_ifo2:.2g}'):g}%"
            gatefig2 = ax.plot(gate_times_in_days_ifo2, self.gates_ifo2[:,1]-self.gates_ifo2[:,0], 's', color=sea[3], label="IFO2:\n" f"{self.gates_ifo2_statement}")
            ax.legend(handles=gatefig2, loc=(0.05, 0.1), fontsize = self.axes_labelsize)
        ax.set_xlabel("Days since start of run", size=self.axes_labelsize)
        ax.set_ylabel("Gate length (s)", size=self.axes_labelsize)
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.title(f"Gates applied to {self.baseline_name} in {self.time_tag}", fontsize=self.title_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.file_tag}-gates_time.png",
            bbox_inches="tight",
        )
        plt.close()
            


    def save_all_statements(self):
        """
        Saves all useful statements gathered throughout the checks to a json file.
        """
        statements = {}
        statements['dsc'] = self.dsc_statement
        statements['gates_ifo1'] = self.gates_ifo1_statement
        statements['gates_ifo2'] = self.gates_ifo2_statement
        with open("stats_statements.json", "w") as outfile:
                json.dump(statements, outfile)

    def generate_all_plots(self):
        """
        Generates and saves all the statistical analysis plots.
        """
        self.plot_running_point_estimate()
        self.plot_running_sigma()
        self.plot_IFFT_point_estimate_integrand()
        self.plot_SNR_spectrum()
        self.plot_cumulative_SNR_spectrum()
        self.plot_real_SNR_spectrum()
        self.plot_imag_SNR_spectrum()
        self.plot_sigma_spectrum()
        self.plot_cumulative_sensitivity()
        self.plot_omega_sigma_in_time()
        self.plot_hist_sigma_dsc()
        self.plot_scatter_sigma_dsc()
        self.plot_scatter_omega_sigma_dsc()
        self.plot_hist_omega_pre_post_dsc()
        self.plot_KS_test()
        self.plot_hist_sigma_squared()
        self.plot_omega_time_fit()
        self.plot_sigma_time_fit()
        self.plot_gates_in_time()
        if self.coherence_spectrum is not None:
            self.plot_coherence_spectrum()
            self.plot_hist_coherence()
        self.save_all_statements()


def sortingFunction(item):
    return float(item[5:].partition("-")[0])


def run_statistical_checks_from_file(
    combine_file_path, dsc_file_path, plot_dir, param_file, legend_fontsize=16, coherence_file_path = None, file_tag = None,
):
    """
    Assumes files are in npz for now. Will generalize later.
    """
    params = Parameters()
    params.update_from_file(param_file)

    spectra_file = np.load(combine_file_path)
    dsc_file = np.load(dsc_file_path)

    badGPStimes = dsc_file["badGPStimes"]
    delta_sigmas = dsc_file["delta_sigmas"]
    sliding_times = dsc_file["times"]
    naive_sigma_all = dsc_file["naive_sigmas"]
    gates_ifo1 = dsc_file["gates_ifo1"]
    gates_ifo2 = dsc_file["gates_ifo2"]
    if gates_ifo1.size==0:
        gates_ifo1=None
    if gates_ifo2.size==0:
        gates_ifo2=None

    sliding_omega_all, sliding_sigmas_all = (
        spectra_file["point_estimates_seg_UW"],
        spectra_file["sigmas_seg_UW"],
    )

    freqs = np.arange(
        0,
        params.new_sample_rate / 2.0 + params.frequency_resolution,
        params.frequency_resolution,
    )
    cut = (params.fhigh >= freqs) & (freqs >= params.flow)

    spectrum_file = np.load(combine_file_path, mmap_mode="r")

    point_estimate_spectrum = spectrum_file["point_estimate_spectrum"]
    sigma_spectrum = spectrum_file["sigma_spectrum"]

    baseline_name = params.interferometer_list[0] + params.interferometer_list[1]

    # select alpha for statistical checks
    delta_sigmas_sel = delta_sigmas.T[1]
    naive_sigmas_sel = naive_sigma_all.T[1]

    if coherence_file_path is not None:
        coherence_spectrum = np.load(coherence_file_path, allow_pickle=True)['coherence']
    else:
        coherence_spectrum = None

    return StatisticalChecks(
        sliding_times,
        sliding_omega_all,
        sliding_sigmas_all,
        naive_sigmas_sel,
        coherence_spectrum,
        point_estimate_spectrum,
        sigma_spectrum,
        freqs[cut],
        badGPStimes,
        delta_sigmas_sel,
        plot_dir,
        baseline_name,
        param_file,
        gates_ifo1,
        gates_ifo2,
        file_tag=file_tag,
        legend_fontsize=legend_fontsize
    )

