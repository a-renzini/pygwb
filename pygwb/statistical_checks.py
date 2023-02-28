from os import listdir
from os.path import isfile, join
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (8,6)
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linestyle'] = ':'
matplotlib.rcParams['grid.color'] = 'grey'
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['legend.handlelength'] = 3

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

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
        point_estimate_spectrum,
        sigma_spectrum,
        freqs,
        badGPSTimes,
        delta_sigmas,
        plot_dir,
        baseline_name,
        param_file,
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

        (
            self.running_pt_estimate,
            self.running_sigmas,
        ) = self.compute_running_quantities()

        self.legend_fontsize = legend_fontsize
        self.axes_labelsize = legend_fontsize + 2
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
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-running_point_estimate.png", bbox_inches='tight'
        )

    def plot_running_sigma(self):
        """
        Generates and saves a plot of the running sigma. The plotted values are the ones after the delta sigma cut. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `days_cut`).

        """
        fig = plt.figure(figsize=(10, 8))
        plt.semilogy(
            self.days_cut, self.running_sigmas, color=sea[0], label=self.baseline_name
        )
        plt.grid(True)
        plt.xlim(self.days_cut[0], self.days_cut[-1])
        plt.xlabel("Days since start of run", size=self.axes_labelsize)
        plt.ylabel(r"$\sigma$", size=self.axes_labelsize)
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-running_sigma.png", bbox_inches = 'tight'
        )

    def plot_IFFT_point_estimate_integrand(self):
        """
        Generates and saves a plot of the IFFT of the point estimate integrand. The IFFT of the point estimate integrand is computed using the method "compute_ifft_integrand". This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `point_estimate_integrand`).
        """
        t_array, omega_array = self.compute_ifft_integrand()

        fig = plt.figure(figsize=(10, 8))
        plt.plot(t_array, omega_array, color=sea[0], label=self.baseline_name)
        plt.grid(True)
        plt.xlim(t_array[0], t_array[-1])
        plt.xlabel("Lag (s)", size=self.axes_labelsize)
        plt.ylabel("IFFT of Integrand of Pt Estimate", size=self.axes_labelsize)
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-IFFT_point_estimate_integrand.png", bbox_inches='tight'
        )

    def plot_SNR_spectrum(self):
        """
        Generates and saves a plot of the point estimate integrand. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `point_estimate_integrand`).

        """
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
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-abs_point_estimate_integrand.png",
            bbox_inches="tight",
        )

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
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-cumulative_SNR_spectrum.png",
            bbox_inches="tight",
        )

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
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-real_SNR_spectrum.png",
            bbox_inches="tight",
        )

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
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-imag_SNR_spectrum.png",
            bbox_inches="tight",
        )

    def plot_sigma_spectrum(self):
        """
        Generates and saves a plot of the sigma spectrum. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `sigma_spectrum`).
        """
        plt.figure(figsize=(10, 8))
        plt.plot(self.freqs, self.sigma_spectrum, color=sea[0])
        plt.xlabel("Frequency (Hz)", size=self.axes_labelsize)
        plt.ylabel(r"$\sigma(f)$", size=self.axes_labelsize)
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-sigma_spectrum.png",
            bbox_inches="tight",
        )

    def plot_cumulative_sensitivity(self):
        """
        Generates and saves a plot of the cumulative sensitivity. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `sigma_spectrum`).

        """

        cumul_sens = integrate.cumtrapz((1 / self.sigma_spectrum ** 2), self.freqs)
        cumul_sens = cumul_sens / cumul_sens[-1]
        plt.figure(figsize=(10, 8))
        plt.plot(self.freqs[:-1], cumul_sens, color=sea[0])
        plt.xlabel("Frequency (Hz)", size=self.axes_labelsize)
        plt.ylabel("Cumulative sensitivity", size=self.axes_labelsize)
        plt.xscale("log")
        plt.xticks(fontsize=self.legend_fontsize)
        plt.yticks(fontsize=self.legend_fontsize)
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-cumulative_sigma_spectrum.png",
            bbox_inches="tight",
        )

    def plot_omega_sigma_in_time(self):
        r"""
        Generates and saves a panel plot with a scatter plot of :math:`\sigma` vs :math:`\Delta{\rm SNR}_i`, as well as the evolution of :math:`\Omega`, :math:`\sigma`, and :math:`(\Omega-\langle\Omega\rangle)/\sigma` as a function of the days since the start of the run. All plots show the data before and after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `sliding_sigmas_all`).
        """
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 18))

        axs[0].plot(self.days_all, self.sliding_omega_all, color=sea[3], label="All data")
        axs[0].plot(
            self.days_cut,
            self.sliding_omega_cut,
            color=sea[0],
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
        )
        axs[0].set_xlabel("Days since start of run", size=self.axes_labelsize)
        axs[0].set_ylabel(r"$\Omega$", size=self.axes_labelsize)
        axs[0].legend(loc="upper left", fontsize=self.legend_fontsize)
        axs[0].set_xlim(0, self.days_all[-1])
        axs[0].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[0].tick_params(axis="y", labelsize=self.legend_fontsize)
        axs[0].yaxis.offsetText.set_fontsize(self.legend_fontsize)

        axs[1].plot(self.days_all, self.sliding_sigmas_all, color=sea[3], label="All data")
        axs[1].plot(
            self.days_cut,
            self.sliding_sigma_cut,
            color=sea[0],
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
        )
        axs[1].set_xlabel("Days since start of run", size=self.axes_labelsize)
        axs[1].set_ylabel(r"$\sigma$", size=self.axes_labelsize)
        axs[1].legend(loc="upper left", fontsize=self.legend_fontsize)
        axs[1].set_xlim(0, self.days_all[-1])
        axs[1].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[1].tick_params(axis="y", labelsize=self.legend_fontsize)
        axs[1].yaxis.offsetText.set_fontsize(self.legend_fontsize)

        axs[2].plot(self.days_all, self.sliding_deviate_all, color=sea[3], label="All data")
        axs[2].plot(
            self.days_cut,
            self.sliding_deviate_cut,
            color=sea[0],
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
        )
        axs[2].set_xlabel("Days since start of run", size=self.axes_labelsize)
        axs[2].set_ylabel(r"$\Delta{\rm SNR}_i$", size=self.axes_labelsize)
        axs[2].legend(loc="upper left", fontsize=self.legend_fontsize)
        axs[2].set_xlim(0, self.days_all[-1])
        axs[2].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[2].tick_params(axis="y", labelsize=self.legend_fontsize)

        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-omega_sigma_time.png", bbox_inches='tight'
        )

    def plot_hist_sigma_dsc(self):
        r"""
        Generates and saves a panel plot with a histogram of :math:`|\Delta\sigma|/\sigma`, as well as a histogram of :math:`\sigma`. Both plots show the data before and after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `delta_sigmas_all`).

        """
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 14))

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
            ec="k",
            lw=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
            range=(0.0001, 1),
        )
        axs[0].set_xlabel(r"$|\Delta\sigma|/\sigma$", size=self.axes_labelsize)
        axs[0].set_ylabel(r"\# per bin", size=self.axes_labelsize)
        axs[0].legend(fontsize=self.legend_fontsize)
        axs[0].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[0].tick_params(axis="y", labelsize=self.legend_fontsize)
        axs[0].yaxis.offsetText.set_fontsize(self.legend_fontsize)

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
            ec="k",
            lw=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
            range=(minx1, maxx1),
        )
        axs[1].set_xlabel(r"$\sigma$", size=self.axes_labelsize)
        axs[1].set_ylabel(r"\# per bin", size=self.axes_labelsize)
        axs[1].legend(fontsize=self.legend_fontsize)
        axs[1].set_yscale("log")
        axs[1].tick_params(axis="x", labelsize=self.legend_fontsize)
        axs[1].tick_params(axis="y", labelsize=self.legend_fontsize)
        axs[1].xaxis.offsetText.set_fontsize(self.legend_fontsize)

        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-histogram_sigma_dsc.png", bbox_inches='tight'
        )

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

        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-scatter_sigma_dsc.png", bbox_inches = 'tight'
        )

    def plot_scatter_omega_sigma_dsc(self):
        r"""
        Generates and saves a panel plot with scatter plots of :math:`|\Delta\sigma|/\sigma` vs :math:`\Delta{\rm SNR}_i`, as well as :math:`\sigma` vs :math:`(\Omega-\langle\Omega\rangle)/\sigma`. All plots show the data before and after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `delta_sigmas_all`).
        """
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 14))

        maxx0 = max(self.delta_sigmas_cut)
        maxx0 += maxx0 / 10.0

        minx0 = min(self.delta_sigmas_cut)
        minx0 -= minx0 / 10.0

        maxy0 = np.nanmax(self.sliding_deviate_cut)
        maxy0 += maxy0 / 10.0
        miny0 = np.nanmin(self.sliding_deviate_cut)
        miny0 -= miny0 / 10.0

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

        maxx1 = max(self.sliding_sigma_cut)
        maxx1 += maxx1 / 10.0

        minx1 = min(self.sliding_sigma_cut)
        minx1 -= minx1 / 10.0

        maxy1 = max(self.sliding_deviate_cut)
        maxy1 += maxy1 / 10.0

        miny1 = min(self.sliding_deviate_cut)
        miny1 -= miny1 / 10.0

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
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-scatter_omega_sigma_dsc.png", bbox_inches = 'tight')

    def plot_hist_omega_pre_post_dsc(self):
        r"""
        Generates and saves a histogram of the :math:`\Delta{\rm SNR}_i` distribution. The plot shows the data before and after the delta-sigma cut (bad GPS times) was applied. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `sliding_deviate_all`).
        """
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

        axs.hist(
            self.sliding_deviate_all,
            bins=101,
            color=sea[3],
            ec="k",
            lw=0.5,
            label="All data",
        )
        axs.hist(
            self.sliding_deviate_cut,
            bins=101,
            color=sea[0],
            ec="k",
            lw=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
        )
        axs.set_xlabel(r"$\Delta{\rm SNR}_i$", size=self.axes_labelsize)
        axs.set_ylabel(r"\# per bin", size=self.axes_labelsize)
        axs.legend(fontsize=self.legend_fontsize)
        axs.set_yscale("log")
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)

        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-histogram_omega_dsc.png", bbox_inches = 'tight'
        )

    def plot_KS_test(self, bias_factor=None):
        """
        Generates and saves a panel plot with results of the Kolmogorov-Smirnov test for Gaussianity. The cumulative distribution of the data (after the delta-sigma (bad GPS times) cut) is compared to the one of Gaussian data, where the bias factor for the sigmas is taken into account. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `sliding_deviate_cut`).

        Parameters
        ==========
        bias_factor: float
            Bias factor to consider in the KS calculation.

        """
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

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
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
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-KS_test.png", bbox_inches = 'tight'
        )

    def plot_hist_sigma_squared(self):
        """
        Generates and saves a histogram of :math:`\sigma^2/\langle\sigma^2\rangle`. The plot shows data after the delta-sigma (bad GPS times) cut. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.sliding_sigma_cut).

        Parameters
        ==========

        Returns
        =======

        """
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        axs.hist(
            1 / np.nanmean(self.sliding_sigma_cut ** 2) * self.sliding_sigma_cut ** 2,
            bins=1500,
            color=sea[0],
            ec="k",
            lw=0.5,
            label=r"Data after $|\Delta\sigma|/\sigma$ outlier cut",
        )
        axs.set_xlabel(r"$\sigma^2/\langle\sigma^2\rangle$", size=self.axes_labelsize)
        axs.set_ylabel(r"\# per bin", size=self.axes_labelsize)
        axs.set_yscale("log")
        axs.set_xlim(0, 5)
        axs.legend(fontsize=self.legend_fontsize)
        axs.tick_params(axis="x", labelsize=self.legend_fontsize)
        axs.tick_params(axis="y", labelsize=self.legend_fontsize)

        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-histogram_sigma_squared.png", bbox_inches = 'tight'
        )

    def plot_omega_time_fit(self):
        """
        Generates and saves a plot of :math:`\Omega` as a function of time and fits the data to perform a linear trend analysis. The plot shows data after the delta-sigma (bad GPS times) cut. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. self.sliding_omega_cut).

        Parameters
        ==========

        Returns
        =======

        """
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
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-omega_time_fit.png", bbox_inches = 'tight'
        )

    def plot_sigma_time_fit(self):
        """
        Generates and saves a plot of :math:`\sigma` as a function of time and fits the data to perform a linear trend analysis. The plot shows data after the delta-sigma (bad GPS times) cut. This function does not require any input parameters, as it accesses the data through the attributes of the class (e.g. `sliding_sigma_cut`).
        """
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
        plt.savefig(
            f"{self.plot_dir / self.baseline_name}-{self.sliding_times_all[0]}-{self.sliding_times_all[-1]}-sigma_time_fit.png", bbox_inches = 'tight'
        )

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


def sortingFunction(item):
    return float(item[5:].partition("-")[0])


def run_statistical_checks_from_file(
    combine_file_path, dsc_file_path, plot_dir, param_file, legend_fontsize=16
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

    return StatisticalChecks(
        sliding_times,
        sliding_omega_all,
        sliding_sigmas_all,
        naive_sigmas_sel,
        point_estimate_spectrum,
        sigma_spectrum,
        freqs[cut],
        badGPStimes,
        delta_sigmas_sel,
        plot_dir,
        baseline_name,
        param_file,
        legend_fontsize=legend_fontsize
    )


def run_statistical_checks_baseline_pickle(
    baseline_directory, combine_file_path, plot_dir, param_file
):
    params = Parameters()
    params.update_from_file(param_file)
    baseline_directory = Path(baseline_directory)

    baseline_list = [
        f
        for f in listdir(baseline_directory)
        if isfile(join(baseline_directory, f))
        if f.startswith("H1")
    ]
    baseline_list.sort(key=sortingFunction)

    baseline_list = np.array(baseline_list)

    file_0 = join(baseline_directory, baseline_list[0])
    baseline_0 = Baseline.load_from_pickle(file_0)

    freqs = baseline_0.frequencies
    baseline_name = baseline_0.name

    bad_GPS_times = np.array([])
    delta_sigmas = []
    naive_sigmas = []
    sliding_times = []

    for baseline in baseline_list:
        print(f"loading baseline file {baseline}...")
        filename = join(baseline_directory, baseline)
        base = Baseline.load_from_pickle(filename)

        bad_GPS_times = np.append(bad_GPS_times, base.badGPStimes)

        delta_sigmas.append(base.delta_sigmas["values"][1])
        naive_sigmas.append(base.delta_sigmas["naive_sigmas"][1])
        sliding_times.append(base.delta_sigmas["times"])

    delta_sigmas = np.concatenate(delta_sigmas)
    naive_sigmas = np.concatenate(naive_sigmas)
    sliding_times = np.concatenate(sliding_times)

    spectrum_file = np.load(combine_file_path, mmap_mode="r")

    sliding_omega_all, sliding_sigmas_all = (
        spectrum_file["point_estimates_seg_UW"],
        spectrum_file["sigmas_seg_UW"],
    )
    point_estimate_spectrum = spectrum_file["point_estimate_spectrum"]
    sigma_spectrum = spectrum_file["sigma_spectrum"]

    return StatisticalChecks(
        sliding_times,
        sliding_omega_all,
        sliding_sigmas_all,
        naive_sigmas,
        point_estimate_spectrum,
        sigma_spectrum,
        freqs,
        bad_GPS_times,
        delta_sigmas,
        plot_dir,
        baseline_name,
        param_file,
    )
