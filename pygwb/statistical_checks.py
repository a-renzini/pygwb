import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate, special
from scipy.optimize import curve_fit

from pygwb.util import StatKS, calc_bias, read_jobfiles


class StatisticalChecks(object):
    def __init__(
        self,
        segment_duration,
        njobs,
        deltaT,
        deltaF,
        dsc,
        ifo_pair,
        mat_dir,
        pproc_dir,
        plot_dir,
        read_bad_times,
        segs=None,
    ):
        """
        Class for the statistical analysis of the results, which generates and saves plots.

        Parameters
        ==========
        segment_duration: int
            Duration of a segment in seconds
        njobs: int
            Number of jobs
        deltaT: float
            Time delta T used in the analysis
        deltaF: float
            Frequency delta f used in the analysis
        dsc: float
            Value used for the delta-sigma cut
        ifo_pair: str
            Name of the baseline. E.g.: "H1L1"
        mat_dir: str
            Directory where the mat files are stored
        pproc_dir: str
            Directory where the output of post-processing is stored
        plot_dir: str
            Directory where the statistical checks plots will be saved
        read_bad_times: boolean
            Boolean indicating whether the bad GPS times (after delta-sigma cut) should be read in from file (i.e. True) or computed on the fly here (i.e. False)
        segs: float
            Parameter used to perform the delta-sigma cut on the fly
        Returns
        =======
        """
        self.plot_dir = plot_dir
        self.pproc_dir = pproc_dir
        self.mat_dir = mat_dir
        self.segment_duration = segment_duration
        self.njobs = njobs
        self.deltaT = deltaT
        self.deltaF = deltaF
        self.dsc = dsc
        self.ifo_pair = ifo_pair
        self.get_data_before_dsc_cut()
        self.get_data_after_dsc_cut(read_bad_times, segs)
        self.get_running_pt_estimate()
        self.get_pt_estimate_integrand()
        self.get_sensitivity_integrand()
        self.get_ifft_integrand()

    def get_data_before_dsc_cut(self):
        """
        Gets the data (such as times, omegas, sigmas) before the delta sigma cut.
        Parameters
        ==========

        Returns
        =======
        """
        (
            self.sliding_times_all,
            self.sliding_omega_all,
            self.sliding_sigmas_all,
            self.naive_sigma_all,
        ) = read_jobfiles(self.njobs, self.mat_dir, self.segment_duration)
        self.days_all = (self.sliding_times_all - self.sliding_times_all[0]) / 86400.0
        self.sliding_deviate_all = (
            self.sliding_omega_all - np.mean(self.sliding_omega_all)
        ) / self.sliding_sigmas_all
        self.delta_sig_all = (
            np.abs(self.naive_sigma_all - self.sliding_sigmas_all)
            / self.naive_sigma_all
        )

    def get_data_after_dsc_cut(self, read_bad_times, segs=None):
        """
        Gets the data (such as times, omegas, sigmas) after the delta sigma cut.
        Parameters
        ==========
        read_bad_times: boolean
            Boolean indicating whether the bad GPS times (after delta-sigma cut) should be read in from file (i.e. True) or computed on the fly here (i.e. False)
        segs: float
            Parameter used to perform the delta-sigma cut on the fly

        Returns
        =======
        """

        self.sliding_omega_cut = np.array([])
        self.sliding_sigmas_cut = np.array([])
        self.delta_sig_cut = np.array([])
        self.naive_sigma_cut = np.array([])
        self.days_cut = np.array([])

        if read_bad_times:
            file_bad = open(f"{self.pproc_dir[:-3]}badGPSTimes.dat", "r")
            lines_bad = file_bad.readlines()
            bad_times = np.array([])
            for line in lines_bad:
                temp = line.strip("\n")
                bad_times = np.append(bad_times, np.array([float(temp)]))

            for ii in range(self.delta_sig_all.shape[0]):
                if self.sliding_times_all[ii] not in bad_times:
                    self.sliding_omega_cut = np.append(
                        self.sliding_omega_cut, self.sliding_omega_all[ii]
                    )
                    self.naive_sigma_cut = np.append(
                        self.naive_sigma_cut, self.naive_sigma_all[ii]
                    )
                    self.sliding_sigmas_cut = np.append(
                        self.sliding_sigmas_cut, self.sliding_sigmas_all[ii]
                    )
                    self.delta_sig_cut = np.append(
                        self.delta_sig_cut, self.delta_sig_all[ii]
                    )
                    self.days_cut = np.append(self.days_cut, self.days_all[ii])

            self.sliding_deviate_cut = (
                self.sliding_omega_cut - np.mean(self.sliding_omega_cut)
            ) / self.sliding_sigmas_cut
        else:
            nn = 2 * 9 / 11 * segs
            bf_ss = nn / (nn - 1)
            nn = 9 / 11 * segs
            bf_ns = nn / (nn - 1)

            for ii in range(self.delta_sig_all.shape[0]):
                if self.delta_sig_all[ii] < dsc:
                    self.sliding_omega_cut = np.append(
                        self.sliding_omega_cut, self.sliding_omega_all[ii]
                    )
                    self.naive_sigma_cut = np.append(
                        self.naive_sigma_cut, self.naive_sigma_all[ii]
                    )
                    self.sliding_sigmas_cut = np.append(
                        self.sliding_sigmas_cut, self.sliding_sigmas_all[ii]
                    )
                    self.delta_sig_cut = np.append(
                        self.delta_sig_cut, self.delta_sig_all[ii]
                    )
                    self.days_cut = np.append(self.days_cut, self.days_all[ii])

            self.sliding_deviate_cut = (
                self.sliding_omega_cut - np.mean(self.sliding_omega_cut)
            ) / self.sliding_sigmas_cut

    def get_running_pt_estimate(self):
        """
        Get the running point estimate from file.
        Parameters
        ==========

        Returns
        =======
        """
        st, self.Y, self.sig = np.loadtxt(
            f"{self.pproc_dir+self.ifo_pair}_runningPointEstimate.dat",
            unpack=True,
            skiprows=3,
        )
        self.days = (st - st[0]) / 86400.0

    def get_pt_estimate_integrand(self):
        """
        Gets the point estimate integrand from file.
        Parameters
        ==========

        Returns
        =======
        """
        filename_pt = self.pproc_dir + self.ifo_pair + "_ptEstIntegrand.dat"
        self.pt_est_integrand = np.genfromtxt(filename_pt, comments="%")
        self.cum_pt_estimate = integrate.cumtrapz(
            self.pt_est_integrand[:, 2], self.pt_est_integrand[:, 1]
        )

    def get_sensitivity_integrand(self):
        """
        Gets the sensitivity integrand from file.
        Parameters
        ==========

        Returns
        =======
        """
        filename_sens = self.pproc_dir + self.ifo_pair + "_sensIntegrand.dat"
        self.sensitivity_integrand = np.genfromtxt(filename_sens, comments="%")
        self.cumul_sensitivity = integrate.cumtrapz(
            self.sensitivity_integrand[:, 2], self.sensitivity_integrand[:, 1]
        )

    def get_ifft_integrand(self):
        """
        Gets the inverse Fourier transform of the FFT point estimate integrand from file.
        Parameters
        ==========

        Returns
        =======

        """
        FFTcontent = [
            i.strip().split()
            for i in open(
                f"{self.pproc_dir+self.ifo_pair}_FFTofPtEstIntegrand.dat"
            ).readlines()[3:]
        ]
        self.t = [float(FFTcontent[i][0]) for i in range(len(FFTcontent))]
        self.omega_t = [float(FFTcontent[i][1]) for i in range(len(FFTcontent))]

    def plot_running_point_estimate(self):
        """
        Generates and saves a plot of the running point estimate.

        Parameters
        ==========

        Returns
        =======

        """
        fig = plt.figure()
        plt.plot(
            self.days, self.Y, ".", color="black", markersize=2, label=self.ifo_pair
        )
        plt.plot(self.days, self.Y + 1.65 * self.sig, ".", color="green", markersize=2)
        plt.plot(self.days, self.Y - 1.65 * self.sig, ".", color="blue", markersize=2)
        plt.grid(True)
        plt.ylim(-5e-7, 5e-7)
        plt.xlim(self.days[0], self.days[-1])
        plt.xlabel("Days since start of run")
        plt.ylabel("Point estimate +/- 1.65\u03C3")
        plt.legend()
        plt.savefig(f"{self.plot_dir}RunningPointEstimate.png")

    def plot_running_sigma(self):
        """
        Generates and saves a plot of the running sigma.

        Parameters
        ==========

        Returns
        =======

        """
        fig = plt.figure()
        plt.semilogy(self.days, self.sig, color="blue", label=self.ifo_pair)
        plt.grid(True)
        plt.ylim(1e-8, 1e-3)
        plt.xlim(self.days[0], self.days[-1])
        plt.xlabel("Days since start of run")
        plt.ylabel("\u03C3")
        plt.legend()
        plt.savefig(f"{self.plot_dir}RunningSigma.png")

    def plot_IFFT_point_estimate_integrand(self):
        """
        Generates and saves a plot of the IFFT of the point estimate integrand.

        Parameters
        ==========

        Returns
        =======

        """
        fig = plt.figure()
        plt.plot(self.t, self.omega_t, color="b", label=self.ifo_pair)
        plt.grid(True)
        plt.ylim(-1.5e-7, 1e-7)
        plt.xlim(-20, 20)
        plt.xlabel("Lag (s)")
        plt.ylabel("IFFT of Integrand of Pt Estimate")
        plt.legend()
        plt.savefig(f"{self.plot_dir}IFFTPointEstimateIntegrand.png")

    def plot_point_estimate_integrand(self):
        """
        Generates and saves a plot of the point estimate integrand.

        Parameters
        ==========

        Returns
        =======

        """
        plt.figure(figsize=(10, 8))
        plt.semilogy(
            self.pt_est_integrand[:, 1],
            np.sqrt(
                abs(self.pt_est_integrand[:, 2] ** 2 + self.pt_est_integrand[:, 3] ** 2)
            ),
        )
        plt.xlabel("Frequency [Hz]", size=18)
        plt.ylabel("Abs(point estimate integrand)", size=18)
        plt.savefig(
            f"{self.plot_dir}AbsPointEstimateIntegrand.png", bbox_inches="tight"
        )

    def plot_cumulative_point_estimate(self):
        """
        Generates and saves a plot of the cumulative point estimate.

        Parameters
        ==========

        Returns
        =======

        """
        plt.figure(figsize=(10, 8))
        plt.plot(self.pt_est_integrand[:-1, 1], self.cum_pt_estimate)
        plt.xlabel("Frequency [Hz]", size=18)
        plt.ylabel("Cumulative point estimate", size=18)
        plt.savefig(f"{self.plot_dir}CumulativePointEstimate.png", bbox_inches="tight")

    def plot_real_point_estimate_integrand(self):
        """
        Generates and saves a plot of the real part of the point estimate integrand.

        Parameters
        ==========

        Returns
        =======

        """
        plt.figure(figsize=(10, 8))
        plt.plot(self.pt_est_integrand[:, 1], self.pt_est_integrand[:, 2])
        plt.xlabel("Frequency [Hz]", size=18)
        plt.ylabel("Re(point estimate integrand)", size=18)
        plt.savefig(
            f"{self.plot_dir}RealPointEstimateIntegrand.png", bbox_inches="tight"
        )

    def plot_imaginary_point_estimate_integrand(self):
        """
        Generates and saves a plot of the imaginary part of the point estimate integrand.

        Parameters
        ==========

        Returns
        =======

        """
        plt.figure(figsize=(10, 8))
        plt.plot(self.pt_est_integrand[:, 1], self.pt_est_integrand[:, 3])
        plt.xlabel("Frequency [Hz]", size=18)
        plt.ylabel("Im(point estimate integrand)", size=18)
        plt.savefig(
            f"{self.plot_dir}ImagPointEstimateIntegrand.png", bbox_inches="tight"
        )

    def plot_sensitivity_integrand(self):
        """
        Generates and saves a plot of the sensitivity integrand.

        Parameters
        ==========

        Returns
        =======

        """
        plt.figure(figsize=(10, 8))
        plt.plot(self.sensitivity_integrand[:, 1], self.sensitivity_integrand[:, 2])
        plt.xlabel("Frequency [Hz]", size=18)
        plt.ylabel("Sensitivity integrand", size=18)
        plt.savefig(f"{self.plot_dir}SensitivityIntegrand.png", bbox_inches="tight")

    def plot_cumulative_sensitivity_integrand(self):
        """
        Generates and saves a plot of the cumulative sensitivity integrand.

        Parameters
        ==========

        Returns
        =======

        """
        plt.figure(figsize=(10, 8))
        plt.plot(self.sensitivity_integrand[:-1, 1], self.cumul_sensitivity)
        plt.xlabel("Frequency [Hz]", size=18)
        plt.ylabel("Cumulative sensitivity", size=18)
        plt.savefig(
            f"{self.plot_dir}CumulativeSensitivityIntegrand.png", bbox_inches="tight"
        )

    def plot_omega_sigma_in_time(self):
        """
        Generates and saves a panel plot with a scatter plot of \u03C3 vs (\u03A9-<\u03A9>)/\u03C3, as well as the evolution of \u03A9, \u03C3, and (\u03A9-<\u03A9>)/\u03C3 as a function of the days since the start of the run. All plots show the data before and after the delta-sigma cut was applied.

        Parameters
        ==========

        Returns
        =======

        """
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
        fig.suptitle(
            "Results from " + self.ifo_pair + " for Abs[\u03B4\u03C3]/\u03C3 cut",
            fontsize=20,
        )

        axs[0, 0].scatter(
            self.sliding_sigmas_all,
            self.sliding_deviate_all,
            marker=".",
            c="r",
            s=3,
            label="All data",
        )
        axs[0, 0].scatter(
            self.sliding_sigmas_cut,
            self.sliding_deviate_cut,
            marker=".",
            c="b",
            s=3,
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
        )
        axs[0, 0].set_xlabel("\u03C3")
        axs[0, 0].set_ylabel("(\u03A9-<\u03A9>)/\u03C3")
        axs[0, 0].legend(loc="upper left")

        axs[0, 1].plot(self.days_all, self.sliding_omega_all, c="r", label="All data")
        axs[0, 1].plot(
            self.days_cut,
            self.sliding_omega_cut,
            c="b",
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
        )
        axs[0, 1].set_xlabel("Days since start of run")
        axs[0, 1].set_ylabel("\u03A9")
        axs[0, 1].legend(loc="upper left")

        axs[1, 0].plot(self.days_all, self.sliding_sigmas_all, c="r", label="All data")
        axs[1, 0].plot(
            self.days_cut,
            self.sliding_sigmas_cut,
            c="b",
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
        )
        axs[1, 0].set_xlabel("Days since start of run")
        axs[1, 0].set_ylabel("\u03C3")
        axs[1, 0].legend(loc="upper left")

        axs[1, 1].plot(self.days_all, self.sliding_deviate_all, c="r", label="All data")
        axs[1, 1].plot(
            self.days_cut,
            self.sliding_deviate_cut,
            c="b",
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
        )
        axs[1, 1].set_xlabel("Days since start of run")
        axs[1, 1].set_ylabel("(\u03A9-<\u03A9>)/\u03C3")
        axs[1, 1].legend(loc="upper left")

        plt.savefig(f"{self.plot_dir}OmegaSigmaTime.png")

    def plot_hist_sigma_dsc(self):
        """
        Generates and saves a panel plot with a histogram of Abs[\u03B4\u03C3]/\u03C3, as well as a histogram of \u03C3. Both plots show the data before and after the delta-sigma cut was applied.

        Parameters
        ==========

        Returns
        =======

        """
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(13, 13))
        fig.suptitle(
            "Results from " + self.ifo_pair + " for Abs[\u03B4\u03C3]/\u03C3",
            fontsize=20,
        )

        axs[0].hist(
            self.delta_sig_all,
            bins=80,
            color="r",
            ec="k",
            lw=0.5,
            label="All data",
            range=(0.0001, 1),
        )
        axs[0].hist(
            self.delta_sig_cut,
            bins=80,
            color="b",
            ec="k",
            lw=0.5,
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
            range=(0.0001, 1),
        )
        axs[0].set_xlabel("Abs[\u03B4\u03C3]/\u03C3")
        axs[0].set_ylabel("# per bin")
        axs[0].set_xlim([0, 1])
        axs[0].legend()

        minx1 = min(self.sliding_sigmas_cut)
        maxx1 = max(self.sliding_sigmas_cut)
        nx = 50

        axs[1].hist(
            self.sliding_sigmas_all,
            bins=nx,
            color="r",
            ec="k",
            lw=0.5,
            label="All data",
            range=(minx1, maxx1),
        )
        axs[1].hist(
            self.sliding_sigmas_cut,
            bins=nx,
            color="b",
            ec="k",
            lw=0.5,
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
            range=(minx1, maxx1),
        )
        axs[1].set_xlabel("\u03C3")
        axs[1].set_ylabel("# per bin")
        axs[1].legend()
        axs[1].set_yscale("log")
        axs[1].set_xlim([0, maxx1])
        axs[1].set_ylim([1, 10 ** 4])

        plt.savefig(f"{self.plot_dir}HistogramSigmaDSC.png")

    def plot_scatter_sigma_dsc(self):
        """
        Generates and saves a scatter plot of Abs[\u03B4\u03C3]/\u03C3 vs \u03C3. The plot shows the data before and after the delta-sigma cut was applied.

        Parameters
        ==========

        Returns
        =======

        """
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        fig.suptitle(
            "Results from " + self.ifo_pair + " for Abs[\u03B4\u03C3]/\u03C3",
            fontsize=20,
        )

        axs.scatter(
            self.delta_sig_all,
            self.naive_sigma_all,
            marker=".",
            c="r",
            label="All data",
            s=5,
        )
        axs.scatter(
            self.delta_sig_cut,
            self.naive_sigma_cut,
            marker=".",
            c="b",
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
            s=5,
        )
        axs.set_xlabel("Abs[\u03B4\u03C3]/\u03C3")
        axs.set_ylabel("\u03C3")
        axs.legend()

        plt.savefig(f"{self.plot_dir}ScatterSigmaDSC.png")

    def plot_scatter_omega_sigma_dsc(self):
        """
        Generates and saves a panel plot with scatter plots of Abs[\u03B4\u03C3]/\u03C3 vs (\u03A9-<\u03A9>)/\u03C3, as well as \u03C3 vs (\u03A9-<\u03A9>)/\u03C3. All plots show the data before and after the delta-sigma cut was applied.

        Parameters
        ==========

        Returns
        =======

        """
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(13, 13))
        fig.suptitle(
            "Results from " + self.ifo_pair + " for Abs[\u03B4\u03C3]/\u03C3",
            fontsize=20,
        )

        axs[0].scatter(
            self.delta_sig_all,
            self.sliding_deviate_all,
            marker=".",
            c="r",
            label="All data",
            s=3,
        )
        axs[0].scatter(
            self.delta_sig_cut,
            self.sliding_deviate_cut,
            marker=".",
            c="b",
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
            s=3,
        )
        axs[0].set_xlabel("Abs[\u03B4\u03C3]/\u03C3")
        axs[0].set_ylabel("(\u03A9-<\u03A9>)/\u03C3")
        axs[0].legend()
        axs[0].set_xlim(0, 2)
        axs[0].set_ylim(-20, 20)

        maxx1 = max(self.sliding_sigmas_cut)

        axs[1].scatter(
            self.sliding_sigmas_all,
            self.sliding_deviate_all,
            marker=".",
            c="r",
            label="All data",
            s=3,
        )
        axs[1].scatter(
            self.sliding_sigmas_cut,
            self.sliding_deviate_cut,
            marker=".",
            c="b",
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
            s=3,
        )
        axs[1].set_xlabel("\u03C3")
        axs[1].set_ylabel("(\u03A9-<\u03A9>)/\u03C3")
        axs[1].set_xlim([0, maxx1])
        axs[1].set_ylim([-20, 20])
        axs[1].legend()

        plt.savefig(f"{self.plot_dir}ScatterOmegaSigmaDSC.png")

    def plot_hist_omega_pre_post_dsc(self):
        """
        Generates and saves a histogram of the (\u03A9-<\u03A9>)/\u03C3 distribution. The plot shows the data before and after the delta-sigma cut was applied.

        Parameters
        ==========

        Returns
        =======

        """
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        fig.suptitle(
            "Results from "
            + self.ifo_pair
            + ": Histogram of (\u03A9-<\u03A9>)/\u03C3 with outlier cuts",
            fontsize=20,
        )

        axs.hist(
            self.sliding_deviate_all,
            bins=101,
            color="r",
            ec="k",
            lw=0.5,
            label="All data",
            range=(-50, 50),
        )
        axs.hist(
            self.sliding_deviate_cut,
            bins=101,
            color="b",
            ec="k",
            lw=0.5,
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
            range=(-50, 50),
        )
        axs.set_xlabel("(\u03A9-<\u03A9>)/\u03C3")
        axs.set_ylabel("# per bin")
        axs.set_xlim(-60, 60)
        axs.set_ylim(1, 1e4)
        axs.legend()
        axs.set_yscale("log")

        plt.savefig(f"{self.plot_dir}HistogramOmegaPrePostDSC.png")

    def plot_KS_test(self):
        """
        Generates and saves a panel plot with results of the Kolmogorov-Smirnov test for Gaussianity. The cumulative distribution of the data is compared to the one of Gaussian data.

        Parameters
        ==========

        Returns
        =======

        """
        bias_factor = calc_bias(self.segment_duration, self.deltaF, self.deltaT)
        dof_scale_factor = 1.0 / (1.0 + 3.0 / 35.0)
        lx = len(self.sliding_deviate_cut)
        cdf_x = 1.0 / lx * np.linspace(1, lx, lx)

        sorted_deviates_U = np.sort(self.sliding_deviate_cut)
        phi_x_U = 0.5 * (
            1.0 + special.erf(sorted_deviates_U / (np.sqrt(2) * bias_factor))
        )
        dks_x = max(abs(cdf_x - phi_x_U))
        lx_eff = lx * dof_scale_factor

        lam = (np.sqrt(lx_eff) + 0.12 + 0.11 / np.sqrt(lx_eff)) * dks_x
        pval_KS = StatKS(lam)

        count, bins_count = np.histogram(self.sliding_deviate_cut, bins=500)

        pdf = count / sum(count)
        cdf = np.cumsum(pdf)

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9, 9))
        fig.suptitle(
            "Results from " + self.ifo_pair + " for Abs[\u03B4\u03C3]/\u03C3",
            fontsize=16,
        )
        axs[0].plot(sorted_deviates_U, cdf_x, "k", label="Data")
        axs[0].plot(
            sorted_deviates_U,
            phi_x_U,
            "r",
            label="Erf with \u03C3=" + str(round(bias_factor, 2)),
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
        axs[0].legend(loc="lower right")

        axs[1].plot(
            bins_count[1:],
            cdf - 0.5 * (1 + special.erf(bins_count[1:] / (np.sqrt(2) * bias_factor))),
        )
        axs[1].annotate(
            "Maximum absolute difference: " + str(round(dks_x, 3)),
            xy=(0.025, 0.9),
            xycoords="axes fraction",
        )
        plt.savefig(f"{self.plot_dir}KSTest.png")

    def plot_hist_scatter_omega_sigma(self):
        """
        Generates and saves a panel plot with a scatter plot of \u03C3 vs (\u03A9-<\u03A9>)/\u03C3, as well as histograms of the \u03C3 and (\u03A9-<\u03A9>)/\u03C3 distributions. All plots show the data after the delta-sigma cut.

        Parameters
        ==========

        Returns
        =======

        """
        muXY = np.cov(self.sliding_sigmas_cut, self.sliding_deviate_cut)
        binsize = 0.2
        bins = np.arange(-5, 5, binsize)
        count, bins_count = np.histogram(self.sliding_deviate_cut, bins=bins)

        bias_factor = calc_bias(self.segment_duration, self.deltaF, self.deltaT)
        dof_scale_factor = 1.0 / (1.0 + 3.0 / 35.0)
        lx = len(self.sliding_deviate_cut)
        cdf_x = 1.0 / lx * np.linspace(1, lx, lx)

        sorted_deviates_U = np.sort(self.sliding_deviate_cut)
        phi_x_U = 0.5 * (
            1.0 + special.erf(sorted_deviates_U / (np.sqrt(2) * bias_factor))
        )
        dks_x = max(abs(cdf_x - phi_x_U))

        muR = np.mean(self.sliding_deviate_cut)
        sigmaR = bias_factor
        theory = np.exp(-((bins_count - muR + binsize / 2) ** 2) / (2 * sigmaR ** 2))
        tmax = 10 ** np.log10(np.max(count))

        nbins = 200
        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
        (ax1, ax2), (ax3, ax4) = gs.subplots(sharex="col", sharey="row")
        ax1.hist(
            self.sliding_sigmas_cut,
            bins=nbins,
            color="b",
            ec="k",
            lw=0.5,
            orientation="vertical",
        )
        ax1.set_ylabel("Relative Frequency, N")

        ax2.axis("off")

        maxx1 = np.max(self.sliding_sigmas_cut) / 4
        ax3.text(
            1.35 * maxx1 / 3,
            2.75,
            "N = "
            + str(len(self.sliding_sigmas_cut))
            + "\n\u03C3$_{XY}^2$ = "
            + str(f"{muXY[0, 0]:0.2e}")
            + "\n\u03C1 = \u03C3$_{XY}^2$/\u03C3$_{X}$\u03C3$_{Y}$ = "
            + str(f"{muXY[0, 1] / np.sqrt(muXY[0, 0] * muXY[1, 1]):0.2e}")
            + "\n\u03A3 (\u03A9$_i$-<\u03A9>)$^2$/\u03C3$_i$$^2$ = "
            + str(np.floor(np.sum(self.sliding_deviate_cut ** 2))),
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.3),
        )
        ax3.scatter(
            self.sliding_sigmas_cut, self.sliding_deviate_cut, marker=".", s=1, c="b"
        )
        ax3.set_xlabel("\u03C3")
        ax3.set_ylabel("(\u03A9-<\u03A9>)/\u03C3")
        ax3.set_xlim(0, maxx1)
        ax3.set_ylim(-6, 6)

        ax4.hist(
            self.sliding_deviate_cut,
            bins=bins,
            color="b",
            ec="k",
            lw=0.5,
            orientation="horizontal",
        )
        ax4.plot(tmax * theory, bins_count, color="r")
        ax4.text(
            10,
            4.4,
            "Fit: $e^{-x^2/(2\u03C3^2)}$; \u03C3 = "
            + str(round(bias_factor, 2))
            + "\nKS-test: "
            + str(round(dks_x, 3)),
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
        ax4.set_xlabel("Relative Frequency, N")

        plt.savefig(f"{self.plot_dir}HistogramScatterOmegaSigma.png")

    def plot_hist_sigma_squared(self):
        """
        Generates and saves a histogram of \u03C3^2/<\u03C3^2>. The plot shows data after the delta-sigma cut.

        Parameters
        ==========

        Returns
        =======

        """
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        fig.suptitle(
            "Results from "
            + self.ifo_pair
            + ": Distribution of \u03C3$^2$/<\u03C3$^2$>",
            fontsize=20,
        )

        axs.hist(
            1 / np.mean(self.sliding_sigmas_cut ** 2) * self.sliding_sigmas_cut ** 2,
            bins=300,
            color="b",
            ec="k",
            lw=0.5,
            label="Data after Abs[\u03B4\u03C3]/\u03C3 outlier cut",
        )
        axs.set_xlabel("\u03C3$^2$/<\u03C3$^2$>")
        axs.set_ylabel("# per bin")
        axs.legend()

        plt.savefig(f"{self.plot_dir}HistogramSigmaSquared.png")

    def plot_omega_time_fit(self):
        """
        Generates and saves a plot of \u03A9 as a function of time and fits the data to perform a linear trend analysis. The plot shows data after the delta-sigma cut.

        Parameters
        ==========

        Returns
        =======

        """
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        fig.suptitle("\u03A9 vs time for each segment", fontsize=20)

        t_obs = self.days_cut[-1]
        scale = np.sqrt(np.var(self.sliding_omega_cut))

        def func(x, a, b):
            return a * (x - t_obs / 2) * t_obs + b

        popt, pcov = curve_fit(
            func, self.days_cut, self.sliding_omega_cut, sigma=self.sliding_sigmas_cut
        )
        c1, c2 = popt[0], popt[1]
        axs.plot(self.days_cut, func(self.days_cut, c1, c2), "r")
        axs.plot(self.days_cut, self.sliding_omega_cut, "b.", markersize=1)
        axs.plot(self.days_cut, 3 * self.sliding_sigmas_cut, "b", linewidth=1.5)
        axs.plot(self.days_cut, -3 * self.sliding_sigmas_cut, "b", linewidth=1.5)
        axs.set_xlabel("Days since start of run")
        axs.set_ylabel("\u03A9$_i$")
        axs.set_xlim(self.days_cut[0], self.days_cut[-1])
        axs.set_ylim(-6 * scale, 6 * scale)
        axs.text(
            0.5,
            scale * 4.25,
            "Linear trend analysis: \u03A9(t) = C$_1$*(t-T$_{obs}$/2)*T$_{obs}$ + C$_2$\nC$_1$ = "
            + str(f"{c1:.3e}")
            + "\nC$_2$ = "
            + str(f"{c2:.3e}"),
            fontsize=15,
            bbox=dict(boxstyle="round", facecolor="white", alpha=1),
        )
        plt.savefig(f"{self.plot_dir}OmegaTimeFit.png")

    def generate_all_plots(self):
        """
        Generates and saves all the statistical analysis plots.

        Parameters
        ==========

        Returns
        =======

        """
        self.plot_running_point_estimate()
        self.plot_running_sigma()
        self.plot_IFFT_point_estimate_integrand()
        self.plot_point_estimate_integrand()
        self.plot_cumulative_point_estimate()
        self.plot_real_point_estimate_integrand()
        self.plot_imaginary_point_estimate_integrand()
        self.plot_sensitivity_integrand()
        self.plot_cumulative_sensitivity_integrand()
        self.plot_omega_sigma_in_time()
        self.plot_hist_sigma_dsc()
        self.plot_scatter_sigma_dsc()
        self.plot_scatter_omega_sigma_dsc()
        self.plot_hist_omega_pre_post_dsc()
        self.plot_KS_test()
        self.plot_hist_scatter_omega_sigma()
        self.plot_hist_sigma_squared()
        self.plot_omega_time_fit()
