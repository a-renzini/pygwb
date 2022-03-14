import pickle

import h5py
import numpy as np
from loguru import logger
from tqdm import tqdm

from .util import calc_bias, window_factors


class SingleStochasticJob(object):
    """Base class for a stochastic job. In this case, the assumption
    is that we have some number of segments we  might want to combine together.

    Attributes:
    -----------
    Y : `numpy.ndarray`
        point estimate array. dimensions: (Nsegs, Nfreqs) or (Nsegs, Ndirections)
    sigma : `numpy.ndarray`
        standard deviation on point estimate. dimensions:
        (segs, freqs/directions)
    times : `numpy.ndarray`
        list of segment start times
    segdur : `float`
        segment duration
    sample_rate : `int`, required
        sample rate
    fref : `float`, optional
        reference frequency if integrated statistic
    alpha : `float`, optional
        powerlaw spectral index if integrated statistic
    frequencies : `numpy.ndarray`, optional
        list of frequencies if narrowband radiometer
    directions : `numpy.ndarray`, optional
        directions (if broadband radiometer)
    """

    def __init__(
        self,
        Y,
        sigma,
        times,
        segdur,
        sample_rate,
        fref=25,
        alpha=0,
        frequencies=None,
        directions=None,
    ):
        """initialize"""
        self.Y = np.array(Y)
        self.sigma = np.array(sigma)
        self.times = np.array(times)
        self.segdur = segdur
        self.fref = fref
        self.alpha = alpha
        self.sample_rate = sample_rate
        self.Nsamples = int(self.segdur * self.sample_rate)
        self.Nsegments = np.size(times)
        if frequencies is not None and directions is not None:
            raise ValueError(
                "Can't specify frequencies and direction at the same time."
            )
        if directions is not None:
            self.directions = np.array(directions)
            self.ndirections = self.directions.squeeze().size
            self.frequencies = None
            self.dim2_type = "directions"
        elif frequencies is not None:
            self.frequencies = np.array(frequencies)
            self.df = np.min(self.frequencies[1:] - self.frequencies[:-1])
            self.nfrequencies = self.frequencies.size
            self.directions = None
            self.dim2_type = "frequencies"

    @property
    def dim2(self):
        # returns second dimension
        return self.Y.shape[-1]

    def consistency_check(self):
        """
        Throws an error if the
        number of directions or freuqencies (dim2)
        doesn't match with the second dimension of Y
        and sigma.

        Parameters:
        -----------

        Returns:
        --------
        """
        # check shapes
        # if frequencies is none,
        # then we assume it's a map and use directions
        ntimes = np.size(self.times)
        if self.frequencies is not None:
            dim2 = self.nfrequencies
        else:
            # directions are (Ndirectionss, 2)
            dim2 = self.ndirections
        if self.Y.shape != (ntimes, dim2):
            raise ValueError("Shape of Y is incorrect (may be transposed?)")
        if self.sigma.shape != (ntimes, dim2):
            raise ValueError("Shape of sigma is incorrect (may be transposed?)")
        if self.sample_rate is None:
            raise ValueError("Must specify a sample rate")

    def apply_bad_gps_times(self, times_to_remove):
        """
        Applies bad gps times to current object in place.
        Includes a new attribute `ntimes_removed` to track
        how many times have been removed by cut.

        Parameters:
        -----------
        times_to_remove : `numpy.ndarray`
            times to remove from supercut

        Returns:
        --------

        """
        times_to_remove = np.array(times_to_remove)
        good_times = []
        for ii, our_time in enumerate(self.times):
            good_times.append(~np.any(times_to_remove == our_time))
        self.Y = self.Y[np.where(good_times)[0]]
        self.sigma = self.sigma[np.where(good_times)[0]]
        self.times = self.times[np.where(good_times)[0]]
        try:
            self.ntimes_removed += np.size(good_times) - np.sum(good_times)
        except AttributeError:
            self.ntimes_removed = np.size(good_times) - np.sum(good_times)
        self.consistency_check()

    def _combine_non_time_dimension_even_odd(self):
        """
        taken from
        stochastic/trunk/PostProcessing/combineResults.m
        written by Joe (quite a nice function)
        """
        _, w1w2squaredbar, _, w1w2squaredovlbar = window_factors(self.Nsamples)
        k = w1w2squaredovlbar / w1w2squaredbar
        size = np.size(self.times)

        # even/odd indices
        evens = np.arange(0, size, 2)
        odds = np.arange(1, size, 2)

        # evens
        X_even = np.nansum(self.Y[evens] / self.sigma[evens] ** 2, axis=0)
        GAMMA_even = np.nansum(self.sigma[evens] ** -2, axis=0)

        # odds
        if size == 1:
            return X_even / GAMMA_even, GAMMA_even**-0.5
        else:
            X_odd = np.nansum(self.Y[odds] / self.sigma[odds] ** 2, axis=0)
            GAMMA_odd = np.nansum(self.sigma[odds] ** -2, axis=0)

        # get rid of zero values
        GAMMA_odd[GAMMA_odd == 0] = np.nan
        GAMMA_even[GAMMA_even == 0] = np.nan
        sigma2_oo = 1 / np.nansum(GAMMA_odd)
        sigma2_ee = 1 / np.nansum(GAMMA_even)
        sigma2_1 = 1 / np.nansum(self.sigma[0, :] ** -2)
        sigma2_N = 1 / np.nansum(self.sigma[-1, :] ** -2)
        sigma2IJ = (
            1 / sigma2_oo + 1 / sigma2_ee - (1 / 2) * (1 / sigma2_1 + 1 / sigma2_N)
        )

        Y_f_new = (
            X_odd * (1 - (k / 2) * sigma2_oo * sigma2IJ)
            + X_even * (1 - (k / 2) * sigma2_ee * sigma2IJ)
        ) / (
            GAMMA_even
            + GAMMA_odd
            - k
            * (
                GAMMA_even
                + GAMMA_odd
                - (1 / 2) * (1 / self.sigma[0, :] ** 2 + 1 / self.sigma[-1, :] ** 2)
            )
        )

        inv_var_f_new = (
            GAMMA_odd
            + GAMMA_even
            - k
            * (
                GAMMA_odd
                + GAMMA_even
                - (1 / 2) * (1 / self.sigma[0, :] ** 2 + 1 / self.sigma[-1, :] ** 2)
            )
        ) / (1 - (k**2 / 4) * sigma2_oo * sigma2_ee * sigma2IJ**2)

        bias = calc_bias(self.segdur, self.df, 1 / self.sample_rate, N_avg_segs=2)
        logger.debug(f"Bias factor: {bias}")
        var_f_new = (1 / inv_var_f_new) * bias**2
        # I don't think notching should happen here.
        # I thought it happened somewhere else.
        # var_f_new[notch_freq] = np.inf

        return Y_f_new, np.sqrt(var_f_new)

        # return combined_Y, errorBar * bias

    def __repr__(self):
        """
        simple representation.
        """
        title = f"Single stochastic job from: {self.times[0]}-{self.times[-1]}\n"
        times = "\tNumber of segments: %d\n" % self.times.size
        if self.dim2_type == "frequencies":
            type_line = "\tFrequency Range: %4.6f - %4.6f Hz\n" % (
                self.frequencies[0],
                self.frequencies[-1],
            )
        else:
            type_line = "\tNumber of directions: %d" % (self.ndirections)
        return title + times + type_line


class IsotropicJob(SingleStochasticJob):
    """single stochastic job results"""

    def __init__(self, *args, **kwargs):
        super(IsotropicJob, self).__init__(*args, **kwargs)
        # get final averaged spectra
        (
            self.combined_Y_spectrum,
            self.combined_sigma_spectrum,
        ) = self._combine_non_time_dimension_even_odd()

    @classmethod
    def from_matlab_file(cls, matfile):
        """
        Load mat file from old matlab code, create a single isotropic job
        and return an `IsotropicJob` class

        NOTE: mat file behaves differently if there's only one segment.
            need to be careful when opening it here.

        Parameters:
        -----------
        matfile : `str`
            matfile containing results from stochastic
            isotropic job

        Returns:
        --------
        job : `IsotropicJob`
            a single `IsotropicJob` instance
        """
        f = h5py.File(matfile, "r")
        names = []
        for name in f:
            names.append(name)
        # if all that's there are
        # parames, then return None
        if names == ["params"]:
            return None
        # data
        segstarts = f["segmentStartTime"][0]
        sensInt_ref = f["sensInt/data"]
        ccspec_ref = f["ccSpec/data"]
        # parameters
        sample_rate = f["params"]["resampleRate1"][()].squeeze()
        dim1 = sensInt_ref.size
        segdur = f["params"]["segmentDuration"][()].squeeze()
        fhigh = f["params"]["fhigh"][()].squeeze()
        flow = f["params"]["flow"][()].squeeze()
        df = f["params"]["deltaF"][()].squeeze()
        freqs = np.arange(flow, fhigh + df, df)
        dim2 = freqs.squeeze().size

        # unpack data...this took quite a while to figure out.
        if dim1 == freqs.size:
            sensInt_segs = sensInt_ref[()].squeeze()
            sensInt_segs = np.reshape(sensInt_segs, (1, sensInt_segs.size))
            ccspec_segs = ccspec_ref["real"] + ccspec_ref["imag"] * 1j
            ccspec_segs = np.reshape(ccspec_segs, (1, ccspec_segs.size))
        else:
            sensInt_segs = np.zeros((dim1, dim2), dtype=complex)
            ccspec_segs = np.zeros((dim1, dim2), dtype=complex)
            for ii in range(dim1):
                sensInt_segs[ii, :] = np.real(f[sensInt_ref[0][ii]][0])
                ccspec_segs[ii, :] = (
                    f[ccspec_ref[0][ii]][0]["real"]
                    + 1j * f[ccspec_ref[0][ii]][0]["imag"]
                )
        # convert to Y and sigma
        Y = np.zeros(sensInt_segs.shape, dtype=complex)
        sig = np.zeros(sensInt_segs.shape)
        sensInt_segs[sensInt_segs == 0] = np.nan
        for ii in range(sensInt_segs.shape[0]):
            Y[ii, :] = (2 * df * ccspec_segs[ii, :]) * (
                sensInt_segs[ii, :] / np.nansum(sensInt_segs[ii, :])
            ) ** -1
            sig[ii, :] = 1 / np.real(sensInt_segs[ii, :] * df)

        # divide by segment duration
        final_Y = Y / segdur
        final_sigma = sig**0.5 / segdur
        # create and return IsotropicJob class
        return cls(
            final_Y, final_sigma, segstarts, segdur, sample_rate, frequencies=freqs
        )

    def calculate_broadband_statistics(self, alpha):
        """
        The combined, time-averaged, broadband statistics.
        """
        # calculate weights for combining
        weights = (self.frequencies / self.fref) ** alpha
        # combine
        X = (weights * self.combined_Y_spectrum) * (
            weights * self.combined_sigma_spectrum
        ) ** -2
        GAMMA = (weights * self.combined_sigma_spectrum) ** -2
        sigma_cumulative_total = np.nansum(GAMMA) ** -0.5
        y_cumulative_total = np.nansum(X) / np.nansum(GAMMA)
        return (y_cumulative_total, sigma_cumulative_total)

    def calculate_segment_by_segment_broadband_statistics(self, alpha):
        """
        The combined broadband segment-by-segment statistics

        Returns:
        --------
        y_cum_ts : `numpy.ndarray`
            cumulative broadband point estimate for each
            time segment
        sig_cum_ts : `numpy.ndarray`
            uncertainty on `y_cumulative_ts`
        """
        # calculate weights
        # for alpha examples
        weights = (self.frequencies / self.fref) ** alpha
        WEIGHTS, _ = np.meshgrid(weights, self.times)
        # X = Y/sigma^2
        # GAMMA = 1/sigma^2
        X = (self.Y * WEIGHTS) * (self.sigma * WEIGHTS) ** -2
        GAMMA = (self.sigma * WEIGHTS) ** -2
        sigma_cumulative_ts = np.nansum(GAMMA, axis=1) ** -0.5
        y_cumulative_ts = np.nansum(X, axis=1) / np.nansum(GAMMA, axis=1)
        return (y_cumulative_ts, sigma_cumulative_ts)


class StochasticJobList(object):
    """list of stochastic jobs
    The object is just a container. It doesn't have any
    point estimate or sigma associated with it. It keeps track

    Attributes:
    -----------
    output_files_list : `list`
        list of strings pointing to files to load
    """

    search_types = ["isotropic"]
    file_types = ["old_matlab"]

    def __init__(
        self,
        output_files_list,
        checkpoint_filename="checkpoint.pkl",
        bad_gps_times_file=None,
        search_type="isotropic",
        file_type="old_matlab",
    ):
        super(StochasticJobList, self).__init__()

        # check valid search type
        if search_type not in StochasticJobList.search_types:
            raise ValueError(
                f"Invalid search type. Only {StochasticJobList.search_types} are implemented"
            )
        else:
            self.search_type = search_type

        # check valid file type
        if file_type not in StochasticJobList.file_types:
            raise ValueError(
                f"Invalid file type. Only {StochasticJobList.file_types} are implemented"
            )
        else:
            self.file_type = file_type

        self.output_files_list = output_files_list
        self.checkpoint_filename = checkpoint_filename
        self.bad_gps_times_file = bad_gps_times_file

    def load_job_file(self, file):
        if self.search_type == "isotropic":
            if self.file_type == "old_matlab":
                return IsotropicJob.from_matlab_file(file)

    def combine_jobs(self, checkpoint_number=10, load_checkpoint=False):
        """
        load data for each job in list and combine results from those jobs.

        Parameters:
        -----------
        segdur : `float`
            segment duration for stochastic.m data we load
        checkpoint_number : `int`, optional, default=10
            number of jobs to combine before checkpointing
        load_checkpoint : `bool`, optional, default=False
            load checkpoint file if available

        Returns:
        --------
        Y : `numpy.ndarray`
            point estimate
        sigma : `numpy.ndarray`
            standard deviation on point estimate
        dim2 : `numpy.ndarray`
            second dimension for Y and sigma
        dim2type : `str`
            type of second dimension
        """
        # for 2/3 compatibility...
        try:
            FileNotFoundError
        except NameError:
            FileNotFoundError = IOError
        import sys

        self.jobs_combined = np.array([])
        self.jobs_failed = np.array([])

        X = None  # Y/sigma^2
        GAMMA = None  # sigma^-2
        # load in supercut times
        if self.bad_gps_times_file is not None:
            bad_times = np.loadtxt(self.bad_gps_times_file)
        else:
            bad_times = []

        if load_checkpoint:
            try:
                tmp = pickle.load(open(self.checkpoint_filename, "rb"))
                X = tmp["X"]
                GAMMA = tmp["GAMMA"]
                self.jobs_combined = tmp["jobs_combined"]
                dim2 = tmp["dim2"]
                dim2type = tmp["dim2type"]
            except FileNotFoundError:
                print("Checkpoint file does not exist. Running from the start")

        # loop over files to load
        pbar = tqdm(self.output_files_list)
        job = 0
        for myfile in pbar:
            pbar.set_description(
                f"Job {job+1} of {len(self.output_files_list)}, {np.size(self.jobs_failed)} jobs have failed"
            )
            if np.in1d(job, self.jobs_combined):
                # jobs have already been included
                continue
            tmpjob = self.load_job_file(myfile)

            # main failure points are empty job or all NaN's
            # In a few cases load_from_matfile will return none
            # if it can't figure out what to do. That way we can
            # still combine over failed jobs if we want.
            if np.sum(1 * np.isnan(tmpjob.Y)) == np.size(tmpjob.Y):
                self.jobs_failed = np.append(self.jobs_failed, job)
                continue
            if tmpjob is None:
                self.jobs_failed = np.append(self.jobs_failed, job)
                continue

            tmpjob.apply_bad_gps_times(bad_times)
            if tmpjob.Y.shape[0] == 0:
                # nothing is in the matfile
                self.jobs_failed = np.append(self.jobs_failed, job)
                continue
            if X is None:
                # initialize
                Ytmp, sigtmp = (
                    tmpjob.combined_Y_spectrum,
                    tmpjob.combined_sigma_spectrum,
                )

                sigtmp[np.isnan(sigtmp) * np.isnan(Ytmp)] = np.inf
                Ytmp[np.isnan(Ytmp)] = 0
                X = Ytmp / sigtmp**2
                GAMMA = sigtmp**-2
                X[np.where(np.isnan(X))[0]] = 0
                if tmpjob.frequencies is not None:
                    dim2 = tmpjob.frequencies
                    dim2type = "freqs"
                else:
                    dim2 = tmpjob.directions
                    dim2type = "directions"
            else:
                # combine already loaded results
                Ytmp, sigtmp = (
                    tmpjob.combined_Y_spectrum,
                    tmpjob.combined_sigma_spectrum,
                )
                sigtmp[np.isnan(sigtmp)] = np.inf
                Ytmp[np.isnan(Ytmp)] = 0
                X = X + Ytmp * sigtmp**-2
                X[np.where(np.isnan(X))[0]] = 0
                GAMMA += sigtmp**-2
            self.jobs_combined = np.append(self.jobs_combined, job)
            if (job % checkpoint_number) == 0:
                checkpoint_material = {
                    "X": X,
                    "GAMMA": GAMMA,
                    "jobs_combined": self.jobs_combined,
                    "dim2": dim2,
                    "dim2type": dim2type,
                }
                with open(self.checkpoint_filename, "wb") as f:
                    pickle.dump(checkpoint_material, f)
            job += 1
        GAMMA[GAMMA == 0] = np.nan
        Y = X / GAMMA
        sigma = GAMMA**-0.5

        return Y, sigma, dim2, dim2type


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
