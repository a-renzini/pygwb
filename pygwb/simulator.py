import sys

import bilby
import gwpy
import h5py
import numpy as np
from bilby.core.utils import create_frequency_series
from scipy.interpolate import interp1d

from pygwb.baseline import Baseline
from pygwb.constants import H0
from pygwb.util import get_baselines, interpolate_frequencySeries, omegaToPower

if sys.version_info >= (3, 0):
    import configparser
else:
    import ConfigParser as configparser

"""
TODOS
--------

urgent
-------

* everything is initiated via baseline_1; how to change this?
* write all the necessary import checks
    - noise PSD of interferometers is set (add this to get_noise_PSD_list)
    - check frequencies from baseline; remove 0s (add this to set_frequencies)
        in principle we would have to "copy over" ALL the checks implemented in `baseline`;
        this may seem like a good reason to initialise the `Simulator` with baselines directly,
        but then in principle we would have to repeat all those checks between the baselines,
        so I don't think it really matters. If there is a mismatch between `duration` or 
        `sampling_frequency` at the level of individual baselines, this will get flagged when
        initialising a baseline; as each baseline shares a detector with at least 1 other baseline,
        we're fine.
    - other?
* unit tests
* add flag to set NoisePSDs to 0

less urgent
-----------
* discuss with group re- initialising with baselines vs ifos
* discuss Network object
* add save to file functionality


"""


class Simulator(object):
    def __init__(
        self,
        interferometers,
        omegaGW,
        NSegments,
        duration,
        sampling_frequency,
        startTime=0,
        save_to_file=False,
        no_noise=False,
    ):  # (self, noisePSD, omegaGW, orf, sampling_frequency, duration, NSegments):
        """
        Class that simulates an isotropic stochastic background.

        Parameters
        ==========
        interferometers: list of bilby interferometer objects
        omegaGW:
        NSegments:
        duration:
        sampling_frequency:
        startTime:
        save_to_file:
        no_noise:

        Returns
        =======
        """

        self.interferometers = interferometers
        self.Nd = len(self.interferometers)
        # (int(1 + np.sqrt(1 + 8 * len(baselines)))) // 2

        self.sampling_frequency = sampling_frequency
        self.duration = duration

        for ifo in interferometers:
            ifo.sampling_frequency = self.sampling_frequency
            ifo.duration = self.duration

        self.NSegments = NSegments
        self.frequencies = self.get_frequencies()
        self.Nf = len(self.frequencies)
        self.t0 = startTime
        self.NSamplesPerSegment = int(self.sampling_frequency * self.duration)
        self.deltaT = 1 / self.sampling_frequency

        self.noise_PSD_array = self.get_noise_PSD_array()
        if no_noise == True:
            self.noise_PSD_array = np.zeros_like(noise_PSD_array)

        baselines = get_baselines(
            self.interferometers,
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
        )
        self.orf = np.array(
            [baseline.overlap_reduction_function for baseline in baselines]
        )

        self.OmegaGW = interpolate_frequencySeries(omegaGW, self.frequencies)

        self.gen_data = self.generate_data()

        if save_to_file == True:
            self.write()

    @classmethod
    def from_ifo_list(
        cls,
        ifo_list,
        omegaGW,
        NSegments,
        duration,
        sampling_frequency,
        startTime=0,
        save_to_file=False,
        no_noise=False,
    ):

        interferometers = bilby.gw.detector.InterferometerList(ifo_list)

        return cls(
            interferometers,
            omegaGW,
            NSegments,
            duration,
            sampling_frequency,
            startTime=startTime,
            save_to_file=save_to_file,
            no_noise=no_noise,
        )

    def write(self, flag="to_h5"):
        """
        TODO: develop write method; make decisions here
        """

        # def save_data_to_npz(self):
        #    """ """
        #    np.savez("data.npz", data=self.gen_data)

        def save_data_to_h5(self):
            """ """
            timeseries_data = gwpy.timeseries.Timeseries.self.gen_data
            timeseries_data.t0 = my_t0
            Timeseries.write_to_h5(self.gen_data)

        if flag == "to_h5":
            save_data_to_h5()
        else:
            raise ValueError(f"Unknown flag: '{flag}'")

    def get_frequencies(self):
        """ """
        frequencies = create_frequency_series(
            sampling_frequency=self.sampling_frequency, duration=self.duration
        )

        return frequencies

    def get_noise_PSD_array(self):
        """ """
        noisePSDs = []
        for ifo in self.interferometers:
            psd = ifo.power_spectral_density_array
            psd[np.isinf(psd)] = 1
            # ^^^ this makes sure that there are no infinities in the psd
            noisePSDs.append(psd)

        return np.array(noisePSDs)

    def timeseries_data(self):
        return self.gen_data

    def generate_data(self):
        """
        Function that simulates an isotropic stochastic background given the
        input parameters. The data is simulated and spliced together to prevent
        periodicity artifacts related to IFFTs.

        Parameters
        ==========

        Returns
        =======
        data: array_like
            An array of size Nd (number of detectors) with gwpy TimeSeries with the
            data containing the simulated isotropic stochastic background.
        """
        y = self.simulate_data()
        dataTemp = self.splice_segments(y)

        data = np.zeros(self.Nd, dtype=gwpy.timeseries.TimeSeries)
        for ii in range(self.Nd):
            data[ii] = gwpy.timeseries.TimeSeries(
                dataTemp[ii], t0=self.t0, dt=self.deltaT
            )

        return data

    def orfToArray(self):
        """
        Function that converts the list of overlap reduction functions into an array
        to facilitate the correct implementation when computing the covariance matrix.

        Parameters
        ==========

        Returns
        =======
        orf_array: array_like
            Array of shape Nd x Nd containing the orfs, where Nd is the number of
            detectors. The convention used for consistency with the remainder of the
            simulation is as follows. Orfs are only present in the off-diagonal slots
            in the array. Only the part below the diagonal is filled, after which this
            is copied to the upper part by transposing and summing. The array is filled
            by starting from the first free slot below the diagonal, from left to right,
            until the diagonal is reached, after which the line below in the array is
            filled analogously and so on.
        """
        index = 0
        orf_array = np.zeros(
            (self.Nd, self.Nd),
            dtype=gwpy.frequencyseries.frequencyseries.FrequencySeries,
        )
        for ii in range(self.Nd):
            for jj in range(ii):
                orf_array[ii, jj] = self.orf[index]
                index += 1
        orf_array = orf_array + orf_array.transpose()
        return orf_array

    def covariance_matrix(self):
        """
        Function to compute the covariance matrix corresponding to a stochastic
        background in the various detectors.

        Parameters
        ==========

        Returns
        =======
        C: array_like
            Covariance matrix corresponding to a stochastic background in the
            various detectors. Dimensions are Nd x Nd x Nf, where Nd is the
            number of detectors and Nf the number of frequencies.
        """
        GWBPower = omegaToPower(self.OmegaGW, self.frequencies)
        orf_array = self.orfToArray()

        C = np.zeros((self.Nd, self.Nd, self.Nf))

        for ii in range(self.Nd):
            for jj in range(self.Nd):
                if ii == jj:
                    C[ii, jj, :] = self.noise_PSD_array[ii] + GWBPower.value[:]
                else:
                    C[ii, jj, :] = orf_array[ii, jj] * GWBPower.value[:]

        C = self.NSamplesPerSegment / (self.deltaT * 4) * C
        return C

    def compute_eigval_eigvec(self, C):
        """
        Function to compute the eigenvalues and eigenvectors of the covariance
        matrix corresponding to a stochastic background in the various detectors.

        Parameters
        ==========
        C: array_like
            Covariance matrix corresponding to a stochastic background in the
            various detectors. Dimensions are Nd x Nd x Nf, where Nd is the
            number of detectors and Nf the number of frequencies.

        Returns
        =======
        eigval: array_like
            Array of diagonal matrices containing the eigenvalues of the
            covariance matrix C.
        eigvec: array_like
            Array of matrices containing the eigenvectors of the covariance
            matrix C.
        """
        eigval, eigvec = np.linalg.eig(C.transpose((2, 0, 1)))
        eigval = np.array([np.diag(x) for x in eigval])
        return eigval, eigvec

    def generate_freq_domain_data(self):
        """
        Function that generates the uncorrelated frequency domain data with
        random phases for the stochastic background.

        Parameters
        ==========

        Returns
        =======
        z: array_like
            Array of size Nf x Nd containing uncorrelated frequency domain data.
        """
        z = np.zeros((self.Nf, self.Nd), dtype="complex_")
        re = np.random.randn(self.Nf, self.Nd)
        im = np.random.randn(self.Nf, self.Nd)
        z = re + im * 1j
        return z

    def transform_to_correlated_data(self, z, C):
        """
        Function that transforms the uncorrelated stochastic background
        simulated data, to correlated data.

        Parameters
        ==========
        z: array_like
            Array containing the uncorrelated data with random phase.
        C: array_like
            Array of size Nd x Nd x Nf representing the covariance matrices
            between detectors for a desired stochastic background, where Nd
            is the number of detectors and Nf is the number of frequencies.

        Returns
        =======
        x: array_like
            Array of size Nf x Nd, containing the correlated stochastic
            background data.
        """
        eigval, eigvec = self.compute_eigval_eigvec(C)

        A = np.einsum("...ij,jk...", np.sqrt(eigval), eigvec.transpose())
        x = np.einsum("...j,...jk", z, A)
        return x

    def simulate_data(self):
        """
        Function that simulates the data corresponding to an isotropic stochastic
        background.

        Parameters
        ==========

        Returns
        =======
        y: array_like
            Array of size Nd x 2*(NSegments+1) x NSamplesPerSegment containing the
            various segments with the simulated data.
        """
        C = self.covariance_matrix()

        y = np.zeros(
            (self.Nd, 2 * self.NSegments + 1, self.NSamplesPerSegment), dtype=np.ndarray
        )

        for kk in range(2 * self.NSegments + 1):
            z = self.generate_freq_domain_data()

            xtemp = self.transform_to_correlated_data(z, C)

            for ii in range(self.Nd):
                if self.NSamplesPerSegment % 2 == 0:
                    xtilde = np.concatenate(
                        (
                            np.array([0]),
                            xtemp[:, ii],
                            np.array([0]),
                            np.flipud(np.conjugate(xtemp[:, ii])),
                        )
                    )
                else:
                    xtilde = np.concatenate(
                        (
                            np.array([0]),
                            xtemp[:, ii],
                            np.flipud(np.conjugate(xtemp[:, ii])),
                        )
                    )
                y[ii, kk, :] = np.real(np.fft.ifft(xtilde))[: self.NSamplesPerSegment]
        return y

    def splice_segments(self, segments):
        """
        This function splices together the various segments to prevent
        artifacts related to the periodicity that can arise from inverse
        Fourier transforms.

        Parameters
        ==========
        segments: array_like
            Array of size Nd x (2*NSegments+1) x NSamplesPerSegment containing the
            various segments with the simulated data that need to be spliced
            together, where Nd is the number of detectors.

        Returns
        =======
        data: array_like
            Array of size Nd x (NSegments*NSamplesPerSegment) containing the simulated
            data corresponding to an isotropic stochastic background for each of the
            detectors, where Nd is the number of detectors.
        """
        w = np.zeros(self.NSamplesPerSegment)

        for ii in range(self.NSamplesPerSegment):
            w[ii] = np.sin(np.pi * ii / self.NSamplesPerSegment)

        data = np.zeros(
            (self.Nd, self.NSamplesPerSegment * self.NSegments), dtype=np.ndarray
        )

        for ii in range(self.Nd):
            for jj in range(self.NSegments):
                y0 = w * segments[ii][2 * jj][:]
                y1 = w * segments[ii][2 * jj + 1][:]
                y2 = w * segments[ii][2 * jj + 2][:]

                z0 = np.concatenate(
                    (
                        y0[int(self.NSamplesPerSegment / 2) : self.NSamplesPerSegment],
                        np.zeros(int(self.NSamplesPerSegment / 2)),
                    )
                )
                z1 = y1[:]
                z2 = np.concatenate(
                    (
                        np.zeros(int(self.NSamplesPerSegment / 2)),
                        y2[0 : int(self.NSamplesPerSegment / 2)],
                    )
                )

                data[
                    ii,
                    jj * self.NSamplesPerSegment : (jj + 1) * self.NSamplesPerSegment,
                ] = (
                    z0 + z1 + z2
                )

        return data
