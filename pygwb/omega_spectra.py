import pickle
import warnings

import numpy as np
from astropy.io import registry as io_registry
from gwpy.frequencyseries import FrequencySeries
from gwpy.spectrogram import Spectrogram
from gwpy.types.io.hdf5 import register_hdf5_array_io

from pygwb.constants import h0


class OmegaSpectrogram(Spectrogram):

    """Subclass of gwpy's Spectrogram class."""

    _metadata_slots = Spectrogram._metadata_slots + ("alpha", "fref", "h0")
    # _print_slots = FrequencySeries._print_slots + ["alpha", "fref", "h0"]

    def __new__(cls, data, **kwargs):
        kwargs.pop("alpha", None)
        kwargs.pop("fref", None)
        kwargs.pop("h0", None)
        return super(OmegaSpectrogram, cls).__new__(cls, data, **kwargs)

    def __init__(self, data, alpha=None, fref=None, h0=h0, **kwargs):
        if not isinstance(alpha, (float, int, np.number)):
            raise ValueError("Spectral index alpha must be a valid number.")
        if not isinstance(fref, (float, int, np.number)):
            raise ValueError("Reference frequency fref must be a valid number.")
        self._alpha = alpha
        self._fref = fref
        self._h0 = h0

    @property
    def alpha(self):
        """Spectral index alpha."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        raise AttributeError(
            "alpha is protected! To change alpha, use the reweight method."
        )

    @property
    def fref(self):
        """Reference frequency."""
        return self._fref

    @fref.setter
    def fref(self, fref):
        raise AttributeError(
            "fref is protected! To change fref, use the reweight method."
        )

    @property
    def h0(self):
        """Hubble parameter h0. Default is pygwb.constants.h0 = 0.693199."""
        return self._h0

    @h0.setter
    def h0(self, h0):
        raise AttributeError("h0 is protected! To change h0, use the reset_h0 method.")

    @classmethod
    def read(cls, source, *args, **kwargs):
        """Read data into a Spectrogram. Same usage as read method of gwpy.Spectrogram.

        Parameters
        ==========
        source: str
            Source file path.
        """
        return io_registry.read(cls, source, *args, **kwargs)

    def write(self, target, *args, **kwargs):
        """Write this Spectrogram to a file. Same usage as write method of gwpy.Spectrogram.

        Parameters
        ==========
        target: str
            Target file path.
        """
        return io_registry.write(self, target, *args, **kwargs)

    @classmethod
    def load_from_pickle(cls, filename):
        """
        Load spectrogram object from pickle file.

        Parameters
        ==========
        filename: str
            Filename (inclusive of path) to load the pickled spectrogram from.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def save_to_pickle(self, filename):
        """
        Save spectrogram object to pickle file.

        Parameters
        ==========
        filename: str
            Filename (inclusive of path) to save the pickled spectrogram to.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def reweight(self, *, new_alpha=None, new_fref=None):
        """
        Reweight the spectrogram by a new spectral index alpha, and/or refer to a new reference freuency.

        Parameters:
        ==========

        new_alpha: float
            New spectral index.
        new_fref: float
            New reference frequency.
        """
        if new_alpha is None:
            new_alpha = self.alpha
        if new_fref is None:
            new_fref = self.fref
        for spectrum in self.value:
            new_spectrum = reweight_spectral_object(
                spectrum,
                self.frequencies,
                new_alpha,
                new_fref,
                old_alpha=self.alpha,
                old_fref=self.fref,
            )
            spectrum[:] = new_spectrum
        self._alpha = new_alpha
        self._fref = new_fref

    def reset_h0(self, new_h0):
        """
        Reset the hubble parameter h0. Expected values range between 0.5 and 1.

        Parameters:
        ==========

        new_h0: float
            New h0 to set the spectrum at.
        """
        if (new_h0 < 0.5) or (new_h0 > 1.0):
            warnings.warn(
                f"h0 should be between 0.5 and 1. The selected value of {new_h0} does not fall within this range."
            )
        new_spectrum = self.value * (self.h0 / new_h0) ** 2
        self.value[:] = new_spectrum
        self._h0 = new_h0


class OmegaSpectrum(FrequencySeries):

    """Subclass of gwpy's FrequencySeries class."""

    _metadata_slots = FrequencySeries._metadata_slots + ("alpha", "fref", "h0")
    _print_slots = FrequencySeries._print_slots + ["alpha", "fref", "h0"]

    def __new__(cls, data, **kwargs):
        kwargs.pop("alpha", None)
        kwargs.pop("fref", None)
        kwargs.pop("h0", None)
        return super(OmegaSpectrum, cls).__new__(cls, data, **kwargs)

    def __init__(self, data, alpha=None, fref=None, h0=h0, **kwargs):
        if not isinstance(alpha, (float, int, np.number)):
            raise ValueError("Spectral index alpha must be a valid number.")
        if not isinstance(fref, (float, int, np.number)):
            raise ValueError("Reference frequency fref must be a valid number.")
        self._alpha = alpha
        self._fref = fref
        self._h0 = h0

    @property
    def alpha(self):
        """Spectral index alpha."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        raise AttributeError(
            "alpha is protected! To change alpha, use the reweight method."
        )

    @property
    def fref(self):
        """Reference frequency."""
        return self._fref

    @fref.setter
    def fref(self, fref):
        raise AttributeError(
            "fref is protected! To change fref, use the reweight method."
        )

    @property
    def h0(self):
        """Hubble parameter h0. Default is pygwb.constants.h0 = 0.693199."""
        return self._h0

    @h0.setter
    def h0(self, h0):
        raise AttributeError("h0 is protected! To change h0, use the reset_h0 method.")

    @classmethod
    def read(cls, source, *args, **kwargs):
        """Read data into a Spectrum. Same usage as read method of gwpy.FrequencySeries.

        Parameters
        ==========
        source: str
            Source file path.
        """
        return io_registry.read(cls, source, *args, **kwargs)

    def write(self, target, *args, **kwargs):
        """Write this Spectrum to a file. Same usage as write method of gwpy.FrequencySeries.


        Parameters
        ==========
        target: str
            Target file path.
        """
        return io_registry.write(self, target, *args, **kwargs)

    @classmethod
    def load_from_pickle(cls, filename):
        """
        Load spectrum object from pickle file.

        Parameters
        ==========
        filename: str
            Filename (inclusive of path) to load the pickled spectrum from.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def save_to_pickle(self, filename):
        """
        Save spectrum object to pickle file.

        Parameters
        ==========
        filename: str
            Filename (inclusive of path) to save the pickled spectrum to.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def reweight(self, *, new_alpha=None, new_fref=None):
        """
        Reweight the spectrum by a new spectral index alpha, and/or refer to a new reference freuency.

        Parameters:
        ==========

        new_alpha: float
            New spectral index.
        new_fref: float
            New reference frequency.
        """
        if new_alpha is None:
            new_alpha = self.alpha
        if new_fref is None:
            new_fref = self.fref
        new_spectrum = reweight_spectral_object(
            self.value,
            self.frequencies,
            new_alpha,
            new_fref,
            old_alpha=self.alpha,
            old_fref=self.fref,
        )
        self.value[:] = new_spectrum
        self._alpha = new_alpha
        self._fref = new_fref

    def reset_h0(self, new_h0):
        """
        Reset the hubble parameter h0. Expected values range between 0.5 and 1.

        Parameters:
        ==========

        new_h0: float
            New h0 to set the spectrum at.
        """
        if (new_h0 < 0.5) or (new_h0 > 1.0):
            raise ValueError("h0 must be between 0.5 and 1.")
        new_spectrum = self.value * (self.h0 / new_h0) ** 2
        self.value[:] = new_spectrum
        self._h0 = new_h0


def reweight_spectral_object(
    spec, freqs, new_alpha, new_fref, old_alpha=0.0, old_fref=25.0
):
    """
    Reweight a spectrum or spectrogram object.
    Input spectrogram assumes a shape of: N_frequencies x N_times
    This is meant to be a helper function used to change the spectral index of the stochastic results.

    Parameters
    ----------
        spec: array-like
            Spectrum or spectrogram (with shape N_frequencies x Ntimes)
        freqs: array-like
            Frequencies associated with `spec`.
        new_alpha: float
            New spectral index
        new_fref: float
            New reference frequency
        old_alpha: float, optional
            Spectral index of input `spec` array (i.e. weighting of `spec`). Defaults to zero (assumes unweighted)
        old_fref:
            Reference frequency of current `spec` weighting (assumes 1 Hz)

    Returns
    -------
        new_spec: array-like
            Reweighted spectrum or spectrogram array.
    """
    weights_old = (freqs / old_fref) ** old_alpha
    weights_new = (freqs / new_fref) ** new_alpha
    return (spec.T * (weights_old / weights_new)).T


register_hdf5_array_io(OmegaSpectrogram)
register_hdf5_array_io(OmegaSpectrum)
