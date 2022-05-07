import numpy as np
from gwpy.frequencyseries import FrequencySeries
from gwpy.spectrogram import Spectrogram

from pygwb.spectral import reweight_spectral_object


class OmegaSpectrogram(Spectrogram):

    """Subclass of gwpy's Spectrogram class"""

    # (data, unit=None, t0=None, dt=None, f0=None, df=None, times=None, frequencies=None, name=None, channel=None, **kwargs)
    def __new__(cls, data, **kwargs):
        kwargs.pop("alpha", None)
        kwargs.pop("fref", None)
        kwargs.pop("h0", None)
        return super(OmegaSpectrogram, cls).__new__(cls, data, **kwargs)

    def __init__(self, data, alpha=None, fref=None, h0=1.0, **kwargs):
        self.alpha = alpha
        self.fref = fref
        self.h0 = h0

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, (float, int)):
            raise ValueError("Spectral index alpha must be a valid number.")
        self._alpha = alpha
        self._alpha_set = True

    @property
    def fref(self):
        return self._fref

    @fref.setter
    def fref(self, fref):
        if not isinstance(fref, (float, int)):
            raise ValueError("Reference frequency fref must be a valid number.")
        self._fref = fref

    @property
    def h0(self):
        if self._h0_set:
            return self._h0
        else:
            raise ValueError("Hubble parameter h0 has not yet been set")

    @h0.setter
    def h0(self, h0):
        self._h0 = h0
        self._h0_set = True

    def reweight(self, *, new_alpha=None, new_fref=None):
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
        self.alpha = new_alpha
        self.fref = new_fref

    def reset_h0(self, new_h0):
        new_spectrum = self.value * (self.h0 / new_h0) ** 2
        self.value[:] = new_spectrum


class OmegaSpectrum(FrequencySeries):

    """Subclass of gwpy's FrequencySeries class"""

    def __new__(cls, data, **kwargs):
        kwargs.pop("alpha", None)
        kwargs.pop("fref", None)
        kwargs.pop("h0", None)
        return super(OmegaSpectrum, cls).__new__(cls, data, **kwargs)

    def __init__(self, data, alpha=None, fref=None, h0=1.0, **kwargs):
        self.alpha = alpha
        self.fref = fref
        self.h0 = h0

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        if not isinstance(alpha, (float, int)):
            raise ValueError("Spectral index alpha must be a valid number.")
        self._alpha = alpha
        self._alpha_set = True

    @property
    def fref(self):
        return self._fref

    @fref.setter
    def fref(self, fref):
        if not isinstance(fref, (float, int)):
            raise ValueError("Reference frequency fref must be a valid number.")
        self._fref = fref

    @property
    def h0(self):
        if self._h0_set:
            return self._h0
        else:
            raise ValueError("Hubble parameter h0 has not yet been set")

    @h0.setter
    def h0(self, h0):
        self._h0 = h0
        self._h0_set = True

    def reweight(self, *, new_alpha=None, new_fref=None):
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
        self.alpha = new_alpha
        self.fref = new_fref

    def reset_h0(self, new_h0):
        new_spectrum = self.value * (self.h0 / new_h0) ** 2
        self.value[:] = new_spectrum
