"""
The ``coherence`` module only contains one method, which is used to calculate the coherence spectrum given two power spectrograms and a cross-spectrogram.

Examples
--------
    
Given two power spectrograms and a cross-spectrogram, the coherence can be computed by
    
>>> from pygwb.coherence import calculate_coherence
>>> calculate_coherence(psd_1, psd_2, csd)
"""

import numpy as np


def calculate_coherence(psd_1, psd_2, csd):
    '''
    Calculate a coherence spectrum given two power spectrograms and a cross-spectrogram.

    Parameters
    =======
    psd_1_spectrum: ``np.array``
        PSD spectrum of first detector.
    psd_2_spectrum: ``np.array``
        PSD spectrum of second detector.
    csd_spectrum: ``np.array``
        Spectrum of the cross-spectral density of the two detectors.

    Returns
    =======
    coherence: ``np.array``
        Coherence spectrum
    '''
    #I don't think we need these terms anymore:
    #fftlength = int(1.0/psd_1_spectrogram.df.value)
    #norm1 = int(np.floor(duration/(fftlength*overlap_factor))-1)
    return  np.real(csd*np.conj(csd)/(psd_1*psd_2))