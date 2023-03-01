import numpy as np

def calculate_coherence(psd_1_spectrogram, psd_2_spectrogram, csd_spectrogram):
    '''
    Calculate a coherence spectrum given two power spectrograms and a cross-spectrogram.

    Parameters
    ==========
    psd_1_spectrogram: gwpy.Spectrogram
        PSD spectrogram of first detector.
    psd_2_spectrogram: gwpy.Spectrogram
        PSD spectrogram of second detector.
    csd_spectrogram: gwpy.Spectrogram
        Spectrogram of the cross-spectral density of the two detectors.
    '''
    #I don't think we need these terms anymore:
    #fftlength = int(1.0/psd_1_spectrogram.df.value)
    #norm1 = int(np.floor(duration/(fftlength*overlap_factor))-1)
    psd_1_average = np.mean(psd_1_spectrogram, axis=0)
    psd_2_average = np.mean(psd_2_spectrogram, axis=0)
    csd_average = np.mean(psd_2_spectrogram, axis=0)
    return  np.real(csd_average*np.conj(csd_average)/(psd_1_average*psd_2_average))

