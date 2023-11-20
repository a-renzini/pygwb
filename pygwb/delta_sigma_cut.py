"""
In general, the noise level in ground-based detectors changes slowly on time-scales of tens of minutes to hours. The
variance associated to each segment is an indicator of that level of noise, which typically changes
at roughly the percent level from one data segment to the next. However, there are occasional very loud disturbances
to the detectors, such as glitches, which violate the Gaussianity of the noise. Auto-gating procedures are in place
to remove loud glitches from the data; however the procedure does not remove all non-stationarities. 
To avoid biases due to these noise events, an automated technique, called delta-sigma cut, 
to exclude them from the analysis has been developed, which flags specific segments to be cut from the analyzed set.

Examples
--------
    
As an example, we show how to use delta sigma cut. To this end, we import the relevant packages:
    
>>> import numpy as np
>>> from pygwb.delta_sigma_cut import dsc_cut
    
For concreteness, we use some randomly generated data arrays as placeholders for ``naive_sigma`` and
``sliding_sigma``:
    
>>> naive_sigma = np.random.normal(size=10)
>>> sliding_sigma = np.random.normal(size=10)
    
The ``dsc_cut`` method can be called with its default parameters:
    
>>> dsigma_mask, dsigma = dsc_cut(naive_sigma, sliding_sigma)
    
The result is a mask containing booleans, which indicates whether or not the segment should be
considered in the remainder of the analysis. In addition, the actual value of the difference
in sigmas is given as well.
"""

import numpy as np
from loguru import logger

from pygwb.postprocessing import calculate_point_estimate_sigma_spectra
from pygwb.util import calc_bias


def dsc_cut(
    naive_sigma: np.ndarray,
    slide_sigma: np.ndarray,
    dsc: float = 0.2,
    bf_ss: float = 1,
    bf_ns: float = 1,
):
    r"""
    Function that performs the delta sigma cut, a veto that marks certain GPS times as unusable if the estimation of
    the PSD in the naive (estimating sigma in bin J) and sliding (estimating sigma in bins J \pm 1) differ by more than
    a certain threshold:
    
    .. math::
        \frac{|\bar{\sigma}_{t, \alpha} b_{\rm avg} - \sigma_{t, \alpha} b_{\rm nav} |} {\bar{\sigma}_{t, \alpha} b_{\rm avg}}>{\rm threshold}
    
    Parameters
    =======

    naive_sigma: ``array_like``
        Array containing the naive sigmas.

    slide_sigma: ``array_like``
        Array containing the sliding sigmas.

    dsc: ``float``, optional
        Threshold used for the delta sigma cut. Default is 0.2.

    bf_ss: ``float``, optional
        Bias factor for sliding sigmas. Default is 1.

    bf_ns: ``float``, optional
        Bias factor for naive sigmas. Default is 1.

    Returns
    =======
    dsigma >= dsc: ``bool``
        Mask containing bools indicating whether the segment's delta sigma exceeds the threshold value or not. True indicates that the 
        corresponding GPS times were bad, whereas False denotes good GPS times.
    dsigma: ``array_like``
        Values of the difference between sliding sigma and naive sigma, i.e., the actual value of the delta sigma per segment.
    """
    dsigma = np.abs(slide_sigma * bf_ss - naive_sigma * bf_ns) / (slide_sigma * bf_ss)

    return dsigma >= dsc, dsigma

def run_dsc(
    dsc: float,
    segment_duration: int,
    psd1_naive: np.ndarray,
    psd2_naive: np.ndarray,
    psd1_slide: np.ndarray,
    psd2_slide: np.ndarray,
    alphas: np.ndarray,
    sample_rate: int,
    orf: np.array,
    fref: int,
    frequency_mask: np.array = True,
    window_fftgram_dict: dict = {"window_fftgram": "hann"},
    overlap_factor: float=0.5,
    N_average_segments_psd: int = 2,
    return_naive_and_averaged_sigmas: bool = False,
):
    """
    Function that runs the delta sigma cut.

    Parameters
    =======

    dsc: ``float``
        Threshold used for the delta sigma cut.

    segment_duration: ``int``
        Duration of each segment.

    psd1_naive, psd2_naive: ``array_like``
        An FFTgram of the PSD computed naively, as in in the particular bin J for detector 1 and 2.

    psd1_slide, psd2_slide: ``array_like``
        An FFTgram of the PSD computed by considering the noise in adjacent bins to the bin J, i.e. J-1, J+1 for
        detectors 1 and 2.

    alphas: ``array_like``
        The spectral indices to use. The bad GPS times from all alphas are combined at the end of this code.

    sample_rate: ``int``
        Sampling rate (Hz)

    notch_list_path: ``str``
        Path to the file containing the frequency notches to apply.
        
    orf: ``array_like``
        The overlap reduction function as a function of frequency that quantifies the overlap of a detector baseline,
        which depends on the detector locations, relative orientations, etc.

    fref: ``int``
        Reference frequency (Hz).

    window_fftgram_dict: ``dictionary``, optional
        Dictionary with window characteristics used in the computation of the sigmas, given the PSD. Default is 
        `(window_fftgram_dict={"window_fftgram": "hann"}`.

    frequency_mask: ``array_like``, optional
        Frequency mask to apply when computing the sigmas. Default is `True`.
    
    overlap_factor: ``float``, optional
        Overlap factor to use when computing the sigmas, given the PSD. Default is 0.5.
        
    N_average_segments_psd: ``int``, optional
        Number of segments to use during Welch averaging. Used in the computation of the bias factors. Default is 2.

    return_naive_and_averaged_sigmas: ``bool``, optional
        Option to return the naive and sliding sigmas. Default is `False`.

    Returns
    =======

    BadGPStimes: ``array_like``
        Array containing the bad GPS times to not be considered, based on the chosen value of the delta sigma cut.
        
    dsigmas_dict: ``array_like``
        Array containing the values of the difference between sliding sigma and naive sigma, i.e., the actual value of the delta sigma per segment.

    See also
    --------
    pygwb.postprocessing.calculate_point_estimate_sigma_spectra

    pygwb.util.calc_bias
    """
    logger.info("Running delta sigma cut")
    nalphas = len(alphas)
    times = np.array(psd1_naive.times)
    ntimes = len(times)
    df = psd1_naive.df.value
    dt = 1 / sample_rate  
    # Naive estimate
    bf_ns = calc_bias(
        segmentDuration=segment_duration,
        deltaF=df,
        deltaT=dt,
        N_avg_segs=1,
        window_fftgram_dict=window_fftgram_dict,
        overlap_factor=overlap_factor
    )
    # Sliding estimate
    bf_ss = calc_bias(
        segmentDuration=segment_duration,
        deltaF=df,
        deltaT=dt,
        N_avg_segs=N_average_segments_psd,
        window_fftgram_dict=window_fftgram_dict,
        overlap_factor=overlap_factor
    )
    freqs = np.array(psd1_naive.frequencies)
    overall_cut = np.zeros((ntimes, 1), dtype="bool")
    cuts = np.zeros((nalphas, ntimes), dtype="bool")
    naive_sigmas = np.zeros((nalphas, ntimes))
    slide_sigmas = np.zeros((nalphas, ntimes))
    dsigmas = np.zeros((nalphas, ntimes))

    for idx, alpha in enumerate(alphas):
        cut = np.zeros((ntimes, 1), dtype="bool")
        dsigma = np.zeros((ntimes, 1), dtype="float")
        for time in range(len(times)):
            psd1_naive_time = psd1_naive[time, :]
            psd1_slide_time = psd1_slide[time, :]
            psd2_naive_time = psd2_naive[time, :]
            psd2_slide_time = psd2_slide[time, :]

            naive_sigma_with_Hf = calculate_point_estimate_sigma_spectra(
                freqs=freqs,
                avg_psd_1=psd1_naive_time,
                avg_psd_2=psd2_naive_time,
                orf=orf,
                sample_rate=sample_rate,
                window_fftgram_dict=window_fftgram_dict,
                overlap_factor=overlap_factor,
                segment_duration=segment_duration,
                csd=None,
                fref=fref,
                alpha=alpha,
            )

            slide_sigma_with_Hf = calculate_point_estimate_sigma_spectra(
                freqs=freqs,
                avg_psd_1=psd1_slide_time,
                avg_psd_2=psd2_slide_time,
                orf=orf,
                sample_rate=sample_rate,
                window_fftgram_dict=window_fftgram_dict,
                overlap_factor=overlap_factor,
                segment_duration=segment_duration,
                csd=None,
                fref=fref,
                alpha=alpha,
            )

            naive_sensitivity_integrand_with_Hf = 1.0 / naive_sigma_with_Hf
            slide_sensitivity_integrand_with_Hf = 1.0 / slide_sigma_with_Hf

            naive_sigma_alpha = np.sqrt(
                1 / np.sum(naive_sensitivity_integrand_with_Hf[frequency_mask])
            )
            slide_sigma_alpha = np.sqrt(
                1 / np.sum(slide_sensitivity_integrand_with_Hf[frequency_mask])
            )
            naive_sigmas[idx, time] = naive_sigma_alpha
            slide_sigmas[idx, time] = slide_sigma_alpha

            cut[time], dsigma[time] = dsc_cut(
                naive_sigma=naive_sigma_alpha,
                slide_sigma=slide_sigma_alpha,
                dsc=dsc,
                bf_ss=bf_ss,
                bf_ns=bf_ns,
            )

        cuts[idx, :] = np.squeeze(cut)
        dsigmas[idx, :] = np.squeeze(dsigma)

    for time in range(len(times)):
        overall_cut[time] = any(cuts[:, time])

    BadGPStimes = times[np.squeeze(overall_cut)]

    dsigmas_dict = {}
    dsigmas_dict["alphas"] = alphas
    dsigmas_dict["times"] = times
    dsigmas_dict["values"] = dsigmas

    if return_naive_and_averaged_sigmas:
        dsigmas_dict["slide_sigmas"] = slide_sigmas * bf_ss
        dsigmas_dict["naive_sigmas"] = naive_sigmas * bf_ns

    return BadGPStimes, dsigmas_dict
