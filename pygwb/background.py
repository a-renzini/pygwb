import warnings

import bilby
import numpy as np
from loguru import logger
from tqdm import tqdm

from pygwb.constants import H0


def calculate_num_injections(T_obs, zs, p_dz):
    """
    Calculate the number of mergers in a given time T_obs, assuming a given merger rate redshift distribution.
    
    Parameters
    =======
    
    T_obs : ``float``
        Observation time in years.
    zs : ``np.array``
        Array of redshifts.
    p_dz : ``np.array``
        Redshift probability distribution.
        
    Returns
    =======
    
    N : ``int``
        Number of mergers.
    """
    
    p_dz_centers = (p_dz[1:] + p_dz[:-1])/2.
    total_sum = np.sum(np.diff(zs) * p_dz_centers)
    N = T_obs * total_sum
    return N
    
def compute_Omega_from_CBC_dictionary(injection_dict, sampling_frequency, T_obs, return_spectrum=True, f_ref=25, waveform_duration=10, waveform_approximant="IMRPhenomD", waveform_reference_frequency=25, waveform_minimum_frequency=20):
    """
    Compute the total :math:`\Omega_{\\text{GW}}` injected in the data when injecting individual CBCs.
    
    Parameters
    =======

    injection_dict: ``dict``
        Dictionary of injection samples (set of waveform parameters) to calculate
        :math:`\Omega_{\\text{GW}}` from.
    sampling_frequency: ``int``
        Sampling frequency of the final output spectrum.
    T_obs: ``float``
        Total observation time in seconds.
    return_spectrum : ``bool``, optional
	Return the full Omega spectrum. The default is True.
    f_ref : ``float``, optional
	Reference frequency to compute Omega_ref [Hz]. The default is 25 Hz.
    waveform_duration: ``int``, optional
	Duration in seconds for the waveform generation; longer times will take longer. Default is 10 seconds.
    waveform_approximant: ``str``, optional
        Waveform approximant to use for signal calculation. Default is IMRPhenomD.
    waveform_reference_frequency: ``float``, optional
        Reference frequency for waveform calcultion. Default is 25 Hz.
    waveform_minimum frequency: ``float``, optional
        Minimum frequency for waveform calcultion. Default is 20 Hz.

    Returns
    =======

    freqs_psd : ``array_like``
	Frequency array.
    Omega_GW_freq: ``array_like``
	:math:`\Omega_{\\text{GW}}` spectrum array.
    
    OR if return_spectrum is False:
    
    Omega_ref: ``float``
        :math:`\Omega_{\\text{GW}}` at :math:`f=f_{\\text{ref}}`.
    """
    
    waveform_generator = bilby.gw.WaveformGenerator(
	duration=waveform_duration, sampling_frequency=sampling_frequency,
	frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
	parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
	waveform_arguments={
	    "waveform_approximant": waveform_approximant,
	    "reference_frequency": waveform_reference_frequency,
            "minimum_frequency": waveform_minimum_frequency,
	},
    )
    freqs_psd = waveform_generator.frequency_array
    
    key_1 = list(injection_dict.keys())[0]
    try:
        N_inj = len(injection_dict[key_1]['content'])
    except:
        N_inj = len(injection_dict[key_1])

    psd_evs = np.zeros(len(freqs_psd))
    logger.info('Compute the total injected Omega for ' + str(N_inj) + ' injections')

    check_injection_dictionary(injection_dict, N_inj)

    # Loop over injections
    for i in tqdm(range(N_inj)):
        inj_params = {}
        # Generate the individual parameters dictionary for each injection
        for k in injection_dict.keys():
            try:
                inj_params[k] = injection_dict[k]['content'][i]
            except:
                inj_params[k] = injection_dict[k][i]
        # Get frequency domain waveform
        polarizations = waveform_generator.frequency_domain_strain(inj_params)
        
        # Add PSDs of individual events
        psd_evs += np.abs(polarizations['plus'])**2 + np.abs(polarizations['cross'])**2         
    Omega_GW_freq = 4.* np.pi**2 * freqs_psd**3 * psd_evs / (3 * H0.si.value**2) / T_obs 

    logger.debug('Compute Omega_ref at f_ref=' + format(f_ref, '.0f') + ' Hz')
    df = freqs_psd[1] - freqs_psd[0]
    fmin = freqs_psd[0]
    
    i_fref = int((f_ref - fmin) / df)
    logger.debug('True f_ref=' + format(freqs_psd[i_fref], '.1f'))
    
    Omega_ref = Omega_GW_freq[i_fref]
    logger.info(r'Omega_ref=' + format(Omega_ref, '.2e'))

    
    if return_spectrum == True:
        return np.array(freqs_psd), np.array(Omega_GW_freq)
    
    else:
        return Omega_ref

def check_injection_dictionary(injection_dict, N_inj):
    '''
    Check injection dict content
    '''
    # check for missing keys and set values to defaults
    if not 'phase' in injection_dict:
        warnings.warn("'phase' parameter not set in injection set; setting to random value.")
        injection_dict['phase']=2*np.pi*np.random.rand(N_inj)
    if not 'theta_jn' in injection_dict:
        warnings.warn("'theta_jn' parameter not set in injection set; setting to random value.")
        cos_inc = np.random.rand(N_inj)*2.-1.
        injection_dict['theta_jn']=np.arccos(cos_inc)
    if not 'a_1' in injection_dict:
        warnings.warn("'a_1' spin parameter not set in injection set; setting to 0.")
        injection_dict['a_1']=np.zeros(N_inj)
    if not 'a_2' in injection_dict:
        warnings.warn("'a_2' spin parameter not set in injection set; setting to 0.")
        injection_dict['a_2']=np.zeros(N_inj)
    if not 'tilt_1' in injection_dict:
        warnings.warn("'tilt_1' spin parameter not set in injection set; setting to 0.")
        injection_dict['tilt_1']=np.zeros(N_inj)
    if not 'tilt_2' in injection_dict:
        warnings.warn("'tilt_2' spin parameter not set in injection set; setting to 0.")
        injection_dict['tilt_2']=np.zeros(N_inj)
    
