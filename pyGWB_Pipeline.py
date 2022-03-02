import json
import os
import sys
from pathlib import Path

import bilby
import matplotlib.pyplot as plt
import numpy as np
from gwpy import timeseries

from loguru import logger
import pygwb.argument_parser
from pygwb import network, orfs, pre_processing, spectral
from pygwb.baseline import Baseline
from pygwb.constants import H0, speed_of_light
from pygwb.delta_sigma_cut import run_dsc
from pygwb.detector import Interferometer
from pygwb.parameters import Parameters
from pygwb.postprocessing import postprocess_Y_sigma
from pygwb.util import calc_bias, calc_Y_sigma_from_Yf_varf, window_factors

if __name__ == "__main__":
    parser = pygwb.argument_parser.parser
    args = parser.parse_args()
    params = Parameters.from_file(args.param_file)
    if args.t0 is not None:
        params.t0 = args.t0
    if args.tf is not None:
        params.tf = args.tf
    params.alphas = json.loads(params.alphas_delta_sigma_cut)
    logger.info(f"Successfully imported parameters from paramfile.")
    outfile_path = Path(args.param_file)
    outfile_path = outfile_path.with_name(
        f"{outfile_path.stem}_final{outfile_path.suffix}"
    )
    params.save_paramfile(str(outfile_path))
    logger.info(f"Saved final param file at {outfile_path}.")


    '''
    X TODO X make this into a param in the arg class?
    '''
    Boolean_CSD = True  # if False, it will only compute the CSDs and PSDs, if True, it will compute until the point estimates

    ifo_H = Interferometer.from_parameters("H1", params)
    ifo_L = Interferometer.from_parameters("L1", params)
    logger.info(f"Loaded up interferometers with selected parameters.")

    base_HL = Baseline.from_parameters(ifo_H, ifo_L, params)
    logger.info(f"Baseline with interferometers {ifo_H.name}, {ifo_L.name} initialised.")

    logger.info(f"Setting PSDs and CSDs of the baseline...")
    base_HL.set_cross_and_power_spectral_density(params.frequency_resolution)
    base_HL.set_average_power_spectral_densities()
    base_HL.set_average_cross_spectral_density()

    logger.info(f"... done.")

    '''
    check nothing's gone wrong in the frequency handling...
    '''
    deltaF = base_HL.frequencies[1] - base_HL.frequencies[0]
    if abs(deltaF - params.frequency_resolution) > 1e-6:
        raise ValueError("Frequency resolution in PSD/CSD is different than requested.")

    '''
    eventually could move this into baseline?
    '''
    stride = params.segment_duration * (1 - params.overlap_factor)
    csd_segment_offset = int(np.ceil(params.segment_duration / stride))
    base_HL.interferometer_1.psd_spectrogram = base_HL.interferometer_1.psd_spectrogram[csd_segment_offset : - csd_segment_offset]
    base_HL.interferometer_2.psd_spectrogram = base_HL.interferometer_2.psd_spectrogram[csd_segment_offset : - csd_segment_offset]


    badGPStimes = base_HL.calculate_delta_sigma_cut(
        params.delta_sigma_cut,
        params.alphas,
        flow=params.flow,
        fhigh=params.fhigh,
    )

    logger.info(f"times flagged by the delta sigma cut as badGPStimes:{badGPStimes}")
    
    if Boolean_CSD:
        logger.info(f"calculating the point estimate and sigma...")
        base_HL.set_point_estimate_sigma_spectrum(
            badtimes=badGPStimes,
        )

        base_HL.set_point_estimate_sigma(
            alpha=params.alpha,
            fref=params.fref,
            flow=params.flow,
            fhigh=params.fhigh,
        )

        logger.success(f"\tpyGWB: {base_HL.point_estimate:e}")
        logger.success(f"\tpyGWB: {base_HL.sigma:e}")

        data_file_name = f"Y_sigma_{int(params.t0)}-{int(params.tf)}"

        logger.info("saving Y_f and sigma_f to file.")
        base_HL.save_data(
            params.save_data_type,
            data_file_name,
            base_HL.frequencies,
            base_HL.point_estimate_spectrum.value,
            base_HL.sigma_spectrum.value,
            base_HL.point_estimate,
            base_HL.sigma,
        )

    logger.info("saving average psds and csd to file.")

    data_file_name = f"psds_csds_{int(params.t0)}-{int(params.tf)}"

    base_HL.save_data_csd(
        params.save_data_type, data_file_name, base_HL.frequencies, base_HL.csd, base_HL.interferometer_1.average_psd, base_HL.interferometer_2.average_psd
    )
