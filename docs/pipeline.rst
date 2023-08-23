=========================================
Using ``pygwb_pipe``: a quickstart manual
=========================================


**1. Testing the pipeline**
===========================

You can view the pipeline run options by executing


.. code-block:: shell

   pygwb_pipe --help

This will list all arguments of the pipeline:

.. code-block:: shell

  --param_file PARAM_FILE                                                                                                                                                     
                        Parameter file to use for analysis.                                                                                                                   
  --output_path OUTPUT_PATH                                                                                                                                                   
                        Location to save output to.                                                                                                                           
  --calc_pt_est CALC_PT_EST                                                                                                                                                   
                        Calculate omega point estimate and sigma from data.                                                                                                   
  --apply_dsc APPLY_DSC                                                                                                                                                       
                        Apply delta sigma cut when calculating final output.                                                                                                  
  --pickle_out PICKLE_OUT                                                                                                                                                     
                        Pickle output Baseline of the analysis.                                                                                                               
  --wipe_ifo WIPE_IFO   Wipe interferometer data to reduce size of pickled Baseline.                                                                                          
  --t0 T0               Initial time.                                                                                                                                         
  --tf TF               Final time.                                                                                                                                           
  --data_type DATA_TYPE                                                                                                                                                       
                        Type of data to access/download; options are private,                                                                                                 
                        public, local. Default is public.                                                                                                                     
  --channel CHANNEL     Channel name; needs to match an existing channel. Default is                                                                                          
                        "GWOSC-16KHZ_R1_STRAIN"                                                                                                                               
  --new_sample_rate NEW_SAMPLE_RATE                                                                                                                                           
                        Sample rate to use when downsampling the data (Hz). Default                                                                                           
                        is 4096 Hz.                                                                                                                                           
  --input_sample_rate INPUT_SAMPLE_RATE                                                                                                                                       
                        Sample rate of the read data (Hz). Default is 16384 Hz.                                                                                               
  --cutoff_frequency CUTOFF_FREQUENCY                                                                                                                                         
                        Lower frequency cutoff; applied in filtering in                                                                                                       
                        preprocessing (Hz). Default is 11 Hz.                                                                                                                 
  --segment_duration SEGMENT_DURATION                                                                                                                                         
                        Duration of the individual segments to analyse (seconds).                                                                                             
                        Default is 192 seconds.                                                                                                                               
  --number_cropped_seconds NUMBER_CROPPED_SECONDS                                                                                                                             
                        Number of seconds to crop at the start and end of the                                                                                                 
                        analysed data (seconds). Default is 2 seconds.                                                                                                        
  --window_downsampling WINDOW_DOWNSAMPLING                                                                                                                                   
                        Type of window to use in preprocessing. Default is "hamming"                                                                                          
  --ftype FTYPE         Type of filter to use in downsampling. Default is "fir"
  --frequency_resolution FREQUENCY_RESOLUTION
                        Frequency resolution of the final output spectrum (Hz).                                                                                               
                        Default is 1\/32 Hz.
  --polarization POLARIZATION
                        Polarisation type for the overlap reduction function calculation; options are scalar, vector, tensor. Default is tensor.                             
  --alpha ALPHA         Spectral index to filter the data for. Default is 0.
  --fref FREF           Reference frequency to filter the data at (Hz). Default is 25 Hz.
  --flow FLOW           Lower frequency to include in the analysis (Hz). Default is 20 Hz.
  --fhigh FHIGH         Higher frequency to include in the analysis (Hz). Default is 1726 Hz.
  --coarse_grain COARSE_GRAIN
                        Whether to apply coarse graining to the spectra. Default is 0.
  --interferometer_list INTERFEROMETER_LIST [INTERFEROMETER_LIST ...]                                                                                                         
                        List of interferometers to run the analysis with. Default is                                                                                          
                        ["H1", "L1"]                                                                                                                                          
  --local_data_path LOCAL_DATA_PATH                                                                                                                                           
                        Path(s) to local data, if the local data option is chosen.                                                                                            
                        Default is empty.                                                                                                                                     
  --notch_list_path NOTCH_LIST_PATH                                                                                                                                           
                        Path to the notch list file. Default is empty.                                                                                                        
  --N_average_segments_welch_psd N_AVERAGE_SEGMENTS_WELCH_PSD                                                                                                                 
                        Number of segments to average over when calculating the psd                                                                                           
                        with Welch method. Default is 2.                                                                                                                      
  --window_fft_dict WINDOW_FFT_DICT                                                                                                                                           
                        Dictionary containing name and parameters relative to which                                                                                           
                        window to use when producing fftgrams for psds and csds.                                                                                              
                        Default is "hann".                                                                                                                                    
  --calibration_epsilon CALIBRATION_EPSILON                                                                                                                                   
                        Calibation coefficient. Default 0.                                                                                                                  
  --overlap_factor OVERLAP_FACTOR
                        Factor by which to overlap consecutive segments for
                        analysis. Default is 0.5 (50% overlap)
  --zeropad_csd ZEROPAD_CSD
                        Whether to zeropad the csd or not. Default is True.
  --delta_sigma_cut DELTA_SIGMA_CUT
                        Cutoff value for the delta sigma cut. Default is 0.2.
  --alphas_delta_sigma_cut ALPHAS_DELTA_SIGMA_CUT [ALPHAS_DELTA_SIGMA_CUT ...]
                        List of spectral indexes to use in delta sigma cut
                        calculation. Default is [-5, 0, 3].
  --save_data_type SAVE_DATA_TYPE
                        Suffix for the output data file. Options are hdf5, npz,
                        json, pickle. Default is json.
  --time_shift TIME_SHIFT
                        Seconds to timeshift the data by in preprocessing. Default
                        is 0.
  --gate_data GATE_DATA
                        Whether to apply self-gating to the data in preprocessing.
                        Default is False.
  --gate_tzero GATE_TZERO
                        Gate tzero. Default is 1.0.
  --gate_tpad GATE_TPAD
                        Gate tpad. Default is 0.5.
  --gate_threshold GATE_THRESHOLD
                        Gate threshold. Default is 50.
  --cluster_window CLUSTER_WINDOW
                        Cluster window. Default is 0.5.
  --gate_whiten GATE_WHITEN
                        Whether to whiten when gating. Default is True.
  --tag TAG             Hint for the read_data function to retrieve one specific
                        type of data, e.g.: C00, C01
  --return_naive_and_averaged_sigmas RETURN_NAIVE_AND_AVERAGED_SIGMAS
                        option to return naive and sliding sigmas from delta sigma
                        cut. Default value: False

To test the pipeline, simply run a command like

.. code-block:: shell

   pygwb_pipe --param_file {path_to_param_file} --apply_dsc False

When running on the file ``pygwb_pipe/parameters.ini`` in the repo, one should get as final result

.. code-block:: c

   2023-02-21 14:43:40.817 | SUCCESS  | __main__:main:160 - Ran stochastic search over times 1247644138-1247645038                                           
   2023-02-24 16:35:25.625 | SUCCESS  | __main__:main:163 - POINT ESTIMATE: -6.496991e-06
   2023-02-24 16:35:25.625 | SUCCESS  | __main__:main:164 - SIGMA: 2.688777e-06

Note that this automatically includes the default notching. If an error related to the notch file appears, it may be necessary to add the correct path explicitly in the ``.ini`` file used.

**Important Notes**
===================

**i. Detector--specific parameters** 

It is possible to pass detector--specific parameters, both in the ``.ini`` file and through shell. The Syntax is:

.. code-block:: shell

  param = {IFO1:val1 IFO2:val2}

For example, if passing different channel names for Hanford and Livingston:

.. code-block:: shell

  channel = {H1:GWOSC-16KHZ_R1_STRAIN L1:PYGWB-SIMULATED_STRAIN} 

When passing through shell, double quotes are required, i.e., 

.. code-block:: shell

  --channel "{H1:GWOSC-16KHZ_R1_STRAIN L1:PYGWB-SIMULATED_STRAIN}"
