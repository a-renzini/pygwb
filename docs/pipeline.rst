=========================================
Using ``pygwb_pipe``: a quickstart manual
=========================================

The various modules of the ``pygwb`` package can be combined into a pipeline, as done in the ``pygwb_pipe`` script. This script 
takes the data as input and outputs an estimator of the point estimate and variance of the gravitational-wave background (GWB) for these
data. More information on how the various modules interact and are combined into a pipeline can be found in the `pygwb paper <https://arxiv.org/pdf/2303.15696.pdf>`_.

.. note::
  The proposed ``pygwb_pipe`` pipeline is only one of the many ways to assemble the ``pygwb`` modules, and users should
  feel free to create their own pipeline, that addresses their needs.

**1. Script parameters**
========================

The parameters of the ``pygwb_pipe`` script can be visualized by running the following command:

.. code-block:: shell

   pygwb_pipe --help

This will display the following set of parameters, which can be passed to the pipeline:

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

As can be seen, all of the parameters above come with a brief description, which should help the user identify its functionality. In particular,
we note that the above parameters are the ones present in the ``pygwb.parameters`` module. For more information, one can have a look at the 
`pygwb paper <https://arxiv.org/pdf/2303.15696.pdf>`_, where more details are provided.

.. tip::
  Feeling overwhelmed with the amount of parameters? Make sure to have a look to the ``pygwb.parameters`` `documentation <api/pygwb.parameters.html>`_.

**2. Running the script**
========================

Although all of the parameters shown above can be passed to the script, we start by running ``pygwb_pipe`` without passing any optional parameters directly to the script.
The only required argument is a path to a parameter file, which contains the parameter values
to use for the analysis. As an example, one can run the script with the ``parameters.ini`` file provided in the ``pygwb_pipe`` directory of the 
`repository <https://github.com/a-renzini/pygwb/blob/master/pygwb_pipe/parameters.ini>`_. To test the pipeline, run the command:

.. code-block:: shell

  pygwb_pipe --param_file {path_to_param_file}

The output should be of the command above should be:

.. code-block:: c

  2023-02-21 14:43:40.817 | SUCCESS  | __main__:main:160 - Ran stochastic search over times 1247644138-1247645038                                           
  2023-02-24 16:35:25.625 | SUCCESS  | __main__:main:163 - POINT ESTIMATE: -6.496991e-06
  2023-02-24 16:35:25.625 | SUCCESS  | __main__:main:164 - SIGMA: 2.688777e-06

However, one could have decided to run with different parameters. An option is to modify the ``parameters.ini`` file, or one could also pass the parameters as arguments
to the script directly. For example:

.. code-block:: shell

  pygwb_pipe --param_file {path_to_param_file} --apply_dsc True --gate_data True

.. warning::

  Passing any parameters through the command line overwrites the value in the ``parameters.ini`` file.

**Note: detector--specific parameters** 

It is possible to pass detector--specific parameters, both in the ``.ini`` file and through shell. The syntax is:

.. code-block:: shell

  param = {IFO1:val1 IFO2:val2}

For example, if passing different channel names for LIGO Hanford and LIGO Livingston:

.. code-block:: shell

  channel = {H1:GWOSC-16KHZ_R1_STRAIN L1:PYGWB-SIMULATED_STRAIN} 

When passing through shell, double quotes are required, i.e., 

.. code-block:: shell

  --channel "{H1:GWOSC-16KHZ_R1_STRAIN L1:PYGWB-SIMULATED_STRAIN}"

**3. Output of the script**
===========================

As mentioned previously, the purpose of the ``pygwb`` analysis package is to compute an estimator of the GWB, through the computation of a 
point estimate and variance spectrum, which can be translated into one point estimate and variance. By default, the output of the analysis will be saved in 
the ``./output`` folder of your directory, unless otherwise specified through the ``--output_path`` argument of the script.

A few files can be found in this directory, including a version of the parameters file used for the
analysis. Note that this takes into account any parameters that were modified through the command line.

Additionally, the power-spectral densities (PSDs) and cross-spectral densities (CSDs) are saved in a file with naming convention:

.. code-block:: shell

  psds_csds_{start_time_of_job}_{job_duration}.npz

.. tip::
  Not sure about what is exactly in a file? Load in the file and print out all its `keys` as shown 
  `here <https://stackoverflow.com/questions/49219436/how-to-show-all-the-element-names-in-a-npz-file-without-having-to-load-the-compl>`_.

A second file contains the actual point estimate spectrum, variance spectrum, point estimate and variance. This information is accessible in:

.. code-block:: shell

  point_estimate_sigma_{start_time_of_job}_{job_duration}.npz

Furthermore, if the script was run with ``--pickle_out True``, a ``pickle`` file will be present in the output directory, containing a pickled
version of the baseline. This contains all the information present in the other two ``npz`` files, but allows the user to create a baseline object
from this ``pickle`` file. More information about how to create a baseline from such a file can be found `here <api/pygwb.baseline.Baseline.html#pygwb.baseline.Baseline.load_from_pickle>`_.

.. warning::

  Saving ``pickle`` files can take up a lot of memory. Furthermore, loading in a baseline from ``pickle`` file can take quite some time. Working 
  with ``npz`` files is therefore recommended, when possible.

.. note::
  
  Depending on the parameters used to run ``pygwb_pipe``, the output of the script and amount of files might differ from the one described here.

This tutorial provides a brief overview of the ``pygwb_pipe`` script and how to run it for one job, i.e., a small stretch of data. In practice, 
however, one probably wants to analyze months, if not years, of data. To address this need, ``pygwb_pipe`` can be run on multiple jobs, i.e., different
stretches of data, through parallelization using Condor (more information about condor can be found `here <https://htcondor.readthedocs.io/en/latest/index.html>`_).
The concrete implementation within the ``pygwb`` package is outlined in the `following tutorial <multiple_jobs.html>`_.
