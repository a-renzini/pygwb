============
Using ``pygwb_pipe``: a quickstart manual
============


**1. Testing the pipeline**
=========

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
  --pickle_out PICKLE_OUT
                        Pickle output Baseline of the analysis.
  --wipe_ifo WIPE_IFO   Wipe interferometer data to reduce size of pickled Baseline.
  --t0 T0               Initial time.
  --tf TF               Final time.
  --data_type DATA_TYPE
                        Type of data to access/download; options are private, public, local. Default is public.
  --channel CHANNEL     Channel name; needs to match an existing channel. Default is "GWOSC-16KHZ_R1_STRAIN"
  --new_sample_rate NEW_SAMPLE_RATE
                        Sample rate to use when downsampling the data (Hz). Default is 4096 Hz.
  --input_sample_rate INPUT_SAMPLE_RATE
                        Sample rate of the read data (Hz). Default is 16384 Hz.
  --cutoff_frequency CUTOFF_FREQUENCY
                        Lower frequency cutoff; applied in filtering in preprocessing (Hz). Default is 11 Hz.
  --segment_duration SEGMENT_DURATION
                        Duration of the individual segments to analyse (seconds). Default is 192 seconds.
  --number_cropped_seconds NUMBER_CROPPED_SECONDS
                        Number of seconds to crop at the start and end of the analysed data (seconds). Default is 2 seconds.
  --window_downsampling WINDOW_DOWNSAMPLING
                        Type of window to use in preprocessing. Default is "hamming"
  --ftype FTYPE         Type of filter to use in downsampling. Default is "fir"
  --frequency_resolution FREQUENCY_RESOLUTION
                        Frequency resolution of the final output spectrum (Hz). Default is 1\/32 Hz.
  --polarization POLARIZATION
                        Polarisation type for the overlap reduction function calculation; options are scalar, vector, tensor. Default is tensor.
  --alpha ALPHA         Spectral index to filter the data for. Default is 0.
  --fref FREF           Reference frequency to filter the data at (Hz). Default is 25 Hz.
  --flow FLOW           Lower frequency to include in the analysis (Hz). Default is 20 Hz.
  --fhigh FHIGH         Higher frequency to include in the analysis (Hz). Default is 1726 Hz.
  --coarse_grain COARSE_GRAIN
                        Whether to apply coarse graining to the spectra. Default is 0.
  --interferometer_list INTERFEROMETER_LIST
                        List of interferometers to run the analysis with. Default is ["H1", "L1"]
  --local_data_path_dict LOCAL_DATA_PATH_DICT
                        Dictionary of local data, if the local data option is chosen. Default is empty.
  --notch_list_path NOTCH_LIST_PATH
                        Path to the notch list file. Default is empty.
  --N_average_segments_welch_psd N_AVERAGE_SEGMENTS_WELCH_PSD
                        Number of segments to average over when calculating the psd with Welch method. Default is 2.
  --window_fft_dict WINDOW_FFT_DICT
                        Dictionary containing name and parameters relative to which window to use when producing fftgrams for psds and csds. Default is "hann".
  --calibration_epsilon CALIBRATION_EPSILON
                        Calibation coefficient. Default is 0.
  --overlap_factor OVERLAP_FACTOR
                        Factor by which to overlap consecutive segments for analysis. Default is 0.5 (50% overlap)
  --zeropad_csd ZEROPAD_CSD
                        Whether to zeropad the csd or not. Default is True.
  --delta_sigma_cut DELTA_SIGMA_CUT
                        Cutoff value for the delta sigma cut. Default is 0.2.
  --alphas_delta_sigma_cut ALPHAS_DELTA_SIGMA_CUT
                        List of spectral indexes to use in delta sigma cut calculation. Default is [-5, 0, 3].
  --save_data_type SAVE_DATA_TYPE
                        Suffix for the output data file. Options are hdf5, npz, json, pickle. Default is json.
  --time_shift TIME_SHIFT
                        Seconds to timeshift the data by in preprocessing. Default is 0.
  --gate_data GATE_DATA
                        Whether to apply self-gating to the data in preprocessing. Default is False.
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
  --tag TAG             Hint for the read_data function to retrieve one specific type of data, e.g.: C00, C01
  --return_naive_and_averaged_sigmas RETURN_NAIVE_AND_AVERAGED_SIGMAS
                        option to return naive and sliding sigmas from delta sigma cut. Default value: False

To test the pipeline, simply run a command like

.. code-block:: shell

   pygwb_pipe --param_file {path_to_param_file}

When running on the file ``pygwb_pipe/parameters.ini`` in the repo, one should get as final result

.. code-block:: c

   2022-07-06 16:54:56.084 | SUCCESS  | __main__:main:148 - Ran stochastic search over times 1247644138-1247645038
   2022-07-06 16:54:56.085 | SUCCESS  | __main__:main:151 -        POINT ESIMATE: -6.187323e-06
   2022-07-06 16:54:56.085 | SUCCESS  | __main__:main:152 -        SIGMA: 2.562031e-06
   2022-07-06 16:54:56.085 | INFO     | __main__:main:156 - Saving point_estimate and sigma spectrograms, spectra, and final values to file.
   2022-07-06 16:54:56.085 | INFO     | __main__:main:159 - Saving average psds and csd to file.
   2022-07-06 16:54:56.350 | INFO     | __main__:main:170 - Pickling the baseline.


**2. Writing and submitting a `dag` file**
=========

We are now ready to condorise the pipeline and run a batch of jobs, just like the one run in point 4. For the purposes of this example, we'll run on mock data, using the `local` data option available in the package. Let's take this steps:

* *writing the* ``dag`` *file*

To start, let's copy the `DAG` folder to the location where you want to start your jobs. I suggest leaving the ``pygwb`` installation folder, and creating a ``pygwb_run`` folder somewhere completely different (this could even be in your ``public_html`` folder!). Once you have navigated to the folder you want to start from, run

.. code-block:: shell

   cp -r {path-to-pygwb_main_folder}/pygwb_pipe/DAG/* .

You should now see some new files and folders in your ``run`` folder. amongst these, there is a handy script to prepare a ``dag`` file for the mock data analysis submission. To see how to use it, run

.. code-block:: shell

   ./make_DAG_pygwb_pipe -h

As you can see, it expects the following arguments:

.. code-block:: shell

  --subfile SUBFILE     Submission file.
  --data_path DATA_PATH
                        Path to data files folder.
  --parentdir PARENTDIR
                        Starting folder.
  --param_file PARAM_FILE
                        Path to parameters.ini file.
  --dag_name DAG_NAME   Dag file name.


not all of which are necessary. For a basic condor run, you can use the following recipe to compile your ``dag``

.. code-block:: shell

   ./make_DAG_pygwb_pipe --subfile {full-path-to-your-run-dir}/condor/Simulated_Data_New_Pipeline.sub --data_path /home/arianna.renzini/PROJECTS/SMDC_2021/100_day_test_pygwb/MDC_Generation_2/output/ --param_file {full-path-to-your-installation-dir}/pygwb_pipe/parameters_mock.ini --parentdir {full-path-to-your-run-dir}

* *submitting the job*

The ``dag`` file is now created in the ``output`` folder. To submit the job, navigate to that folder and run

.. code-block:: shell
   
   condor_submit_dag {your-dag-file.dag}

If you have not specified the ``dag`` name at the previous step, the current default name is ``condor_simulated_100_day_MDC_2.dag``.

* *writing* ``dag`` *file when joblength and length of datafiles in data_path are misaligned*

It is possible that you want to run job files with a certain length such that the job files need data from two subsequent datafiles in data_path instead of getting its data from only one. In that case, you will have to use a different script ``make_DAG_pygwb_pipe_multifile`` which makes the corresponding dag file accounting for the misalignment of the times of the job files and those of the datafiles. To help you with using it, run

.. code-block:: shell

   ./make_DAG_pygwb_pipe_multifile -h

As you can see, it expects mostly the same arguments as ``make_DAG_pygwb_pipe``. However, there is one (optional) additional argument: 

.. code-block:: shell

  --job_duration JOB_DURATION 
                        Each job duration in seconds.

If this argument is not provided, the job_duration will be equal to the file length of the datafiles in ``data_path``. To run this script, you can try: 

.. code-block:: shell

   ./make_DAG_pygwb_pipe_multifile --subfile {full-path-to-your-run-dir}/condor/Simulated_Data_New_Pipeline.sub --data_path /home/arianna.renzini/PROJECTS/SMDC_2021/100_day_test_pygwb/MDC_Generation_2/output/ --job_duration 10000 --param_file {full-path-to-your-installation-dir}/pygwb_pipe/parameters_mock.ini --parentdir {full-path-to-your-run-dir}

Submitting this dag file can be done in the same way as described above.

**3. Combining the output**
==========

To combine the output files from many runs of ``pygwb_pipe`` on different times one may use ``pygwb_combine``:

.. code-block:: shell

   >> pygwb_combine -h

  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to data files folder.
  --alpha ALPHA         Spectral index alpha to use for spectral re-weighting.
  --fref FREF           Reference frequency to use when presenting results.
  --param_file PARAM_FILE
                        Parameter file
  --h0 H0               Value of h0 to use. Default is 0.7.
  --out_path OUT_PATH   Output path. 
