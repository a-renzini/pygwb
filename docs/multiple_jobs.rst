====================================
Running multiple ``pygwb_pipe`` jobs
====================================

**1. Writing and submitting a `dag` file**
=========

We are now ready to condorise the pipeline and run a batch of jobs, just like the one run in point 4.

* *writing the* ``dag`` *file*

To prepare a dag file one  can use the ``pygwb_dag`` script:

.. code-block:: shell

   pygwb_dag --help

  --subfile SUBFILE     Submission file.
  --jobfile JOBFILE     Job file with start and end times and duration for each job.
  --flag FLAG           Flag that is searched for in the DQSegDB.
  --t0 T0               Begin time of analysed data, will query the DQSegDB. If used with jobfile, it is an optional argument if one does not wish to analyse the whole job
                        file
  --tf TF               End time of analysed data, will query the DQSegDB. If used with jobfile, it is an optional argument if one does not wish to analyse the whole job
                        file
  --parentdir PARENTDIR
                        Starting folder.
  --param_file PARAM_FILE
                        Path to parameters.ini file.
  --dag_name DAG_NAME   Dag file name.
  --apply_dsc APPLY_DSC
                        Apply delta-sigma cut flag for pygwb_pipe.
  --pickle_out PICKLE_OUT
                        Pickle output Baseline of the analysis.
  --wipe_ifo WIPE_IFO   Wipe interferometer data to reduce size of pickled Baseline.
  --calc_pt_est CALC_PT_EST
                        Calculate omega point estimate and sigma from data.

This script passes on relevant arguments to ``pygwb_pipe``, such as the parameter file and the ``apply_dsc`` Flag, etc.
Note that the condor submission file is not included in the package. Its compilation will depend on the specific cluster/setup used, and is left up to the user.

* *submitting the job*

The ``dag`` file is now created in the ``output`` folder. To submit the job, navigate to that folder and run

.. code-block:: shell
   
   condor_submit_dag {your-dag-file.dag}

If you have not specified the ``dag`` name at the previous step, the current default name is ``dag_name.dag``.

**2. Combining the output**
==========

To combine the output files from many runs of ``pygwb_pipe`` on different times one may use ``pygwb_combine``:

.. code-block:: shell

   >> pygwb_combine -h

  --data_path DATA_PATH [DATA_PATH ...]
                        Path to data files or folder.
  --alpha ALPHA         Spectral index alpha to use for spectral re-weighting.
  --fref FREF           Reference frequency to use when presenting results.
  --param_file PARAM_FILE
                        Parameter file
  --h0 H0               Value of h0 to use. Default is pygwb.constants.h0.
  --combine_coherence COMBINE_COHERENCE
                        Calculate combined coherence over all available data.
  --coherence_path COHERENCE_PATH [COHERENCE_PATH ...]
                        Path to coherence data files, if individual files are
                        passed.
  --out_path OUT_PATH   Output path.
  --file_tag FILE_TAG   File naming tag. By default, reads in first and last
                        time in dataset.

This command produces combined spectra in the desired output folder.

**Important Notes**
==========

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
