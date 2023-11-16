====================================
Running multiple ``pygwb_pipe`` jobs
====================================

In practice, one will probably want to run ``pygwb`` on long stretches of data. This is achieved most easily by splitting
the large data set in smaller chunks of data. These can then be analyzed individually, and combined after the analysis to form
one overall result for the whole data set. To this end, ``pygwb`` comes with two scripts: ``pygwb_dag`` and ``pygwb_combine``. 
The former allows the user to run ``pygwb_pipe`` (for which a tutorial can be found `here <pipeline.html>`_) simultaneously on shorter stretches of data, 
whereas the latter allows to combine the output of the individual runs into an overall result for the whole data set.

**The pygwb_dag script**
========================

**1. Script parameteres**
-------------------------

To be able to run multiple ``pygwb_pipe`` jobs simultaneously, ``pygwb`` relies on `Condor <https://htcondor.readthedocs.io/en/latest/>`_.
This requires a ``dag`` file, which contains information about all the jobs, i.e., running ``pygwb_pipe`` on different stretches of data.
In ``pygwb``, this file can be created by using the ``pygwb_dag`` script. To visualize the expected arguments of the script, one can call:

.. code-block:: shell

   pygwb_dag --help

This will display the required parameters, together with a small description:

.. code-block:: shell

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

An important argument of the script, is the path to the job file, passed through ``--jobfile``. The job file is a simple ``.txt`` file and contains the different jobs, or in other words,
the different stretches of data to run the analysis on. For concretenes, consider the case where one would want to run ``pygwb`` on 12000 seconds of data, but split into smaller jobs.
The job file could then look as follows:

.. code-block:: shell

  1 0 4000  4000
  1 4000  9000  5000
  1 9000 12000 3000

The first column does not play a role, the second and third colum indicate the start and end time of the job, respectively, whereas the last column shows the duration of the job, i.e., the 
difference between end and start time. The job file therefore allows the script to *know* on which stretches of data to run. In case one wants to run on a subset of the jobs in the 
job file, one can pass an additional start and end time to the script through the ``--t0`` and ``--tf`` arguments.

The ``--parentdir`` allows to pass the full path to the run directory, and the ``--param_file`` should point to the parameter file to be used by ``pygwb_pipe``.

.. seealso::
  For more information about ``pygwb_pipe`` and the usage of a parameter file, we refer the user to the tutorial `here <pipeline.html>`_.

For the remainder of the arguments, we refer the user to the ``pygwb_pipe`` `tutorial <pipeline.html>`_, as the ``dag`` file passes the relevant arguments to ``pygwb_pipe`` behind the screens, 
e.g., the parameter file and the ``apply_dsc`` flag.

Note that an additional argument should be passed to the script, namely the submission file. This file passes necessary information to Condor, and the cluster/server on which the user is
running the ``pygwb`` jobs. 

.. warning::
  The Condor submission file, passed through ``--subfile``, is not included in the ``pygwb`` package. Its specific implementation will depend on the server or cluster where the user runs the analysis.
  More information about Condor, together with inspiration for the submission file can be found `here <https://htcondor.readthedocs.io/en/latest/users-manual/quick-start-guide.html>`_.

**2. Running the script**
-------------------------

The arguments described above can be passed to the script through the following command:

.. code-block:: shell
   
   pygwb_dag {your-dag-file.dag} --subfile {full_path_to_subfile} --jobfile {full_path_to_jobfile} --parent_dir {full_path_to_parent_dir} --param_file {full_path_to_param_file}

.. note::

  If the ``dag`` name was not specified when calling ``pygwb_dag`` in the previous step, the default name ``dag_name.dag`` is used.

The ``dag`` file is now created in the ``{full_path_to_parent_dir}/output`` folder. To submit the job to condor and actually run all the jobs, 
navigate to that folder and run the following line in the command line:

.. code-block:: shell
   
   condor_submit_dag {your-dag-file.dag}

To check the status of the jobs, one can execute the command: 

.. code-block:: shell

  condor_q

For additional information on Condor jobs, we refer the user to the Condor `documentation <https://htcondor.readthedocs.io/en/latest/>`_.

**3. Output of the script**
---------------------------

Once all the jobs submitted through Condor and the ``dag`` file finish running, the output folder should contain similar files as the ones already discussed in the ``pygwb_pipe``
tutorial `here <pipeline.html#output-of-the-script.html>`_. However, there will be many more files compared to a single run, as ``pygwb_pipe`` was run for all the jobs, and therefore produced the output for each of the jobs.
We refrain from repeating the information about the output of ``pygwb_pipe`` and refer to the previous `tutorial <pipeline.html#output-of-the-script.html>`_ for more information about the output.

**Combining runs with pygwb_combine**
=====================================

The ``pygwb_dag`` script described above runs multiple ``pygwb_pipe`` jobs on stretches of data. For each of these runs,
the usual ``pygwb_pipe`` output is produced (see `here <pipeline.html#output-of-the-script>`_ for more information on the output of the ``pygwb_pipe`` script).
However, the user is usually interested in an overall result for the whole data set. This is where ``pygwb_combine`` comes in, by allowing
the user to combine their separate results into an overall result. For example, all separate point estimate and variance spectra will be 
combined into one overall spectrum for the whole data set. More information on this procedure can be found in the `pygwb paper <https://arxiv.org/pdf/2303.15696.pdf>`_.

**1. Script parameteres**
-------------------------

The required arguments of the ``pygwb_combine`` script can be displayed through:

.. code-block:: shell

   pygwb_combine -h

This shows the following arguments with a short description:

.. code-block:: shell

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



**2. Running the script**
-------------------------

To run the script, one executes the following command:

.. code-block:: shell

  pygwb_combine --data_path {my_pygwb_output_folder} --alpha {my_spectral_index} --fref {my_fref} --param_file {my_parameter_file_path} --out_path {my_combine_folder}

Note that not all arguments listed above are required to be able to run the script.

.. warning::

  The ``--combine_coherence`` functionality is not supported when combining runs as a result of the ``pygwb_dag`` script.


**3. Output of the script**
---------------------------

As mentioned above, the output of the ``pygwb_combine`` script is one overall point estimate and variance (spectrum). The directory 
passed through the ``--out_path``
argument should contain a file that looks as follows:

.. code-block:: shell

    point_estimate_sigma_spectra_alpha_0.0_fref_25_t0-tf.npz

This file contains the combined spectra, where the notation indicates it was run with a spectral index of 0, 
reference frequency of 25 Hz, and t0 and tf would be actual numbers corresponding to the start and end time of the analysis, respectively.

The keys of this ``npz`` file are:

.. code-block:: shell

   ['point_estimate', 'sigma', 'point_estimate_spectrum', 'sigma_spectrum',
   'frequencies', 'frequency_mask', 'point_estimates_seg_UW', 'sigmas_seg_UW']
   
The value associated with the key can be accessed from the ``npz`` file through:

.. code-block:: shell

   npzfile = numpy.load("point_estimate_sigma_spectra_alpha_0.0_fref_25_t0-tf.npz")
   variable = npzfile["key"]
   
One obtains the value for the overall point estimate and its standard deviation through the ``point_estimate`` and ``sigma`` keys, respectively.
The corresponding spectra are found by using the ``point_estimate_spectrum`` and ``sigma_spectrum`` keys. The frequencies for these spectra can be retrieved
through the ``frequencies`` key. The ``frequency_mask`` key returns the notched frequencies. For more information about notching, check the demo 
`here <make_notchlist.html>`_ or the API of the notch module `here <api/pygwb.notch.html>`_.
Lastly, one can also access the unweighted, i.e., without reweighting of the spectral index, point estimates 
and their standard deviations for every segment in the analysis. These are labeled with ``_UW`` at the end of the keys.

.. tip::
  Not sure about what is exactly in the ``.npz`` file? Load in the file and print out all its `keys` as shown 
  `here <https://stackoverflow.com/questions/49219436/how-to-show-all-the-element-names-in-a-npz-file-without-having-to-load-the-compl>`_.

If the ``pygwb_pipe`` analyses were run with the delta sigma cut turned on, a file ``delta_sigma_cut_t0-tf.npz`` should be present in the output directory as well.
This file contains the following keys:

.. code-block:: shell

  ['naive_sigma_values', 'slide_sigma_values', 'delta_sigma_values',
  'badGPStimes', 'delta_sigma_times', 'ifo_1_gates', 'ifo_2_gates',
  'ifo_1_gate_pad', 'ifo_2_gate_pad']
  
The times flagged by the delta sigma cut that are excluded from the analysis can be retrieved with the ``'badGPStimes'`` key. The alphas used for the 
delta sigma cut are stored in ``'delta_sigma_alphas'`` key, the times in ``'delta_sigma_times'``, 
and the actual values of the delta sigmas in ``'delta_sigma_values'``. The delta sigma cut computes both the naive and sliding 
sigma values, which are stored in the keys ``'naive_sigma_values'`` and ``'slide_sigma_values'``.

If gating is turned on, the gated times are saved in ``'ifo_{i}_gates'`` where ``i`` denotes the first and second onterferometer used for the analysis. 
The ``'ifo_{i}_gate_pad'`` refers to the value of the parameter ``gate_tpad`` during the analysis.