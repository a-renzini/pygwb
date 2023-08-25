===============================================
Running statistical checks with ``pygwb_stats``
===============================================

The statistical checks module provides a way to visualize the results of an analysis runs. Through a series of plots, it offers the possibility
to check the results for statistical consistency. The module itself is detailed in the `API documentation <api/pygwb.statistical_checks.html>`_. Here,
we provide some information on how to run ``pygwb_stats``, a script that calls the statistical checks module and generates all the relevant plots.

To obtain some information about the script, one can execute the following command:

.. code-block:: shell

    pygwb_stats --help

This will output the optional arguments that can be passed to the script:

.. code-block:: shell

  -c COMBINE_FILE_PATH, --combine_file_path COMBINE_FILE_PATH
                        combined file containing spectra
  -dsc DSC_FILE_PATH, --dsc_file_path DSC_FILE_PATH
                        delta sigma cut file containing sigmas and more
  -pd PLOT_DIR, --plot_dir PLOT_DIR
                        Directory where plots should be saved
  -pf PARAM_FILE, --param_file PARAM_FILE
                        Parameter file used during analysis
  -fs FONT_SIZE, --font_size FONT_SIZE
                        Primary label font size
  -fcoh COHERENCE_FILE_PATH, --coherence_file_path COHERENCE_FILE_PATH
                        Path to coherence file. If passed, automatically triggers the plot coherences option.
  -t TAG, --tag TAG     Tag to use when saving files
  -co CONVENTION, --convention CONVENTION
                        Overall convention to use in plots

Although a detailed discussion of the interpretation of the plots is delayed to the `demo section <run_statistical_checks.html>`_, 
we provide some additional information about the various parameters shown above. 

After running the ``pygwb_combine`` script (as explained `here <multiple_jobs.html>`_), the cript will have produced a file that looks similar to

.. code-block:: shell

    point_estimate_sigma_spectra_alpha_0.0_fref_25_t0-tf.npz

This file contains the combined spectra, where the notation indicates it was run with a spectral index of 0, 
reference frequency of 25 Hz, and t0 and tf would be actual numbers corresponding to the start and end time of the analysis, respectively.
This file should be passed to the ``pygwb_stats`` through the ``-c`` argument.

If the analysis was run with the delta sigma cut turned on, a file ``delta_sigma_cut_t0-tf`` should be present in the output directory as well. 
This can be passed through the ``-dsc`` argument.

The directory where the plots should be saved is passed through ``-pd``, whereas the parameter file that was used during the ``pygwb`` analysis, i.e.,
the one passed to ``pygwb_pipe``, should be passed through ``-pf``. The label font size can be changed, by using the ``-fs`` option. Finally, the coherence file,
if computed during the ``pygwb_pipe`` run, can also be passed via ``-fcoh``, which will create a series of additional plots related to the coherence.

.. seealso::

    More information on the statistical checks can be found `here <api/pygwb.statistical_checks.html>`_.

.. tip::

    Make sure to check out the demo about the interpretation of the statistical checks results `here <run_statistical_checks.html>`_.