======================
The ``pygwb`` workflow
======================

In practice, one will likely run the ``pygwb_pipe`` script on multiple jobs through the ``pygwb_dag`` script, and combine the output with 
``pygwb_combine``. After this, ``pygwb_stats`` can be run to obtain the statistical checks diagnostic plots. To facilitate the usage
of the different scripts and increase the user-friendliness, ``pygwb`` comes with an additional script that combines the scipts above into a worklfow,
executing one script after the other.

.. warning::

    The ``pygwb`` workflow is tailored to the LIGO-Virgo-KAGRA data stream. Therefore, it is possible that not all
    data outside the standard LIGO-Virgo-KAGRA format are supported by the current version of the workflow.

.. tip::
    
    Still want to run ``pygwb`` on your data without relying on the workflow? Use the `tutorial <pipeline.html>`_ about ``pygwb_pipe`` to 
    get started.

**1. Script parameters**
========================

To show the optional arguments of the script, one can run:

.. code-block:: shell

    pygwb_create_isotropic_workflow --help

This will display the following arguments of the script, with a brief description of the parameters:

.. code-block:: shell

    -h, --help            show this help message and exit
    -v, --verbose         increase verbose output
    --debug               increase verbose output to debug
    -V, --version         show program's version number and exit
    --basedir BASEDIR     Build directory for condor logging
    --configfile CONFIGFILE
                        config file with workflow parameters
    --config-overrides CONFIG_OVERRIDES [CONFIG_OVERRIDES ...]
                        config overrides listed in the form SECTION:KEY:VALUE
    --submit              Submit workflow automatically
    --run_locally         Run job on local universe


**2. Running the script**
=========================

To run the workflow, one executes the command:

.. code-block:: shell

    pygwb_create_isotropic_workflow --debug --basedir ./ --configfile ./my_config_file.ini --submit

The ``--debug`` argument increases the amount of verbose printed out by the script, whereas the ``--submit`` tells the script to automatically submit the different jobs to Condor.

.. tip::

    Need a reminder about multiple jobs being submitted to the cluster through Condor? Check out the tutorial on multiple jobs `here <multiple_jobs.html>`_.


In addition, a configuration file needs to be passed to the script through the ``--configfile`` argument. An example of a possible configuration file for the workflow is 
provided in the ``pygwb`` repo `here <https://github.com/a-renzini/pygwb/blob/master/pygwb_pipe/workflow_config.ini>`_. The configuration file is divided into different
categories, which we discuss step by step below.

A first part of the file contains general information to initialize the workflow, such as the interferometers for which to retrieve data, the duration of the jobs, as well as
the start and end time for which to analyze data.

.. code-block:: shell

    [general]
    accounting_group = ligo.dev.o4.sgwb.isotropic.stochastic
    ifos = H1 L1
    plot_segment_results = False
    max_job_dur = 5000
    min_job_dur = 900
    t0 = 1368921618
    tf = 1374883218

As mentioned earlier, the workflow combines different scripts of the ``pygwb`` package. These are passed through the ``[executables]`` section of the configuration file:

.. code-block:: shell

    [executables]
    pygwb_pipe = pygwb_pipe
    pygwb_combine = pygwb_combine
    pygwb_stats = pygwb_stats
    pygwb_html = pygwb_html

When dealing with real detector data, some tags can be used to define which level of "cleanliness" is required in the data. This is specified in the ``[data_quality]`` 
section of the file:

.. code-block:: shell

    [data_quality]
    science_segment = DMT-ANALYSIS_READY
    veto_definer = /home/arianna.renzini/public_html/ER15_pygwb_run/old_setup/H1L1-HOFT_C00_O4_CBC.xml

An additional section, ``[pygwb_pipe]``, contains all the parameters needed to run the ``pygwb_pipe`` script. We refrain from giving a detailed overview of all these parameters, and 
refer the user to the dedicated tutorial `here <pipeline.html>`_, as well as the ``pygwb.parameters`` API `page <api/pygwb.parameters.html>`_ for further information about these parameters.

The last part of the workflow takes care of combining the output of all the jobs, and runs statistical checks on the combined output. The results are then displayed on an ``html`` webpage.
Parameters related to the last part of the workflow are passed through the following lines in the configuration file:

.. code-block:: shell

    [pygwb_combine]
    alpha = ${pygwb_pipe:alpha}
    fref = ${pygwb_pipe:fref}
    combine_coherence = True

    [pygwb_stats]

    [pygwb_html]

.. seealso::

    For more information about the ``pygwb_combine`` script, see the tutorial `here <multiple_jobs.html>`_. Additional details about the ``pygwb_stats`` script can be found in 
    the dedicated `tutorial <stat_checks.html>`_, with a plot by plot discussion `here <run_statistical_checks.html>`_.

**3. Output of the script**
===========================

As mentioned in the introduction of this tutorial, the workflow combines the different ``pygwb`` scripts. Therefore, the output of the
workflow will be similar to that of the
individual scripts. We refrain from going over the different outputs again, but refer the user to the dedicated tutorials for more 
information (e.g. `pygwb_pipe <pipeline.html#output-of-the-script>`_, 
`pygwb_combine <multiple_jobs.html#id8>`_, `pygwb_stats <stat_checks.html#output-of-the-script>`_). 

Futhermore, we note the additional feature of the workflow which displays all 
results of the run in an ``html`` page. The workflow script creates several directories for the output of the workflow 
and generates the files that will be used for the submission of the ``dag`` file on a cluster. 
In the designated directory, given by ``--basedir BASEDIR`` parameter, the following files and directories can be found:

.. code-block:: shell

   about  condor  config.ini  index.html  output  pygwb_cache.txt
   
The ``about`` directory contains information about the analysis run, as well as the input passed to the workflow and is mainly used to set up the ``html`` pages. 
The ``condor`` directory stores all relevant files needed for the ``dag`` submission to the cluster. Every individual job 
will have an ``output``, ``submit``, ``error`` and a ``log`` file starting with ``pygwb_pipe_{t0}_{length}``.
The dag file itself is labeled ``pygwb_dagman.submit``, and can be submitted through ``condor_submit_dag pygwb_dagman.submit``. The ``error`` and ``log`` files can help finding 
errors when running the ``dag`` file. See the Condor documentation `here <https://htcondor.org/documentation/htcondor.html>`_ 
for more information. The submit files for the ``pygwb_combine``, ``pygwb_stats`` and ``pygwb_html`` scripts are also present in this folder.

The ``config.ini`` is a copy of the configuration file given to the workflow script through ``--configfile CONFIGFILE``.

The ``output`` directory contains a subdirectory for every individual job. In each subdirectory, the output files mentioned
`here <pipeline.html#output-of-the-script>`_ can be found.
In addition, there is also the subdirectory ``combined_results`` which contains the output of the ``pygwb_combine`` script, 
see `here <multiple_jobs.html#id8>`_ for more information. The last subdirectory ``segment_lists`` contains a data file with 
the start times, end times, and lengths of the jobs analyzed by the workflow.
