=========
Tutorials
=========

Below, we provide a series of tutorial which highlight the main features of the ``pygwb`` package.
This package is constituted of several modules, each with different functionalities. These can be 
combined into a pipeline, which takes the user from gravitational-wave data to estimators of the 
gravitational-wave background (GWB). For more details on the methodology of GWB searches, we refer
the reader to the `pygwb paper <https://arxiv.org/pdf/2303.15696.pdf>`_. 

The ``pygwb`` package comes with a default pipeline, ``pygwb_pipe``, which combines the different modules
of the package. However, one of the assets of the code is its high level of modularity. Hence, users should
feel free to assemble a pipeline that addresses their needs. A quickstart manual of the default ``pygwb_pipe``
pipeline is provided below.

.. raw:: html

   <a href="pipeline.html"><button style="background-color:#307FC1;border-color:#307FC1;color:white;width: 220.0px; height: 50.0px;border-radius: 8px;margin-bottom: 10px;display:block;margin: 0 auto">Getting started with pygwb</button></a>
   
   <br />

When running ``pygwb`` on long data sets, it can be more convenient to split the large amount of data into smaller chunks, and run the analysis on those individually. This functionality is supported within
``pygwb`` through the inclusion of two additional scripts: ``pygwb_dag`` and ``pygwb_combine``. For more information, check out the tutorial below.

.. raw:: html

   <a href="multiple_jobs.html"><button style="background-color:#307FC1;border-color:#307FC1;color:white;width: 220.0px; height: 50.0px;border-radius: 8px;margin-bottom: 10px;display:block;margin: 0 auto">Run pygwb on multiple jobs</button></a>
   <br />


The ``pygwb`` package also comes with a statistical checks module, which provides a way to visualize the results of an analysis runs. Through a series of plots, it offers the possibility
to check the results for statistical consistency. To learn how to run a series of statistical checks, check out the tutorial below.

.. raw:: html

   <a href="stat_checks.html"><button style="background-color:#307FC1;border-color:#307FC1;color:white;width: 220.0px; height: 50.0px;border-radius: 8px;margin-bottom: 10px;display:block;margin: 0 auto">Run statistical checks</button></a>
   <br />

The different scripts above are conveniently grouped together into a workflow, which executes one script after the other. For more information on the workflow, we refer the user to the tutorial 
below. 

.. raw:: html

   <a href="workflow.html"><button style="background-color:#307FC1;border-color:#307FC1;color:white;width: 220.0px; height: 50.0px;border-radius: 8px;margin-bottom: 10px;display:block;margin: 0 auto">Learn more about the workflow</button></a>
   <br />

In addition, the ``pygwb`` suite features a parameter estimation module, which relies on the ``bilby`` `package <https://lscsoft.docs.ligo.org/bilby/>`_.
Using Bayesian inference, the user can run parameter estimation on the output of a ``pygwb`` run to constrain different parameters of a given model. More on parameter estimation and how to 
run it in ``pygwb`` below.

.. raw:: html

   <a href="pe.html"><button style="background-color:#307FC1;border-color:#307FC1;color:white;width: 220.0px; height: 50.0px;border-radius: 8px;margin-bottom: 10px;display:block;margin: 0 auto">Run parameter estimation</button></a>
   <br />

The ``pygwb`` package contains a data simulation module, which can be used to simulate a stochastic gravitational-wave background (GWB) 
given by a specific power spectral density (PSD) or as the superposition of individual compact binary coalescences (CBCs). To learn how
to use the simulator module, check out the tutorial below.

.. raw:: html

   <a href="simulator.html"><button style="background-color:#307FC1;border-color:#307FC1;color:white;width: 220.0px; height: 50.0px;border-radius: 8px;margin-bottom: 10px;display:block;margin: 0 auto">Simulate your own data</button></a>

.. toctree::
   :maxdepth: 1
   :hidden:
      
   pipeline
   multiple_jobs
   stat_checks
   workflow
   pe
   simulator