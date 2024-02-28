.. _installation:

============
Installation
============

.. _installing-pygwb:

Installing from source
======================

These are instructions to install :code:`pygwb`, which runs on Python :math:`\ge3.8`.

Currently, there are 2 stable released versions (see the `pygwb PyPi page <https://pypi.org/project/pygwb/>`_ for more details):

.. code-block:: console

   1.0.0
   1.3.0

which may be installed using :code:`pip`:

.. code-block:: console

   $ pip install pygwb==[version]

Otherwise, you may install the cloned repository directly. If you already have an existing Python environment, you can simply clone the code and install in any of the usual ways.

.. tabs::

  .. tab:: pypi

    .. code-block:: console

      $ git clone git@git.ligo.org:pygwb/pygwb.git
      $ pip install .

  .. tab:: setup.py

    .. code-block:: console

      $ git clone git@git.ligo.org:pygwb/pygwb.git
      $ python setup.py install

You may also wish to install in "develop" mode.

.. tabs::

  .. tab:: pypi

    .. code-block:: console

      $ git clone git@git.ligo.org:pygwb/pygwb.git
      $ pip install -e .

  .. tab:: setup.py

    .. code-block:: console

      $ git clone git@git.ligo.org:pygwb/pygwb.git
      $ python setup.py develop

In develop mode, a symbolic link is made between the source directory and the environment site packages.
This means that any changes to the source are immediately propagated to the environment.

.. _creating-environment:

Creating a python environment
=============================

We recommend working with a recent version of Python.
A good reference is to use the default anaconda version.
This is currently :code:`Python 3.8` (August 2021).

.. tabs::

   .. tab:: conda

      :code:`conda` is a recommended package manager which allows you to manage
      installation and maintenance of various packages in environments. For
      help getting started, see the `LSCSoft documentation <https://lscsoft.docs.ligo.org/conda/>`_.

      For detailed help on creating and managing environments see `these help pages
      <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.
      Here is an example of creating and activating an environment named pygwb

      .. code-block:: console

         $ conda create -n pygwb python=3.8
         $ conda activate pygwb

   .. tab:: virtualenv

      :code:`virtualenv` is a similar tool to conda. To obtain an environment, run

      .. code-block:: console

         $ virtualenv --python=/usr/bin/python3.8 $HOME/virtualenvs/pygwb
         $ source virtualenvs/pygwb/bin/activate


   .. tab:: CVMFS

      To source a :code:`Python 3.8` installation on the LDG using CVMFS, run the
      commands

      .. code-block:: console

         $ source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
         $ conda activate igwn-py38

      Documentation for this conda setup can be found here: https://computing.docs.ligo.org/conda/.

.. _installing-python:

Installing Python
=================

Most computers/clusters have a system-installed Python version. You may choose
to use this, but here we describe an alternative. In particular, how to install
the `anaconda distribution Python package
<https://www.anaconda.com/download/#linux>`_. Firstly, download the install
file. You can do this from the link above, or run the command

.. code-block:: console

   $ wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh

This will download an installer for Python 3.8. For other versions check
the `anaconda page <https://www.anaconda.com/download/#linux>`_.
Then, `run the command
<https://conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_

.. code-block:: console

   $ bash Anaconda3-2021.05-Linux-x86_64.sh

and follow the prompts on the install screen.  After this process, you should
have a directory :code:`~/anaconda3` in your home directory. This contains your
Python installation. In particular, if you run the command

.. code-block:: console

   $ which python
   /home/users/USER/anaconda3/bin/python

The output here (with a suitable replacement of the path) indicates that you
are using the anaconda install of Python. If instead, the output says something
like :code:`/usr/bin/python`, then this is not the anaconda installation, but
instead the system Python.

If you are finding that you have run the above steps, but :code:`python` is
not pointing to your anaconda install, make sure that (a) you have appended a
line like this to your :code:`.bashrc` file

.. code-block:: console

   export PATH="${HOME}/anaconda3/bin:$PATH"

and (b) that you have restarted bash for this line to take effect (i.e., run
:code:`$ bash`).

.. note::

    Using your own installation of Python has several advantages: it's generally
    easier to debug, avoids conflicts with other packages, and if you end up
    with a broken installation you can just delete the directory and start
    again.
