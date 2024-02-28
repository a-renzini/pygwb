==================
Contributing guide
==================

We encourage the contributions from the whole gravitational-wave community. Below, we  provide 
instructions on how to contribute to the ``pygwb`` code and make the distinction between the
contributing guidelines for members of the LIGO-Virgo-KAGRA (LVK) collaborations and non-members.


**1. Contributing as LVK member**
=================================

The first stage in contributing to the code is to set up a development environment.
This largely follows the installation instructions with some notable changes.

The intended use for this page is for a user who wants to make an initial contribution to the code.

Firstly, welcome! Any contribution, large or small is always appreciated, even just fixing a single typo in the documentation.

If you are hoping to introduce a new feature, there are a few steps that are generally a good idea to follow:

1. Understand if some version of the new feature is already implemented/planned. Ideally this can be done by looking through existing code/issues/merge requests.
2. If your new feature is not under active development make a new issue proposing the change for feedback.
3. Follow the instructions below to implement the change and open a merge request

If the change is a bug/typo fix, you can skip the first (and possibly second) stages and go straight to an issue/merge request.

Preparing a merge request
-------------------------

:code:`pygwb` development follows the fork and merge workflow.
For a background on the method see, e.g., `here <https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow>`_.

The quick version is to

- create a personal fork, this will live under :code:`git.ligo.org/albert.einstein/pygwb`.
- clone your new fork

  .. code-block:: console

    $ git clone git@git.ligo.org:albert.einstein/pygwb.git
- install the precommits. These are executables that run every time you commit a change to verify that the changes are consistent with our style conventions.
  Many of these checks will also reformat the code to ensure the code matches the previous style.
  Some tests do not, for example, the automated spell checker will just identify issues and suggest changes.

  .. code-block::

    $ pip install pre-commit
    $ pre-commit install
- create a new branch

  .. code-block:: console

    $ git checkout -b new-feature-branch
- make any required changes on your new branch.
- run the test suite.

  .. code-block:: console

    $ pytest .

  This will run the existing unit tests and verify that new changes haven't broken existing behavior.
- write new tests. This is extremely important and probably the most intimidating step for new developers.
  There are various recommendations on how to write good unit tests online, e.g., `this SO <https://stackoverflow.com/questions/3258733/new-to-unit-testing-how-to-write-great-tests>`_.
  A good set of tests should ensure the code works as expected in all reasonable use cases and run in `O(s)`.
- run the test suite again.

  .. code-block:: console

    $ pytest .

  This will run the old tests and the new tests.
- commit the new code.

  .. code-block:: console

    $ git add MY_NEW_FILE
    $ git commit MY_NEW_FILE MY_MODIFIED_FILE -m "commit message describing change"

  At this stage, the pre-commit tests will run and any issues will be identified. If there are issues, fix them an commit again.
- once the tests pass locally and all the changes have been committed push to the remote repository

  .. code-block:: console

    $ git push

  You may be prompted to set the "upstream". This is the label applied to the remote repository.
  You can see where the remote repositories you have access to are and what the labels are by running

  .. code-block:: console

    $ git remote -v
- after you successfully push, you can open a merge request. This can be done either using the link provided or by opening `pygwb <https://git.ligo.org/pygwb/pygwb>`_ in your browser.
- follow any feedback and suggestions from the repository maintainers in order to promptly get the new feature merged!


**2. Contributing as a non-LVK member**
=======================================

As mentioned above, we encourage and appreciate contributions from the whole gravitational-wave community, both 
inside and outside the LVK collaborations. However, the instructions to open a merge request detailed above
are specific to LVK members. Nevertheless, we encourage people who want to contirbute to the 
code to reach out by opening an `issue <https://github.com/a-renzini/pygwb/issues/new>`_ on the Git repo. This will get you in touch with the ``pygwb`` developing
team, who will be able to assist you further in folding in your contribution, and making sure your contribution
is properly acknowledged.