sickkids Dataset
----------------
* Version=v1.0.0

For analysis of sickkids dataset
This repo will be used as a light-weight script to be used to convert
files from the SickKids dataset into the BIDS-compliant layout with
BrainVision data format (`.vhdr`, `.vmrk`, `.eeg`).

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Code style: black


Data Organization
-----------------

Data should be organized in the BIDS-iEEG format:

https://github.com/bids-standard/bids-specification/blob/master/src/04-modality-specific-files/04-intracranial-electroencephalography.md

In order to run the script, minimally, the file needs certain BIDS entities defined:

- subject (user defined)
- session (user defined)
- task (`ictal` or `interictal`)
- acquisition (`eeg`, `ecog`, `seeg`)
- run (`01` starts)
- datatype (`eeg`, or `ieeg`)

Additional data components:

.. code-block::

   1. data_collection_irb_eztrack_v4.xlsx
       A layout of electrodes by contact number denoting white matter (WM), outside brain (OUT), csf (CSF), ventricle (ventricle), or other bad contacts.

   2. <subject_ID>.mat
       A mat file for each subject recordings and some metadata. Please see README of dataset to fully understand how to use this.


Installation Guide
==================
You can set this up via `pipenv`. First install `pipenv` using
your favorite package manager (either with pip install, or conda install).

Then run:

.. code-block::

    # install all libs + development libraries
    pipenv install --dev

    # if dev versions of mne tools are needed
    pipenv install https://api.github.com/repos/mne-tools/mne-bids/zipball/master --dev
    pipenv install https://api.github.com/repos/mne-tools/mne-python/zipball/master --dev

If you are updating packages, make sure that you discuss updating the ``Pipfile``.
Then generate a new lock file by running:

.. code-block::

    # generates a new lock file
    pipenv lock

    # if there is a pre-stable release dependency
    pipenv lock --pre


Setup Jupyter Kernel To Test
============================

You need to install ipykernel to expose your conda environment to jupyter notebooks.

.. code-block::

   pipenv run python -m ipykernel install --name sickkids --user
   # now you can run jupyter lab and select a kernel
   jupyter lab


Tests
-----
How to run tests


    pipenv install --dev
    pytest ./tests

### Contribution guidelines ###

Please submit a pull request updating the code and unit test.
