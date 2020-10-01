
EZTRACK
=======

Master repo with eztrack's UI, data manager, and analysis backend.


.. image:: https://circleci.com/gh/adam2392/eztrack.svg?style=svg&circle-token=be3280d393039eac5067ac529b59241a235a2d4d
   :target: https://circleci.com/gh/adam2392/eztrack
   :alt: CircleCI

.. image:: https://codecov.io/gh/adam2392/eztrack/branch/master/graph/badge.svg?token=UBQMyCETcz
  :target: https://codecov.io/gh/adam2392/eztrack

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Code style: black


Intended Users / Usage
----------------------

EZTrack team. Epilepsy researchers dealing with EEG data. Anyone with human patient EEG data. 

Data Organization
-----------------

Data should be organized in the BIDS-iEEG format:

https://github.com/bids-standard/bids-specification/blob/master/src/04-modality-specific-files/04-intracranial-electroencephalography.md

Additional data components:

.. code-block::

   1. electrode_layout.xlsx 
       A layout of electrodes by contact number denoting white matter (WM), outside brain (OUT), csf (CSF), ventricle (ventricle), or other bad contacts.

   2. clinical_metadata.xlsx     
       A database column layout of subject identifiers and their metadata.


Installation Guide
==================

Setup environment from source

.. code-block::

   pipenv install --dev

   pipenv install -e /Users/adam2392/Documents/eztrack

   pipenv install https://api.github.com/repos/mne-tools/mne-python/zipball/master
   pipenv install https://api.github.com/repos/mne-tools/mne-bids/zipball/master

Setup Jupyter Kernel To Test
============================

You need to install ipykernel to expose your conda environment to jupyter notebooks.

.. code-block::

   pipenv run python -m ipykernel install --name sickkids --user
   # now you can run jupyter lab and select a kernel
   jupyter lab