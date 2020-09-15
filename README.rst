
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

EZTrack is intended to be a lightweight wrapper for easily analyzing large batches of patients with EEG data.

.. code-block::

   numpy
   scipy
   scikit-learn
   pandas
   mne
   mne-bids
   pybv
   pybids
   joblib
   matplotlib
   seaborn
   natsort
   tqdm


Setup environment from source

.. code-block::

   make inplace
   # dev versions of mne-python, mne-bids
   pip install --upgrade --no-deps https://api.github.com/repos/mne-tools/mne-python/zipball/master
   pip install --upgrade https://api.github.com/repos/mne-tools/mne-bids/zipball/master

   pipenv install https://api.github.com/repos/mne-tools/mne-python/zipball/master
   pipenv install https://api.github.com/repos/mne-tools/mne-bids/zipball/master


Setup environment directly via Pipenv recipe:

Note that our repository currently depends a bit on MNE-Python and MNE-BIDS development versions, which isn't maintained in the 
conda recipe file. 

.. code-block::

    pipenv install

   conda env create -f ./environment.yml --name=eztrack
   # dev versions of mne-python, mne-bids
   pipenv install --upgrade --no-deps https://api.github.com/repos/mne-tools/mne-python/zipball/master
   pipenv install --upgrade https://api.github.com/repos/mne-tools/mne-bids/zipball/master


Install from Github
-------------------

To install, run this command in your repo:

.. code-block::

   git clone https://github.com/adam2392/eztrack
   python setup.py install


or 

.. code-block::

   pip install https://api.github.com/repos/adam2392/eztrack/zipball/master


Documentation
=============

.. code-block::

   conda install sphinx sphinx-gallery sphinx_bootstrap_theme numpydoc sphinxcontrib-restbuilder
   sphinx-quickstart
   make build_doc

    pipenv install ipykernel sphinx sphinx-gallery sphinx_bootstrap_theme numpydoc sphinxcontrib-restbuilder black pytest pytest-cov coverage codespell pydocstyle --dev


Setup Jupyter Kernel To Test
============================

You need to install ipykernel to expose your conda environment to jupyter notebooks.

.. code-block::

   conda install ipykernel
   python -m ipykernel install --name eztrack --user
   # now you can run jupyter lab and select a kernel
   jupyter lab 


Testing
=======

Install testing and formatting libs:

.. code-block::

   conda install black pytest pytest-cov coverage codespell pydocstyle
   pip install coverage-badge anybadge mypy


Run tests

.. code-block::

   black eztrack/*
   black tests/*
   pylint ./eztrack/
   anybadge --value=6.0 --file=pylint.svg pylint
   pytest --cov-config=.coveragerc --cov=./eztrack/ tests/
   pytest --cov-config=.coveragerc --cov=./eztrack/ tests/ > docs/tests/test_docs.txt
   coverage-badge -f -o coverage.svg
