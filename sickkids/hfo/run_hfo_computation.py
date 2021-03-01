from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from mne_bids import (read_raw_bids, BIDSPath,
                      get_entity_vals, get_datatypes,
                      make_report)
from mne_bids.stats import count_events

import mne
from mne import make_ad_hoc_cov

from mne_hfo import LineLengthDetector, RMSDetector
from mne_hfo.score import _compute_score_data, accuracy
from mne_hfo.sklearn import make_Xy_sklearn, DisabledCV
from mne_hfo.io import write_annotations


def compute_hfos(bids_path, deriv_root, reference: str = "monopolar", filt_band=None):
    if filt_band is None:
        filt_band = (80, 250)

    # load in raw
    extra_params = dict(preload=True)
    raw = read_raw_bids(bids_path)

    # Set up HFO detector
    kwargs = {
        'filter_band': filt_band,  # (l_freq, h_freq)
        'threshold': 3,  # Number of st. deviations
        'win_size': 100,  # Sliding window size in samples
        'overlap': 0.25,  # Fraction of window overlap [0, 1]
        'hfo_name': "ripple"
    }
    ll_detector = LineLengthDetector(**kwargs)

    # perform the detection
    ll_detector = ll_detector.fit(raw)
    ll_chs_hfo_dict = ll_detector.chs_hfos_
    ll_hfo_event_array = ll_detector.hfo_event_arr_
    ll_hfo_df = ll_detector.df_

    save_fname = ""
    save_fpath = ""
    write_annotations(ll_hfo_df, fname=save_fname,
                      intended_for=bids_path,
                      root=None)
    pass


def main():

    # define hyperparameters
    reference = "monopolar"
    l_freq = 80
    h_freq = 250

    # define bids path
    bids_root = "to/fill/in"
    subjects = get_entity_vals(bids_root, 'subject')
    sessions = get_entity_vals(bids_root, 'session')
    subjectID = subjects[0]
    sessionID = sessions[0]
    bids_paths = BIDSPath(subject=subjectID, session=sessionID,
                          datatype="ieeg",
                          suffix="ieeg",
                          extension=".vhdr", root=bids_root)
    fpaths = bids_paths.match()
    dataset_path = fpaths[0]

    # define derivatives path
    derivatives_path = Path(dataset_path) / "derivatives"

    # run HFO computation
    compute_hfos(dataset_path, derivatives_path, reference=reference, filt_band=(l_freq, h_freq))
