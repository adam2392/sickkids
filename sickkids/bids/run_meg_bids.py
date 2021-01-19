import mne
import collections
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
from mne.channels import make_dig_montage
from mne.io import read_raw_edf
from mne_bids import write_raw_bids, read_raw_bids, get_anonymization_daysback
from mne_bids.path import (BIDSPath, _find_matching_sidecar)

from sickkids.bids.io import MatReader
from sickkids.bids.update import append_original_fname_to_scans
from sickkids.bids.utils import (BadChannelDescription, _update_sidecar_tsv_byname,
                                 _check_bids_parameters, _look_for_bad_channels, _channel_text_scrub)


def write_meg_to_bids(source_fpath, bids_path, line_freq: int = 60):
    raw = mne.io.read_raw_ctf(source_fpath)

    if raw.info['line_freq'] is None:
        raw.info['line_freq'] = line_freq

    # convert to BIDS
    bids_path = write_raw_bids(raw, bids_path, anonymize=None)

    print(bids_path)
    raise Exception('hi')

if __name__ == "__main__":
    WORKSTATION = "home"

    if WORKSTATION == "home":
        # bids root to write BIDS data to
        bids_root = Path("/Users/adam2392/OneDrive - Johns Hopkins/sickkids/")
        source_dir = bids_root / 'sourcedata'

    elif WORKSTATION == "lab":
        bids_root = Path("/home/adam2392/hdd2/epilepsy_bids/")
        source_dir = Path("/home/adam2392/hdd2/epilepsy_bids/sourcedata")

    # define BIDS identifiers
    modality = "meg"
    task = "tbd"
    session = 'extraoperative'
    datatype = 'meg'

    subject_ids = [
        # 'E1',
        'E2',
        # 'E3',
        # 'E4',
        # 'E5',
        # 'E6'
    ]

    # regex pattern for the files is:
    for subject in subject_ids:
        source_folder = source_dir / 'meg' / subject
        for acq_date in [fdir.name for fdir in source_folder.glob('*') if fdir.is_dir()]:
            folder_paths = [f for f in (source_folder / acq_date).glob('*anon.ds') if f.is_dir()]

            for run_id, source_path in enumerate(folder_paths):
                # get the next available run
                run_id = run_id + 1

                bids_kwargs = {
                    'subject': subject,
                    'session': session,
                    'task': task,
                    'acquisition': modality,
                    'run': run_id,
                    'datatype': datatype,
                    'suffix': datatype,
                }
                print(bids_kwargs
                      )
                bids_path = BIDSPath(**bids_kwargs, root=bids_root)
                print(bids_path.fpath)

                # run main bids conversion
                write_meg_to_bids(source_path, bids_path=bids_path)
