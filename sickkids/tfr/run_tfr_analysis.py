from pathlib import Path

import numpy as np
from mne import make_fixed_length_events, Epochs
from mne.time_frequency import tfr_multitaper
from mne.time_frequency.tfr import tfr_raw
from mne.utils import warn
from mne_bids import BIDSPath, get_entity_vals

from sickkids.read import load_data

# Define EEG bands
eeg_bands = {'delta': (1, 4),
             'theta': (4, 8),
             'alpha': (8, 12),
             'beta': (12, 30),
             'gamma': (30, 90),
             'highgamma': (90, 200)}


def run_analysis(
        bids_path, freq_band, reference="monopolar", resample_sfreq=None,
        deriv_path=None, figures_path=None,
        verbose=True, overwrite=False, plot_heatmap=True,
        plot_raw=True, **model_params
):
    subject = bids_path.subject

    # get the root derivative path
    deriv_chain = Path(reference) / f"sub-{subject}"
    figures_path = figures_path / deriv_chain
    deriv_path = deriv_path / deriv_chain

    deriv_path.mkdir(exist_ok=True, parents=True)
    # check if we have original dataset
    source_basename = bids_path.copy().update(extension=None, suffix=None).basename
    deriv_fpaths = deriv_path.glob(f'{source_basename}*desc-{freq_band}*.h5')
    if not overwrite and len(list(deriv_fpaths)) > 0:
        warn(f'Not overwrite and the derivative file path for {source_basename} already exists. '
             f'Skipping...')
        return

    deriv_fpath = deriv_path / f'{source_basename}_desc-{freq_band}_ieeg.h5'

    # load in raw data
    raw = load_data(bids_path, resample_sfreq, deriv_root=figures_path,
                    plot_raw=plot_raw, verbose=verbose)

    # use the same basename to save the data
    raw.drop_channels(raw.info['bads'])
    print(f"Analyzing {raw} with {len(raw.ch_names)} channels.")

    # create epochs data structure from Raw, using equally spaced events
    # epochs = make_fixed_length_epochs(raw, duration=1, )
    print(f'Trying to compute tfr for {freq_band} band.')
    print(raw)
    # events = make_fixed_length_events(raw, id=1, duration=2.0, overlap=0.5)
    # epochs = Epochs(raw, events=events, event_id=1, tmin=0, tmax=2.0,
    #                 baseline=None)
    # print(epochs)
    # print(epochs.get_data().shape)

    # run tfr
    freq_endpoints = eeg_bands[freq_band]
    freqs = np.linspace(*freq_endpoints, num=10)
    # power = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs / 2,
    #                        average=False,
    #                        n_jobs=-1, return_itc=False)

    power = tfr_raw(raw, freqs=freqs, n_cycles=freqs/2, n_jobs=-1,
                    method='morlet', decim=10,
                    statistic=None)

    power.comment = freq_band
    print(power)
    print(power.data.shape)
    # raise Exception('hi')
    power.save(deriv_fpath, overwrite=overwrite)


def main():
    # bids root to write BIDS data to
    # the root of the BIDS dataset
    root = Path("/Users/adam2392/OneDrive - Johns Hopkins/sickkids/")
    # root = Path("/Users/adam2392/Dropbox/resection_tvb/")
    deriv_root = root / 'derivatives' / 'tfr'
    figures_path = deriv_root / "figures"

    # define BIDS entities
    SUBJECTS = [
        # 'id008gc', 'id013pg',
        "E1",
        # 'E3',
        # 'E4',
        # 'E5',
        # 'E6',
        # 'E7'
    ]

    # pre, Sz, Extraoperative, post
    # task = "pre"
    acquisition = "ecog"
    # acquisition = 'seeg'
    datatype = "ieeg"
    extension = ".vhdr"
    session = "postresection"  # only one session

    # analysis parameters
    reference = 'average'
    order = 1
    sfreq = None
    overwrite = True
    SESSIONS = [
        'extraoperative',
        'preresection',
        'intraresection',
        'postresection'
    ]

    # get the runs for this subject
    all_subjects = get_entity_vals(root, "subject")
    for subject in all_subjects:
        if subject not in SUBJECTS:
            continue
        ignore_subs = [sub for sub in all_subjects if sub != subject]

        # get all sessions
        sessions = get_entity_vals(root, 'session', ignore_subjects=ignore_subs)
        for session in sessions:
            if session not in SESSIONS:
                continue
            ignore_sessions = [ses for ses in sessions if ses != session]
            ignore_set = {
                'ignore_subjects': ignore_subs,
                'ignore_sessions': ignore_sessions,
            }
            print(f'Ignoring these sets: {ignore_set}')
            all_tasks = get_entity_vals(root, "task", **ignore_set)
            tasks = all_tasks

            for task in tasks:
                print(f"Analyzing {task} task.")
                ignore_tasks = [tsk for tsk in all_tasks if tsk != task]
                ignore_set['ignore_tasks'] = ignore_tasks
                runs = get_entity_vals(
                    root, 'run', **ignore_set
                )
                print(f'Found {runs} runs for {task} task.')

                for idx, run in enumerate(runs):
                    # create path for the dataset
                    bids_path = BIDSPath(
                        subject=subject,
                        session=session,
                        task=task,
                        run=run,
                        datatype=datatype,
                        acquisition=acquisition,
                        suffix=datatype,
                        root=root,
                        extension=extension,
                    )
                    print(f"Analyzing {bids_path}")

                    for freq_band in eeg_bands.keys():
                        run_analysis(bids_path, freq_band=freq_band,
                                     reference=reference,
                                     resample_sfreq=sfreq,
                                     deriv_path=deriv_root,
                                     figures_path=figures_path,
                                     plot_heatmap=True, plot_raw=True,
                                     overwrite=overwrite, order=order)


if __name__ == "__main__":
    main()
