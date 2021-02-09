from pathlib import Path

import mne
import numpy as np
import pandas as pd
from eztrack.io import read_derivative_npy
from eztrack.io.base import _add_desc_to_bids_fname, DERIVATIVETYPES
from eztrack.preprocess import preprocess_ieeg
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids

import sickkids.io


def run_plot_raw(bids_path, resample_sfreq, figures_path, verbose=None):
    # load in the data
    raw = read_raw_bids(bids_path)

    if resample_sfreq:
        # perform resampling
        raw = raw.resample(resample_sfreq, n_jobs=-1)

    raw = raw.pick_types(seeg=True, ecog=True, eeg=True, misc=False, exclude=[])
    sickkids.io.load_data()

    # pre-process the data using preprocess pipeline
    print("Power Line frequency is : ", raw.info["line_freq"])
    l_freq = 0.5
    h_freq = 300
    raw = preprocess_ieeg(raw, l_freq=l_freq, h_freq=h_freq, verbose=verbose)

    # plot raw data
    figures_path.mkdir(exist_ok=True, parents=True)
    fig_basename = bids_path.copy().update(extension=".svg", check=False).basename
    scale = 200e-6
    fig = raw.plot(
        decim=40,
        duration=20,
        scalings={"ecog": scale, "seeg": scale},
        n_channels=len(raw.ch_names),
        clipping=None,
    )
    fig.savefig(figures_path / fig_basename)
    return raw


def run_plot_heatmap(deriv_path, figures_path):
    deriv = read_derivative_npy(deriv_path)

    # read in channel types
    bids_path = BIDSPath(
        **deriv.info.source_entities,
        datatype=deriv.info["datatype"],
        root=deriv.bids_root,
    )
    bids_path.update(suffix="channels", extension=".tsv")

    # read in sidecar channels.tsv
    channels_pd = pd.read_csv(bids_path.fpath, sep="\t")
    description_chs = pd.Series(
        channels_pd.description.values, index=channels_pd.name
    ).to_dict()
    print(description_chs)
    resected_chs = [
        ch for ch, description in description_chs.items() if description == "resected"
    ]
    print(f"Resected channels are {resected_chs}")

    # set title name as the filename
    title = Path(deriv_path).stem

    figure_fpath = Path(figures_path) / Path(deriv_path).with_suffix(".pdf").name

    # normalize
    deriv.normalize()
    cbarlabel = "Fragility"

    # run heatmap plot
    deriv.plot_heatmap(
        cmap="turbo",
        cbarlabel=cbarlabel,
        title=title,
        figure_fpath=figure_fpath,
        soz_chs=resected_chs,
    )
    print(f"Saved figure to {figure_fpath}")


def plot_tvb():
    reference = "average"

    root = Path("/Users/adam2392/Dropbox/resection_tvb/")
    deriv_root = root / "derivatives" / "radius1.25" / "fragility" / reference
    source_root = root / "sourcedata" / "epilepsy"
    figures_path = root / "figures"

    SUBJECTS = ["id008gc", "id013pg"]
    desc = "perturbmatrix"
    session = "extraoperative"

    sub_ez_regions = {
        "id008gc": [
            "Right-Amygdala",
            "Right-Hippocampus",
            #                                'ctx-rh-medialorbitofrontal'
        ],
        "id013pg": ["ctx-rh-fusiform"],
    }
    sub_pz_regions = {
        "id008gc": [
            "ctx-rh-inferiortemporal",
            "ctx-rh-temporalpole",
        ],
        "id013pg": [
            "Right-Hippocampus",
            "Right-Amygdala",
            #                 'ctx-rh-middltemporal',
            "ctx-rh-inferiortemporal",
            #                 'ctx-rh-entorhinal'
        ],
    }

    for subject in SUBJECTS:
        # get all derivative files
        deriv_path = deriv_root / f"sub-{subject}"
        source_path = source_root / f"sub-{subject}"
        fpaths = list(deriv_path.rglob(f"*desc-{desc}*.json"))

        # load the location of each electrodes
        electrodes_path = BIDSPath(
            root=root,
            subject=subject,
            session=session,
            suffix="electrodes",
            datatype="ieeg",
            acquisition="seeg",
            space="mri",
            extension=".tsv",
        )
        elec_df = pd.read_csv(electrodes_path, delimiter="\t")

        # get the DK atlas locations to estimate the resected channels
        elec_names = elec_df.name.tolist()
        elec_regions = elec_df["desikan-killiany"].tolist()

        # resected channels
        ez_regions = sub_ez_regions[subject]
        pz_regions = sub_pz_regions[subject]
        resected_chs = [
            ch for idx, ch in enumerate(elec_names) if elec_regions[idx] in ez_regions
        ]
        pz_resected_chs = [
            ch for idx, ch in enumerate(elec_names) if elec_regions[idx] in pz_regions
        ]
        resected_chs.extend(pz_resected_chs)

        print(f"Found {fpaths} for subject {subject}")
        for fpath in fpaths:
            deriv = read_derivative_npy(fpath, preload=True, source_check=False)

            # normalize
            deriv.normalize()

            # read in vertical markers
            vertical_markers = {}
            events, event_id = mne.events_from_annotations(deriv)
            if "eeg sz onset" in event_id:
                _id = event_id.get("eeg sz onset")
                sz_onset = int(events[np.argwhere(events[:, -1] == _id), 0].squeeze())
                print(sz_onset)
                vertical_markers[sz_onset] = "seizure onset"
            if "eeg sz offset" in event_id:
                _id = event_id.get("eeg sz offset")
                sz_offset = int(events[np.argwhere(events[:, -1] == _id), 0].squeeze())
                vertical_markers[sz_offset] = "seizure offset"

            # plot heatmap
            figures_path.mkdir(exist_ok=True, parents=True)
            fig_basename = fpath.with_suffix(".pdf").name
            deriv.plot_heatmap(
                cmap="turbo",
                soz_chs=resected_chs,
                vertical_markers=vertical_markers,
                cbarlabel="Fragility",
                title=fig_basename,
                figure_fpath=(figures_path / fig_basename),
            )


if __name__ == "__main__":
    plot_tvb()
    exit(1)

    # the root of the BIDS dataset
    WORKSTATION = "home"

    if WORKSTATION == "home":
        # bids root to write BIDS data to
        # the root of the BIDS dataset
        root = Path("/Users/adam2392/OneDrive - Johns Hopkins/sickkids/")
        deriv_root = root / "derivatives" / "radius1.25"
        figures_root = deriv_root / "figures"
    elif WORKSTATION == "lab":
        root = Path("/home/adam2392/hdd/epilepsy_bids/")

    # define BIDS entities
    SUBJECTS = [
        "E1",
        # 'E2',
        # 'E3',
        # 'E4',
        # 'E5', 'E6'
    ]

    # pre, Sz, Extraoperative, post
    # task = "pre"
    acquisition = "ecog"
    datatype = "ieeg"
    extension = ".vhdr"
    session = "postresection"  # only one session

    # analysis parameters
    reference = "common"
    sfreq = None

    SESSIONS = ["preresection", "extraoperative", "intraoperative", "postsurgery"]

    # get the runs for this subject
    all_subjects = get_entity_vals(root, "subject")
    for subject in all_subjects:
        if subject not in SUBJECTS:
            continue
        ignore_subs = [sub for sub in all_subjects if sub != subject]

        # get all sessions
        sessions = get_entity_vals(root, "session", ignore_subjects=ignore_subs)
        for session in sessions:
            if session not in SESSIONS:
                continue
            ignore_sessions = [ses for ses in sessions if ses != session]
            ignore_set = {
                "ignore_subjects": ignore_subs,
                "ignore_sessions": ignore_sessions,
            }
            print(f"Ignoring these sets: {ignore_set}")
            all_tasks = get_entity_vals(root, "task", **ignore_set)
            tasks = all_tasks

            for task in tasks:
                print(f"Analyzing {task} task.")
                ignore_tasks = [tsk for tsk in all_tasks if tsk != task]
                ignore_set["ignore_tasks"] = ignore_tasks
                runs = get_entity_vals(root, "run", **ignore_set)
                print(f"Found {runs} runs for {task} task.")

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

                    deriv_basename = _add_desc_to_bids_fname(
                        bids_path.basename,
                        description=DERIVATIVETYPES.COLPERTURB_MATRIX.value,
                    )
                    deriv_chain = (
                        Path("originalsampling")
                        / "fragility"
                        / reference
                        / f"sub-{subject}"
                    )
                    deriv_path = deriv_root / deriv_chain / deriv_basename
                    figures_path = figures_root / deriv_chain
                    run_plot_heatmap(deriv_path=deriv_path, figures_path=figures_path)

                    # run plot raw data
                    figures_path = (
                        figures_root
                        / Path("originalsampling")
                        / "raw"
                        / reference
                        / f"sub-{subject}"
                    )
                    run_plot_raw(
                        bids_path=bids_path,
                        resample_sfreq=None,
                        figures_path=figures_path,
                    )
