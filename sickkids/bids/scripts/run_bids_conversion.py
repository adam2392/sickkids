import collections
import os
import random
from pathlib import Path
from typing import Union

import mne
import numpy as np
import pandas as pd
from eztrack.preprocess.bids_conversion import (
    append_original_fname_to_scans,
    _update_sidecar_tsv_byname,
)
from mne.channels import make_dig_montage
from mne.io import read_raw_edf
from mne_bids import write_raw_bids, read_raw_bids, get_anonymization_daysback
from mne_bids.path import BIDSPath, _find_matching_sidecar
from typing import Dict
import sys
sys.path.append('./')
from sickkids.bids.dataset.skids import (
    _set_ch_types,
    _set_ch_meta,
    _get_ch_sizes,
    add_data_to_participants,
)
from sickkids.bids.dataset.tvb_dataset import _set_ch_types_tvb
from sickkids.bids.io import MatReader
from sickkids.bids.utils import (
    _check_bids_parameters,
    _look_for_bad_channels,
    _channel_text_scrub,
)
from sickkids.tvb.utils import read_surf, compute_closest_vertex


def _read_mat_data_file(fpath, subject):
    reader = MatReader()

    # load mat file
    mat = reader.loadmat(fpath)
    mat_dict = mat.get(subject)
    print(mat.keys())
    print(mat_dict.keys())

    # raise Exception('hi')
    return mat_dict


def write_edf_to_bids(
    edf_fpath: Union[str, Path],
    bids_kwargs: Dict,
    bids_root: Union[str, Path],
    line_freq: int = 60,
    montage=None,
    dataset_name=None,
    source_dir=None,
        overwrite:bool=False,
) -> Dict:
    """Write EDF (.edf) files to BIDS format.

    This will convert files to BrainVision (.vhdr, .vmrk, .eeg) data format
    in the BIDS-compliant file-layout.

    The passed in BIDS keyword arguments must adhere to a certain standard.
    Namely, they should include::

        - ``subject``
        - ``session``
        - ``task``
        - ``acquisition``
        - ``run``
        - ``datatype``

    Parameters
    ----------
    edf_fpath : str | Path
    bids_kwargs : dict
    bids_root : str | Path
    line_freq : int

    Returns
    -------
    status_dict : dict
        The resulting status of the BIDS conversion.
    """
    # do check on BIDS kwarg parameters
    entities = _check_bids_parameters(bids_kwargs)

    # construct the BIDS path that the file will get written to
    bids_path = BIDSPath(**entities, root=bids_root)

    # load in EDF file
    raw = read_raw_edf(edf_fpath)

    # set line freq
    raw.info["line_freq"] = line_freq

    subject = bids_path.subject
    if dataset_name == "sickkids":
        # read the corresponding mat file path
        mat_fpath = source_dir / "subject_specific_metadata" / f"{subject}.mat"
        mat = _read_mat_data_file(mat_fpath, subject=bids_path.subject)
        fs = mat.get("fs")
        reference = mat.get("reference")
        bad = mat.get("bad")

        # set channel types - sickkids dataset
        ch_mapping = _set_ch_types(
            raw, entities.get("subject"), entities.get("session")
        )
        resected_chs = ch_mapping["resected"]

        # set impeddances
        raw = _set_ch_meta(raw)
        ch_sizes = _get_ch_sizes(raw)
    elif dataset_name == "tvb":
        raw = _set_ch_types_tvb(raw, subject)

    # make fake np.nan montage
    # if montage is None:
    #     ch_pos = {ch_name: [np.nan] * 3 for ch_name in raw.ch_names}
    #     montage = make_dig_montage(ch_pos=ch_pos, coord_frame="mri")
    # raw.set_montage(montage, on_missing="ignore")

    # channel text scrub
    raw = _channel_text_scrub(raw)

    # look for bad channels
    bad_chs = _look_for_bad_channels(raw.ch_names)
    raw.info["bads"].extend(bad_chs)

    # Get acceptable range of days back and pick random one
    daysback_min, daysback_max = get_anonymization_daysback(raw)
    daysback = random.randrange(daysback_min, daysback_max)
    anonymize = dict(daysback=1, keep_his=False)

    # write to BIDS based on path
    output_bids_path = write_raw_bids(
        raw, bids_path=bids_path,
        anonymize=anonymize, format='BrainVision',
        overwrite=overwrite, verbose=False
    )

    # add resected channels to description
    print(f"output bids path: {output_bids_path}")
    channels_tsv_fname = _find_matching_sidecar(
        output_bids_path, suffix="channels", extension=".tsv", on_error="ignore"
    )
    print(f"The channels tsv file is... {channels_tsv_fname}")
    electrodes_tsv_fname = _find_matching_sidecar(
        output_bids_path, suffix="electrodes", extension=".tsv", on_error="ignore"
    )
    print(f"The electrodes tsv file is... {electrodes_tsv_fname}")

    if dataset_name == "sickkids":
        # update which channels were resected
        for ch in resected_chs:
            _update_sidecar_tsv_byname(
                channels_tsv_fname, ch, "description", "resected"
            )

        # update grouping of channels by grid, strip, depth
        for group in ["grid", "strip", "depth"]:
            chs = ch_mapping[group]
            for ch in chs:
                _update_sidecar_tsv_byname(channels_tsv_fname, ch, "group", group)
                if electrodes_tsv_fname is not None:
                    _update_sidecar_tsv_byname(electrodes_tsv_fname, ch, "group", group)

        # update reference channel
        reference_ch = ch_mapping["reference"]
        for ch in [
            raw.ch_names[idx]
            for idx in mne.pick_types(
                raw.info, meg=False, ecog=True, seeg=True, exclude=[]
            )
        ]:
            _update_sidecar_tsv_byname(
                channels_tsv_fname, ch, "reference", reference_ch
            )

        # update size of channels
        if electrodes_tsv_fname:
            for ch, size in ch_sizes.items():
                _update_sidecar_tsv_byname(electrodes_tsv_fname, ch, "size", size)

        # TODO: update which channels were SOZ
        # update status description
        for ch, description in raw.info["bads_description"].items():
            _update_sidecar_tsv_byname(
                channels_tsv_fname, ch, "status_description", description
            )

    # output status dictionary
    status_dict = {
        "status": 1,
        "output_fname": output_bids_path.basename,
        "original_fname": os.path.basename(edf_fpath),
    }
    return status_dict


def _extraop_maps(taskname, subject):
    taskname = taskname.lower()
    if "awake" in taskname:
        return "interictalawake"
    elif "sleep" in taskname:
        return "interictalasleep"
    elif "interictal" in taskname:
        return "interictal"
    elif "sz" in taskname:
        return "ictal"
    elif "pges" in taskname:
        return "pges"
    return taskname


def convert_tvb_dataset():
    from sickkids.tvb.simulation import load_subject_connectivity
    from sickkids.bids.utils import _update_electrodes_tsv
    from sickkids.bids.dataset.tvb_dataset import add_data_to_participants

    root = Path("/Users/adam2392/Dropbox/resection_tvb")
    source_dir = root / "sourcedata"

    datatype = "ieeg"
    modality = "seeg"
    task = "ictal"
    space = "mri"
    session = "extraoperative"
    extension = ".vhdr"
    line_freq = 50

    subject_ids = [
        "id008_gc",
        "id013_pg",
    ]
    for subject in subject_ids:
        seeg_path = source_dir / "epilepsy" / subject / "seeg" / "edf"

        # read in the surface files to get the closest vertex
        tvb_path = source_dir / "epilepsy" / subject / "tvb"
        verts, normals, areas, regmap = read_surf(tvb_path, use_subcort=True)

        # read in the montage
        ch_xyz_fpath = source_dir / "epilepsy" / subject / "elec" / "seeg.txt"
        df = pd.read_csv(
            ch_xyz_fpath, delim_whitespace=True, index_col=0, header=None
        ).T
        ch_dict = df.to_dict("list")
        montage = make_dig_montage(ch_dict, coord_frame=space)

        # assign each channel to its closest anatomical vertex
        ch_names = list(ch_dict.keys())
        ch_xyz = df.to_numpy().T
        print(ch_xyz.shape, verts.shape)
        closest_idxs = compute_closest_vertex(verts, regmap, ch_xyz)
        closest_idxs = np.array(closest_idxs).astype(int)

        conn = load_subject_connectivity(
            source_file=(tvb_path / "connectivity.zip").as_posix()
        )
        # get the name of the region
        closest_regions = [conn.region_labels[idx] for idx in closest_idxs]
        closest_regs_dict = {ch: reg for ch, reg in zip(ch_names, closest_regions)}

        # make sure subject ID in BIDS path doesn't hve '_'
        subject = subject.replace("_", "")

        source_paths = seeg_path.glob("*.edf")
        for run_id, fpath in enumerate(source_paths):
            bids_kwargs = {
                "subject": subject,
                "session": session,
                "task": task,
                "acquisition": modality,
                "run": run_id + 1,
                "datatype": datatype,
                "suffix": datatype,
                "extension": extension,
            }

            # run main bids conversion
            output_dict = write_edf_to_bids(
                edf_fpath=fpath,
                bids_kwargs=bids_kwargs,
                bids_root=root,
                line_freq=line_freq,
                montage=montage,
                dataset_name="tvb",
            )
            bids_fname = output_dict["output_fname"]

            # append scans original filenames
            append_original_fname_to_scans(fpath.name, root, bids_fname)

            # append region location to electrodes.tsv
            electrodes_tsv_fpath = BIDSPath(
                root=root,
                subject=subject,
                session=session,
                space=space,
                acquisition=modality,
                datatype=datatype,
            )
            electrodes_tsv_fpath.update(suffix="electrodes", extension=".tsv")
            atlas_depth = "desikan-killiany"
            print(closest_regs_dict)
            electrodes_tsv = _update_electrodes_tsv(
                electrodes_tsv_fpath, closest_regs_dict, atlas_depth
            )

        add_data_to_participants(subject=subject, bids_root=root)


def convert_sickkids_dataset():
    WORKSTATION = "home"

    if WORKSTATION == "home":
        # bids root to write BIDS data to
        bids_root = Path("/Users/adam2392/OneDrive - Johns Hopkins/sickkids/")
        source_dir = bids_root / "sourcedata"

    elif WORKSTATION == "lab":
        bids_root = Path("/home/adam2392/hdd2/epilepsy_bids/")
        source_dir = Path("/home/adam2392/hdd2/epilepsy_bids/sourcedata")

    # define BIDS identifiers
    modality = "ecog"
    task = "ictal"
    task = 'pre'
    sessions = [
        "preresection",
        # 'extraoperative',
        # 'intraresection',
        # 'postresection'
    ]
    datatype = "ieeg"
    line_freq = 60
    overwrite = True
    verbose = True

    subject_ids = [
        # "E1",
        # 'E2',
        # "E3",
        # "E4",
        # "E5",
        "E6",
        # "E7",
    ]

    participants_json_fname = os.path.join(bids_root, "participants.json")
    participants_tsv_fname = os.path.join(bids_root, "participants.tsv")

    bids_root = bids_root / 'derivatives'
    # for subject in subject_ids:
    #     add_data_to_participants(subject, bids_root)
    #
    # exit(1)
    for session in sessions:
        source_folder = source_dir / session

        # storage for participant info
        keys, descriptions, values = [], [], []
        levels, units = [], []

        # regex pattern for the files is:
        for subject in subject_ids:
            search_str = f"{subject}*.edf"
            # search_str = '*.EDF'
            filepaths = (source_folder / subject).glob(search_str)
            filepaths = (source_folder).glob(search_str)

            # get a list of filepaths for each "task"
            task_filepaths = collections.defaultdict(list)
            print(source_folder / subject)
            print(search_str)
            print(task_filepaths)
            for idx, fpath in enumerate(filepaths):
                subject = fpath.name.split("_")[0]
                if subject not in subject_ids:
                    print("wtf?")
                    continue

                # get task from the filename
                task = fpath.name.split("_")[1].split(".")[0]
                task = _extraop_maps(task, subject)

                # task = fpath.name.split('_')[]
                task_filepaths[task].append(fpath)
            print(task_filepaths)
            for task, filepaths in task_filepaths.items():
                for run_id, fpath in enumerate(filepaths):
                    # subject = fpath.name.split("_")[0]
                    # if subject not in subject_ids:
                    #     print("wtf?")
                    #     continue

                    # get the next available run
                    run_id = run_id + 1

                    bids_kwargs = {
                        "subject": subject,
                        "session": session,
                        "task": task,
                        "acquisition": modality,
                        "run": run_id,
                        "datatype": datatype,
                        "suffix": datatype,
                    }
                    print(bids_kwargs)

                    # run main bids conversion
                    output_dict = write_edf_to_bids(
                        edf_fpath=fpath,
                        bids_kwargs=bids_kwargs,
                        bids_root=bids_root,
                        line_freq=line_freq,
                        dataset_name="sickkids",
                        source_dir=source_dir,
                        overwrite=overwrite
                    )
                    bids_fname = output_dict["output_fname"]

                    # append scans original filenames
                    append_original_fname_to_scans(fpath.name, bids_root, bids_fname)

                    bids_path = BIDSPath(**bids_kwargs, root=bids_root)
                    print(bids_path.fpath)
                    raw = read_raw_bids(bids_path)


def convert_hsc_interictal_dataset():
    #  bids root to write BIDS data to
    bids_root = Path("/Users/adam2392/OneDrive - Johns Hopkins/sickkids_extraop/")
    source_dir = bids_root / "sourcedata"

    # define BIDS identifiers
    task = "interictal"
    session = 'extraoperative'
    datatype = "ieeg"
    line_freq = 60
    extension = '.edf'
    overwrite = True
    verbose = True

    source_folder = source_dir

    subject_ids = [f.name for f in source_dir.glob('*') if f.is_dir()]

    # regex pattern for the files is:
    for subject in subject_ids:
        search_str = '*.EDF'
        filepaths = (source_folder / subject).rglob(search_str)

        for run_id, fpath in enumerate(filepaths):
            # get the next available run
            run_id = run_id + 1

            bids_kwargs = {
                "subject": subject,
                "session": session,
                "task": task,
                'acquisition': None,
                "run": run_id,
                "datatype": datatype,
                "suffix": datatype,
                'extension': extension
            }
            print(bids_kwargs)

            # run main bids conversion
            output_dict = write_edf_to_bids(
                edf_fpath=fpath,
                bids_kwargs=bids_kwargs,
                bids_root=bids_root,
                line_freq=line_freq,
                dataset_name="sickkids interictal extraop",
                source_dir=source_dir,
                overwrite=overwrite
            )
            bids_fname = output_dict["output_fname"]

            # append scans original filenames
            append_original_fname_to_scans(fpath.name, bids_root, bids_fname)


if __name__ == "__main__":
    convert_sickkids_dataset()
    # convert_hsc_interictal_dataset()
    # convert_tvb_dataset()
    exit(1)
