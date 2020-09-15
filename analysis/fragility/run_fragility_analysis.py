from pathlib import Path

from eztrack import (
    preprocess_raw,
    lds_raw_fragility,
    write_result_fragility,
    plot_result_heatmap,
)
from mne_bids import read_raw_bids, BIDSPath, get_entity_vals


def run_analysis(
    bids_path, reference="monopolar", deriv_path=None, figures_path=None, verbose=True
):
    subject = bids_path.subject

    # set where to save the data output to
    if deriv_path is None:
        deriv_path = (
            bids_path.root
            / "derivatives"
            / "1000Hz"
            / "fragility"
            / reference
            / f"sub-{subject}"
        )
    if figures_path is None:
        figures_path = (
            bids_path.root
            / "derivatives"
            / "figures"
            / "1000Hz"
            / "fragility"
            / reference
            / f"sub-{subject}"
        )

    # use the same basename to save the data
    deriv_basename = bids_path.basename

    # load in the data
    raw = read_raw_bids(bids_path)
    raw = raw.pick_types(seeg=True, ecog=True, eeg=True, misc=False)
    raw.load_data()

    # pre-process the data using preprocess pipeline
    datatype = bids_path.datatype
    print(raw.info["line_freq"])
    raw = preprocess_raw(raw, datatype=datatype, verbose=verbose, method="simple")

    # print(raw.ch_names)
    # print(raw.info['bads'])
    print(f"Analyzing {raw} with {len(raw.ch_names)} channels.")

    # raise Exception('hi')
    model_params = {
        "winsize": 250,
        "stepsize": 125,
        "radius": 1.5,
        "method_to_use": "pinv",
    }
    # run heatmap
    result, A_mats, delta_vecs_arr = lds_raw_fragility(
        raw, reference=reference, return_all=True, **model_params
    )

    # write results to
    result_sidecars = write_result_fragility(
        A_mats,
        delta_vecs_arr,
        result=result,
        deriv_basename=deriv_basename,
        deriv_path=deriv_path,
        verbose=verbose,
    )
    fig_basename = deriv_basename

    result.normalize()
    # create the heatmap
    plot_result_heatmap(
        result=result,
        fig_basename=fig_basename,
        figures_path=figures_path,
    )


if __name__ == "__main__":
    # the root of the BIDS dataset
    root = Path("/Users/adam2392/OneDrive - Johns Hopkins/sickkids/")

    # define BIDS entities
    subjects = [
        "E1",
        # 'E2', 'E3', 'E4', 'E5', 'E6'
    ]

    subject = "E1"
    session = "presurgery"  # only one session

    # pre, Sz, Extraoperative, post
    task = "pre"
    acquisition = "ecog"
    datatype = "ieeg"
    extension = ".vhdr"

    # analysis parameters
    reference = "average"

    # get the runs for this subject
    subjects = get_entity_vals(root, "subject")
    for subject in subjects:
        ignore_subs = [sub for sub in subjects if sub != subject]
        tasks = get_entity_vals(root, "task", ignore_subjects=ignore_subs)

        for task in tasks:
            print(f"Analyzing {task} task.")
            ignore_tasks = [tsk for tsk in tasks if tsk != task]
            runs = get_entity_vals(
                root, "run", ignore_subjects=ignore_subs, ignore_tasks=ignore_tasks
            )

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

                run_analysis(bids_path, reference=reference)
