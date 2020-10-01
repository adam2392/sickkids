from pathlib import Path

from eztrack import (
    preprocess_raw,
    lds_raw_fragility,
    write_result_fragility,
    plot_result_heatmap,
)
from mne_bids import read_raw_bids, BIDSPath, get_entity_vals


def run_analysis(
        bids_path, reference="monopolar", resample_sfreq=None, deriv_path=None, figures_path=None, verbose=True, overwrite=False
):
    subject = bids_path.subject

    # use the same basename to save the data
    deriv_basename = bids_path.basename

    # if len(list(deriv_path.rglob(f'{deriv_basename}*'))) > 0 and not overwrite:
    #     warn('Need to set overwrite to True if you want '
    #          f'to overwrite {deriv_basename}')
    #     return

    # load in the data
    raw = read_raw_bids(bids_path)

    if resample_sfreq:
        # perform resampling
        raw = raw.resample(resample_sfreq, n_jobs=-1)

    if deriv_path is None:
        deriv_path = (
                bids_path.root
                / "derivatives"
                # / 'nodepth'
                / f"{int(raw.info['sfreq'])}Hz"
                / "fragility"
                / reference
                / f"sub-{subject}"
        )
    # set where to save the data output to
    if figures_path is None:
        figures_path = (
                bids_path.root
                / "derivatives"
                / "figures"
                # / 'nodepth'
                / f"{int(raw.info['sfreq'])}Hz"
                / "fragility"
                / reference
                / f"sub-{subject}"
        )
    deriv_root = (root / 'derivatives' / "figures"
                  # / 'nodepth' \
                 / f"{int(raw.info['sfreq'])}Hz" \
                 / "raw" \
                 / reference \
                 / f"sub-{subject}")

    raw = raw.pick_types(seeg=True, ecog=True,
                         eeg=True, misc=False,
                         exclude=[])
    raw.load_data()

    # pre-process the data using preprocess pipeline
    datatype = bids_path.datatype
    print('Power Line frequency is : ', raw.info["line_freq"])
    raw = preprocess_raw(raw, datatype=datatype,
                         verbose=verbose, method="simple", drop_chs=False)

    # plot raw data
    deriv_root.mkdir(exist_ok=True, parents=True)
    fig_basename = bids_path.copy().update(extension='.pdf').basename
    scale = 200e-6
    fig = raw.plot(
        decim=10,
        scalings={
            'ecog': scale,
            'seeg': scale
        }, n_channels=len(raw.ch_names))
    fig.savefig(deriv_root / fig_basename)

    raw.drop_channels(raw.info['bads'])

    # print(raw.ch_names)
    # print(raw.info['bads'])
    print(f"Analyzing {raw} with {len(raw.ch_names)} channels.")

    # raise Exception('hi')
    model_params = {
        "winsize": 500,
        "stepsize": 250,
        "radius": 1.5,
        "method_to_use": "pinv",
    }
    # run heatmap
    if reference == 'common':
        reference = raw.ch_names
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

    # normalize in place
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
        # 'E2',
        # 'E3',
        'E4',
        # 'E5', 'E6'
    ]

    session = "postsurgery"  # only one session

    # pre, Sz, Extraoperative, post
    # task = "pre"
    acquisition = "ecog"
    datatype = "ieeg"
    extension = ".vhdr"

    # analysis parameters
    reference = 'common'
    sfreq = None

    sessions = [
        'presurgery',
        'extraoperative',
        'intraoperative',
        'postsurgery'
    ]

    # get the runs for this subject
    all_subjects = get_entity_vals(root, "subject")
    for subject in all_subjects:
        if subject not in subjects:
            continue
        ignore_subs = [sub for sub in all_subjects if sub != subject]
        ignore_sessions = [ses for ses in sessions if ses != session]
        ignore_set = {
            'ignore_subjects': ignore_subs,
            'ignore_sessions': ignore_sessions,
        }
        print(f'Ignoring these sets: {ignore_set}')
        all_tasks = get_entity_vals(root, "task", **ignore_set)
        tasks = all_tasks
        # tasks = ['pre']

        for task in tasks:
            print(f"Analyzing {task} task.")
            ignore_tasks = [tsk for tsk in all_tasks if tsk != task]
            ignore_set['ignore_tasks'] = ignore_tasks
            runs = get_entity_vals(
                root, 'run', **ignore_set
            )
            print(ignore_subs)
            print(ignore_tasks)
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

                run_analysis(bids_path, reference=reference,
                             resample_sfreq=sfreq)
