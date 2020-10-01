from pathlib import Path
import pandas as pd
from eztrack import (
    preprocess_raw,
    lds_raw_fragility,
    write_result_fragility,
    plot_result_heatmap,
    read_result_eztrack
)
from mne_bids import read_raw_bids, BIDSPath, get_entity_vals


def run_viz(
        bids_path, reference="monopolar", resample_sfreq=None, deriv_path=None, figures_path=None, verbose=True, overwrite=False
):
    subject = bids_path.subject

    # use the same basename to save the data
    deriv_basename = bids_path.basename

    # load in the data
    raw = read_raw_bids(bids_path)

    if resample_sfreq:
        # perform resampling
        raw = raw.resample(resample_sfreq, n_jobs=-1)

    if deriv_path is None:
        deriv_path = (
                bids_path.root
                / "derivatives"
        )
    deriv_path = (deriv_path
                  # /  'nodepth'
                  / f"{int(raw.info['sfreq'])}Hz"
                  / "fragility"
                  / reference
                  / f"sub-{subject}")
    # set where to save the data output to
    if figures_path is None:
        figures_path = (
                bids_path.root
                / "derivatives"
                / "figures"
        )
    figures_path = (figures_path
                    # / 'nodepth'
                    / f"{int(raw.info['sfreq'])}Hz"
                    / "fragility"
                    / reference
                    / f"sub-{subject}")

    # write results to
    source_entities = bids_path.entities
    raw_basename = BIDSPath(**source_entities).basename
    deriv_fname = list(deriv_path.glob(f'{raw_basename}*perturbmatrix*'))[0]
    result = read_result_eztrack(deriv_fname=deriv_fname,
                                 description='perturbmatrix',
                                 normalize=True)
    fig_basename = deriv_basename

    # read in sidecar channels.tsv
    channels_pd = pd.read_csv(bids_path.copy().update(suffix='channels', extension='.tsv'), sep='\t')
    description_chs = pd.Series(channels_pd.description.values, index=channels_pd.name).to_dict()
    print(description_chs)
    resected_chs = [ch for ch, description in description_chs.items() if description == 'resected']
    print(f'Resected channels are {resected_chs}')
    print(result.get_data()[:].min(), result.get_data()[:].max())
    print(result.info['sfreq'], result.get_metadata().get('model_params'))
    # channels_pd['description'].tolist()

    # create the heatmap
    plot_result_heatmap(
        result=result,
        fig_basename=fig_basename,
        figures_path=figures_path,
        red_chs=resected_chs
    )


if __name__ == "__main__":
    # the root of the BIDS dataset
    root = Path("/Users/adam2392/OneDrive - Johns Hopkins/sickkids/")

    # define BIDS entities
    subjects = [
        "E1",
        # 'E2',
        # 'E3',
        # 'E4',
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

    all_sessions = [
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
        ignore_sessions = [ses for ses in all_sessions if ses != session]
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

                deriv_path = (
                        bids_path.root
                        / "derivatives"
                )
                figures_path = (deriv_path
                        / "figures"
                        # / 'norms'
                )
                run_viz(bids_path, reference=reference,
                        resample_sfreq=sfreq,
                        deriv_path=deriv_path, figures_path=figures_path)
