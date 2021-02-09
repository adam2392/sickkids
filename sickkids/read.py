import mne
from eztrack import preprocess_ieeg
from mne_bids import read_raw_bids


def load_data(bids_path, resample_sfreq, deriv_root, plot_raw=False, verbose=None):
    # load in the data
    raw = read_raw_bids(bids_path)

    if resample_sfreq:
        # perform resampling
        raw = raw.resample(resample_sfreq, n_jobs=-1)

    raw = raw.pick_types(seeg=True, ecog=True, eeg=True, misc=False, exclude=[])
    raw.load_data()

    # pre-process the data using preprocess pipeline
    print("Power Line frequency is : ", raw.info["line_freq"])
    l_freq = 0.5
    h_freq = 200
    raw = preprocess_ieeg(raw, l_freq=l_freq, h_freq=h_freq, verbose=verbose)

    if plot_raw is True:
        # plot raw data
        deriv_root.mkdir(exist_ok=True, parents=True)
        fig_basename = bids_path.copy().update(extension=".svg", check=False).basename
        scale = 200e-6
        fig = raw.plot(
            decim=40,
            duration=20,
            scalings={"ecog": scale, "seeg": scale},
            n_channels=len(raw.ch_names),
            clipping=None,
        )
        fig.savefig(deriv_root / fig_basename)
    return raw
