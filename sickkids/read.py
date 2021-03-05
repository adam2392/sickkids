import mne
import numpy as np
from eztrack import preprocess_ieeg
from mne_bids import read_raw_bids


def raw_from_neo(fname):
    """See: https://gist.github.com/agramfort/7fc27a18fcdc0e8cff3f"""
    import neo

    seg_micromed = neo.MicromedIO(filename=fname).read_segment()

    # convert channel bundle
    ch_bundle = seg_micromed.analogsignals[0].name
    ch_names = ch_bundle.split(",")
    ch_names = [ch if "(" not in ch else ch.split("(")[1] for ch in ch_names]
    ch_names = [ch.split(")")[0] for ch in ch_names]
    # ch_names = [sig.name for sig in seg_micromed.analogsignals]

    print(len(seg_micromed.analogsignals[0]))
    print(seg_micromed)
    print(seg_micromed.analogsignals[0].shape)

    # Because here we have the same on all chan
    sfreq = seg_micromed.analogsignals[0].sampling_rate

    data = np.asarray(seg_micromed.analogsignals).squeeze().T
    data *= 1e-6  # putdata from microvolts to volts
    # add stim channel
    # ch_names.append('STI 014')
    # data = np.vstack((data, np.zeros((1, data.shape[1]))))

    # To get sample number:
    # events_time = seg_micromed.eventarrays[0].times.magnitude * sfreq
    # n_events = len(events_time)
    # events = np.empty([n_events, 3])
    # events[:, 0] = events_time
    # events[:, 2] = seg_micromed.eventarrays[0].labels.astype(int)

    ch_types = ["eeg"] * len(ch_names)
    print(ch_names)
    print(sfreq)
    print(data.shape)
    print(ch_types)
    info = mne.create_info(ch_names, sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(data, info)
    # raw.add_events(events)
    return raw


def _handle_bids_trc(raw, bids_path):
    from mne_bids.read import (
        _handle_info_reading,
        _handle_events_reading,
        _handle_channels_reading,
    )

    sidecar_fname = bids_path.copy().update(extension=".json")
    events_fname = bids_path.copy().update(suffix="events", extension=".tsv")
    channels_fname = bids_path.copy().update(suffix="channels", extension=".tsv")

    # basically redo read_raw_bids
    raw = _handle_info_reading(sidecar_fname, raw)

    raw = _handle_events_reading(events_fname, raw)

    raw = _handle_channels_reading(channels_fname, bids_path, raw)
    return raw


def load_data(bids_path, resample_sfreq, deriv_root, plot_raw=False, verbose=None):
    # load in the data
    if bids_path.fpath.as_posix().endswith(".TRC"):
        raw = raw_from_neo(bids_path.fpath.as_posix())

        raw = _handle_bids_trc(raw, bids_path)

        raw._filenames.append(bids_path.fpath.as_posix())
        print(raw)
        print(raw.info)
    else:
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
