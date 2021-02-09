import mne

from eztrack.preprocess.bids_conversion import update_participants_info


def _set_ch_types_tvb(raw, subject):
    raw.set_channel_types({ch: "seeg" for ch in raw.ch_names})
    if subject == "id013pg":
        bads = [
            "A3",
            "A8",
            "A9",
            "A10",
            "A11",
            "A12",
            "A13",
            "A14",
            "A15",
            "B1",
            "B2",
            "B3",
            "B4",
            "B8",
            "B9",
            "B13",
            "B14",
            "B15",
            "TB1",
            "C7",
            "C8",
            "C9",
            "C10",
            "OT1",
            "OP1",
            "PI1",
        ]
        raw.info["bads"].extend(bads)

        # map events
        events, event_id = mne.events_from_annotations(raw)
        # map all event descriptions to lower case
        event_id = {key.lower(): val for key, val in event_id.items()}

        # map
        for key in event_id:
            if "crise" in key:
                event_id["eeg sz onset"] = event_id.pop(key)
                break
        if "fin" in event_id:
            event_id["eeg sz offset"] = event_id.pop("fin")

        # create annotations from events
        event_desc = {val: key for key, val in event_id.items()}
        annot = mne.annotations_from_events(
            events, raw.info["sfreq"], event_desc, orig_time=raw.info["meas_date"]
        )
        raw.set_annotations(annot)
    elif subject == "id008gc":
        bads = [
            "R1",
            "CR1",
            "FD1",
            "CC1",
            "OR1",
            "OR8",
            "OR9",
            "TP1",
            "TP2",
            "A3",
            "B1",
            "B9",
            "B10",
            "GPH1",
            "T1",
        ]
        raw.info["bads"].extend(bads)

        # map events
        events, event_id = mne.events_from_annotations(raw)
        # map all event descriptions to lower case
        event_id = {key.lower(): val for key, val in event_id.items()}

        for key in event_id:
            if "crise" in key:
                event_id["eeg sz onset"] = event_id.pop(key)
                break
        if "fin" in event_id:
            event_id["eeg sz offset"] = event_id.pop("fin")
        elif "fin eeg" in event_id:
            event_id["eeg sz offset"] = event_id.pop("fin eeg")

        # create annotations from events
        event_desc = {val: key for key, val in event_id.items()}
        annot = mne.annotations_from_events(
            events, raw.info["sfreq"], event_desc, orig_time=raw.info["meas_date"]
        )
        raw.set_annotations(annot)
    return raw


def add_data_to_participants(subject, bids_root):
    if subject == "id008gc":
        entries = [
            ("age", 33),
            ("onset_age", 28),
            ("sex", "M"),
            ("outcome", "F"),
            ("engel_score", 3),
            ("clinically_annotated_soz", "right-temporal-lobe"),
        ]
    if subject == "id013pg":
        entries = [
            ("age", 36),
            ("onset_age", 7),
            ("sex", "M"),
            ("outcome", "S"),
            ("engel_score", 1),
            ("clinically_annotated_soz", "right-temporal-lobe"),
        ]

    description_dict = {
        "onset_age": "Age at epilepsy onset",
        "outcome": "Success of failed surgery.",
        "engel_score": "A clinical classification from 1-4. See literature for better overview.",
        "clinically_annotated_soz": "The region that was clinically isolated to probably contain the SOZ.",
    }

    levels_dict = {
        "outcome": {
            "S": "successful surgery; seizure freedom",
            "F": "failed surgery; recurring seizures",
            "NR": "no resection/surgery",
        },
        "engel_score": {
            "1": "seizure free",
            "2": "significant seizure frequency reduction",
            "3": "slight seizure frequency reduction",
            "4": "no change",
            "-1": "no surgery, or outcome",
        },
        "clinically_annotated_soz": "",
    }
    units_dict = {
        "onset_age": "years",
        "outcome": "",
        "engel_score": "",
        "clinically_annotated_soz": "",
    }

    # append subject information
    for key, value in entries:
        levels = levels_dict.get(key)
        units = units_dict.get(key)
        description = description_dict.get(key)
        if description == "":
            description = None
        update_participants_info(
            bids_root,
            subject,
            key,
            value,
            description=description,
            levels=levels,
            units=units,
        )
