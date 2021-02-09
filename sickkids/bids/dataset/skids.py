import numpy as np
from eztrack.preprocess.bids_conversion import update_participants_info

from sickkids.bids.utils import BadChannelDescription


def _set_ch_types(raw, subject, session):
    """Set channel types manually.

    bads = bad electrophsyio chs
    grids = grid electrodes
    strips = strip electrodes
    depths = depth (sEEG) electrodes
    resected = removed during resective surgery
    disconnected_chs = disconnected to facilitate surgery.
    """
    raw.set_channel_types({ch: "misc" for ch in raw.ch_names})
    ref = ""
    if subject == "E1":
        grids = [f"C{idx}" for idx in range(1, 65)]
        strips = []
        depths = (
            [f"SAD{idx}" for idx in range(1, 7)]
            + [f"SPD{idx}" for idx in range(1, 7)]
            + [f"MPOD{idx}" for idx in range(1, 7)]
            + [f"IAD{idx}" for idx in range(1, 7)]
            + [f"MAOD{idx}" for idx in range(1, 7)]
            + [f"IPD{idx}" for idx in range(1, 7)]
        )
        ref = "C15"

        # cut out grid around here and scooped all the depths
        # underneat the 4 channel grid
        resected = ["C19", "C20", "C27", "C28"]
        resected.extend(depths)

        # electrophysiologically bad
        bads = {
            "C15": BadChannelDescription.REFERENCE,
            "C57": BadChannelDescription.HIGHFREQ,
        }
        # append bads by clinicians without resected though
        # add bad resected channels
        disconnected_chs = []
        if session in ["postresection", "intraresection"]:
            disconnected_chs.extend(
                ["C15", "C17", "C18", "C19", "C20"]
                + [f"C{idx}" for idx in range(25, 29)]
                + ["C34", "C51", "C57", "C59"]
            )
            disconnected_chs.extend(depths)
    elif subject == "E2":
        grids = [f"C{idx}" for idx in range(1, 65)]
        strips = []
        depths = []
        resected = ["C1", "C2"]
        bads = [f"C{idx}" for idx in range(1, 11)]
        bads = []
    elif subject == "E3":
        grids = [f"C{idx}" for idx in range(1, 65)]
        strips = [f"PS{idx}" for idx in range(1, 5)]
        depths = (
            [f"ASD{idx}" for idx in range(1, 7)]
            + [f"PD{idx}" for idx in range(1, 7)]
            + [f"AID{idx}" for idx in range(1, 7)]
        )

        # resected included some of the depth region
        resected = (
            ["C3", "C4", "C11", "C12", "C19", "C20", "C27", "C28", "PD3", "PD4", "PD5"]
            + [f"ASD{idx}" for idx in range(1, 7)]
            + [f"AID{idx}" for idx in range(1, 7)]
        )
        ref = "C51"

        bads = {
            "C51": BadChannelDescription.REFERENCE,
            "PD6": BadChannelDescription.HIGHFREQ,
            "ASD1": BadChannelDescription.HIGHFREQ,
        }
        # append bads by clinicians without resected though
        # add bad resected channels
        disconnected_chs = []
        if session in ["postresection", "intraresection"]:
            disconnected_chs.extend(
                [f"C{idx}" for idx in range(1, 6)]
                + [f"C{idx}" for idx in range(9, 15)]
                + [f"C{idx}" for idx in range(17, 24)]
                + [f"C{idx}" for idx in range(25, 32)]
                + ["C51"]
            )
            disconnected_chs.extend(depths)
    elif subject == "E4":
        grids = [f"C{idx}" for idx in range(1, 49)]
        strips = []
        depths = (
            [f"F2AL{idx}" for idx in range(1, 7)]
            + [f"F2BC{idx}" for idx in range(1, 7)]
            + [f"F2CL{idx}" for idx in range(1, 7)]
        )

        # depths?
        resected = (
            ["C19", "C20", "C27", "C28", "C35", "C36"]
            + [f"F2AL{idx}" for idx in range(1, 7)]
            + [f"F2BC{idx}" for idx in range(1, 7)]
        )
        # + [f'F2CL{idx}' for idx in range(1, 7)]
        ref = "C3"
        bads = {
            "C1": BadChannelDescription.FLAT,
            "C2": BadChannelDescription.FLAT,
            "C3": BadChannelDescription.REFERENCE,
            "C9": BadChannelDescription.FLAT,
            "C10": BadChannelDescription.FLAT,
            "C17": BadChannelDescription.FLAT,
            "C18": BadChannelDescription.FLAT,
            "C25": BadChannelDescription.FLAT,
            "C26": BadChannelDescription.FLAT,
            "C33": BadChannelDescription.FLAT,
            "C34": BadChannelDescription.FLAT,
            "C41": BadChannelDescription.FLAT,
            "C42": BadChannelDescription.FLAT,
        }

        # add bad resected channels
        disconnected_chs = []
        if session in ["postresection", "intraresection"]:
            disconnected_chs.extend(
                [f"C{idx}" for idx in range(1, 6)]
                + [f"C{idx}" for idx in range(9, 14)]
                + [f"C{idx}" for idx in range(17, 20)]
                + [f"C{idx}" for idx in range(25, 27)]
                + [f"C{idx}" for idx in range(33, 37)]
                + ["C20", "C25", "C26", "C27", "C28", "C41", "C42", "C47"]
            )
            disconnected_chs.extend(depths)
    elif subject == "E5":
        grids = [f"C{idx}" for idx in range(1, 49)]
        strips = []
        depths = (
            [f"ML{idx}" for idx in range(1, 7)]
            + [f"F3C{idx}" for idx in range(1, 7)]
            + [f"F1OF{idx}" for idx in range(1, 7)]
        )

        # includes ML/F3C depth undernead
        resected = ["C19", "C20", "C21", "C27", "C28", "C29", "C36", "C37"]
        resected.extend(depths)
        ref = "C7"
        bads = {
            "C1": BadChannelDescription.FLAT,
            "C7": BadChannelDescription.REFERENCE,
            "C9": BadChannelDescription.FLAT,
            "C11": BadChannelDescription.HIGHFREQ,
            "C17": BadChannelDescription.FLAT,
            "C25": BadChannelDescription.FLAT,
            "C33": BadChannelDescription.FLAT,
            "C41": BadChannelDescription.FLAT,
        }
        # add bad resected channels
        disconnected_chs = []
        if session in ["postresection", "intraresection"]:
            disconnected_chs.extend(
                ["C2", "C7", "C16"]
                + [f"C{idx}" for idx in range(10, 15)]
                + [f"C{idx}" for idx in range(18, 22)]
                + [f"C{idx}" for idx in range(26, 30)]
                + [f"C{idx}" for idx in range(34, 38)]
            )
            disconnected_chs.extend(depths)
    elif subject == "E6":
        grids = [f"C{idx}" for idx in range(1, 49)] + [
            f"C{idx}" for idx in range(50, 57)
        ]
        strips = []
        depths = (
            [f"1D{idx}" for idx in range(1, 7)]
            + [f"2D{idx}" for idx in range(1, 7)]
            + [f"3D{idx}" for idx in range(1, 7)]
        )
        resected = ["C19", "C20", "C28", "C29", "C37", "C38"]
        resected.extend(depths)
        # resected.extend(['1D4', '1D5', '1D6',
        #                  '2D3', '2D4', '2D5', '2D6',
        #                  '3D1'])
        bads = {
            "C15": BadChannelDescription.HIGHFREQ,
        }
        # add disconnected channels
        disconnected_chs = []
        if session in ["postresection", "intraresection"]:
            disconnected_chs.extend(
                ["C15", "C49"]
                + [f"C{idx}" for idx in range(17, 21)]
                + [f"C{idx}" for idx in range(24, 30)]
                + [f"C{idx}" for idx in range(33, 39)]
                + [f"C{idx}" for idx in range(52, 55)]
            )
            disconnected_chs.extend(depths)
    elif subject == "E7":
        grids = [f"C{idx}" for idx in range(1, 65)]
        strips = (
            [f"ATPS{idx}" for idx in range(1, 7)]
            + [f"ABTS{idx}" for idx in range(1, 7)]
            + [f"PBTS{idx}" for idx in range(1, 7)]
        )
        depths = (
            [f"AD{idx}" for idx in range(1, 7)]
            + [f"HD{idx}" for idx in range(1, 7)]
            + [f"TOD{idx}" for idx in range(1, 7)]
            + [f"POD{idx}" for idx in range(1, 7)]
            + [f"FOD{idx}" for idx in range(1, 7)]
        )

        resected = [
            "C41",
            "C42",
            "C49",
            "C50",
            "C51",
            "C57",
            "C58",
            "C59",
            "C60",
            "AD1",
            "AD2",
            "AD3",
            "AD4",
            "AD5",
            "AD6",
        ]
        resected.extend(["HD3", "HD4", "HD5", "HD6"])
        resected.extend(strips)
        ref = "C12"
        bads = {
            "C12": BadChannelDescription.REFERENCE.value,
            # 'AD1': BadChannelDescription.HIGHFREQ.value
        }

        # add disconnected channels
        disconnected_chs = []
        if session in ["postresection", "intraresection"]:
            disconnected_chs.extend(
                ["C12", "C24", "C38", "C41", "C42", "C49", "C50", "C51"]
                + [f"C{idx}" for idx in range(57, 61)]
            )
            disconnected_chs.extend(
                [f"AD{idx}" for idx in range(1, 7)]
                + [f"HD{idx}" for idx in range(1, 7)]
            )

    # set additional bad channels
    if session in ["postresection", "intraresection"]:
        for ch in disconnected_chs:
            bads[ch] = BadChannelDescription.DISCONNECTED
        for ch in resected:
            bads[ch] = BadChannelDescription.DISCONNECTED

    # set strip, grid, depth electrodes
    raw.set_channel_types({ch: "ecog" for ch in strips})
    raw.set_channel_types({ch: "ecog" for ch in grids})
    raw.set_channel_types({ch: "seeg" for ch in depths})

    # set bads
    raw.info["bads"].extend(bads.keys())
    raw.info["bads_description"] = bads

    # return channel mapping
    ch_mapping = {
        "grid": grids,
        "strip": strips,
        "depth": depths,
        "resected": resected,
        "reference": ref,
    }
    return ch_mapping


def _get_ch_sizes(raw):
    ch_sizes = {}
    for ch_name in raw.ch_names:
        ch_type = raw.get_channel_types(picks=[ch_name])
        assert len(ch_type) == 1
        ch_type = ch_type[0]
        if ch_type == "ecog":
            size = 4.0 * np.pi  # 100 Ohm
        elif ch_type == "seeg":
            size = 0.55 * 0.55 * np.pi  # 150 Ohms for SD depths with 8 contacts or less
        else:
            continue
        ch_sizes[ch_name] = size
    return ch_sizes


def _set_ch_meta(raw):
    impedances = {}
    for ch_name in raw.ch_names:
        ch_type = raw.get_channel_types(picks=[ch_name])
        assert len(ch_type) == 1
        ch_type = ch_type[0]
        if ch_type == "ecog":
            impedance = 0.1  # 100 Ohm
        elif ch_type == "seeg":
            impedance = 0.15  # 150 Ohms for SD depths with 8 contacts or less
        else:
            continue
        imp_unit = "kOhm"
        impedances[ch_name] = {"imp": impedance, "imp_unit": imp_unit}
    raw.impedances = impedances
    return raw


def add_data_to_participants(subject, bids_root):
    if subject == "E1":
        entries = [
            ("age", 9),
            ("sex", "M"),
            ("hand", "n/a"),
            ("intellectual-function", 77),
            ("verbal-comprehension", 78),
            ("visual-spatial", 77),
            ("visual-fluid-reasoning", 78),
            ("working-memory", 77),
            ("visual-processing-speed", 77),
            ("verbal-memory-delay", "average"),
            ("visual-memory-delay", "average"),
            ("outcome", "F"),
            ("engel_score", 3),
            ("months_follow_up", 24),
            ("clinically_annotated_soz", "right-frontal-lobe"),
        ]
    if subject == "E2":
        entries = [
            ("age", 14),
            ("sex", "M"),
            ("hand", "n/a"),
            ("intellectual-function", 91),
            ("verbal-comprehension", 78),
            ("visual-spatial", 95),
            ("visual-fluid-reasoning", 115),
            ("working-memory", 69),
            ("visual-processing-speed", 103),
            ("verbal-memory-delay", "limited"),
            ("visual-memory-delay", "average"),
            ("outcome", "S"),
            ("engel_score", 1),
            ("months_follow_up", 18),
            ("clinically_annotated_soz", "left-parietal-lobe"),
        ]
    elif subject == "E3":
        entries = [
            ("age", 13),
            ("sex", "F"),
            ("hand", "n/a"),
            ("intellectual-function", 104),
            ("verbal-comprehension", 92),
            ("visual-spatial", 115),
            ("visual-fluid-reasoning", 106),
            ("working-memory", 83),
            ("visual-processing-speed", 100),
            ("verbal-memory-delay", "average"),
            ("visual-memory-delay", "average"),
            ("outcome", "S"),
            ("engel_score", 1),
            ("months_follow_up", 6),
            ("clinically_annotated_soz", "right-parietal-lobe"),
        ]
    elif subject == "E4":
        entries = [
            ("age", 11),
            ("sex", "M"),
            ("hand", "n/a"),
            ("intellectual-function", 100),
            ("verbal-comprehension", 106),
            ("visual-spatial", 107),
            ("visual-fluid-reasoning", 103),
            ("working-memory", 96),
            ("visual-processing-speed", 92),
            ("verbal-memory-delay", "high-average"),
            ("visual-memory-delay", "average"),
            ("outcome", "S"),
            ("engel_score", 1),
            ("months_follow_up", 14),
            ("clinically_annotated_soz", "right-frontal-lobe"),
        ]
    elif subject == "E5":
        entries = [
            ("age", 11),
            ("sex", "M"),
            ("hand", "n/a"),
            ("intellectual-function", 73),
            ("verbal-comprehension", 64),
            ("visual-spatial", 95),
            ("visual-fluid-reasoning", 81),
            ("working-memory", 77),
            ("visual-processing-speed", 77),
            ("verbal-memory-delay", "low-average"),
            ("visual-memory-delay", "low-average"),
            ("outcome", "S"),
            ("engel_score", 1),
            ("months_follow_up", 24),
            ("clinically_annotated_soz", "right-frontal-lobe"),
        ]
    elif subject == "E6":
        entries = [
            ("age", 14),
            ("sex", "F"),
            ("hand", "n/a"),
            ("intellectual-function", 110),
            ("verbal-comprehension", 103),
            ("visual-spatial", 125),
            ("visual-fluid-reasoning", 106),
            ("working-memory", 96),
            ("visual-processing-speed", 109),
            ("verbal-memory-delay", "high-average"),
            ("visual-memory-delay", "average"),
            ("outcome", "S"),
            ("engel_score", 1),
            ("months_follow_up", 7),
            ("clinically_annotated_soz", "right-parietal-lobe"),
        ]
    elif subject == "E7":
        entries = [
            ("age", 17),
            ("sex", "M"),
            ("hand", "n/a"),
            ("intellectual-function", 115),
            ("verbal-comprehension", 118),
            ("visual-spatial", 100),
            ("visual-fluid-reasoning", 110),
            ("working-memory", 117),
            ("visual-processing-speed", 111),
            ("verbal-memory-delay", "low-average"),
            ("visual-memory-delay", "high-average"),
            ("outcome", "S"),
            ("engel_score", 1),
            ("months_follow_up", 3),
            ("clinically_annotated_soz", "left-frontal-temporal-lobe"),
        ]

    description_dict = {
        "intellectual-function": "",
        "verbal-comprehension": "",
        "visual-spatial": "",
        "visual-fluid-reasoning": "",
        "working-memory": "",
        "visual-processing-speed": "",
        "verbal-memory-delay": "",
        "visual-memory-delay": "",
        "outcome": "Success of failed surgery.",
        "engel_score": "A clinical classification from 1-4. See literature for better overview.",
        "months_follow_up": "Number of months since surgery that followed up to get surgical outcome.",
        "clinically_annotated_soz": "The region that was clinically isolated to probably contain the SOZ.",
    }

    levels_dict = {
        "intellectual-function": "",
        "verbal-comprehension": "",
        "visual-spatial": "",
        "visual-fluid-reasoning": "",
        "working-memory": "",
        "visual-processing-speed": "",
        "verbal-memory-delay": "",
        "visual-memory-delay": "",
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
        "months_follow_up": "",
        "clinically_annotated_soz": "",
    }
    units_dict = {
        "intellectual-function": "",
        "verbal-comprehension": "",
        "visual-spatial": "",
        "visual-fluid-reasoning": "",
        "working-memory": "",
        "visual-processing-speed": "",
        "verbal-memory-delay": "",
        "visual-memory-delay": "",
        "outcome": "",
        "engel_score": "",
        "months_follow_up": "months",
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
