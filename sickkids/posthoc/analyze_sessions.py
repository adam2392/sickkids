from pathlib import Path

from eztrack.io import read_derivative_npy


def load_all_sessions(subject, sessions, deriv_root, deriv_chain):
    deriv_path = deriv_root / deriv_chain / f"sub-{subject}"

    deriv_list = []
    for session in sessions:
        pattern = f"*ses-{session}*_desc-perturbmatrix*.json"

        # get all files
        fpaths = list(deriv_path.glob(pattern))
        for fpath in fpaths:
            deriv = read_derivative_npy(fpath)

            deriv_list.append(deriv)
    return deriv_list


def run_analysis():
    root = Path("/Users/adam2392/OneDrive - Johns Hopkins/sickkids/")
    deriv_root = root / "derivatives" / "originalsampling" / "radius1.5"
    figures_path = deriv_root / "figures"

    # define BIDS entities
    SUBJECTS = [
        "E1",
        # 'E3',
        # 'E4',
        # 'E5', 'E6', 'E7'
    ]

    # task = "pre"
    acquisition = "ecog"
    datatype = "ieeg"
    extension = ".vhdr"
    session = "postresection"  # only one session

    # analysis parameters
    reference = "monopolar"
    sfreq = None
    overwrite = False

    sessions = ["extraoperative", "preresection", "intraresection", "postresection"]

    deriv_chain = Path("originalsampling") / "radius1.5" / "fragility" / reference

    for subject in SUBJECTS:
        deriv_list = load_all_sessions(subject, sessions, deriv_root, deriv_chain)

        # concatenate
