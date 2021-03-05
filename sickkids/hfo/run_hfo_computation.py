def compute_hfos(bids_path, deriv_root, reference: str = "monopolar"):
    """Compute HFOs using mne-hfo from a dataset.

    Derivative output should be named accordingly and inside
    ``derivatives/hfo/sub-<subject>/ses-<session>/ieeg/<bids_derivative_fname>``.
    """
    pass


def main():
    # define hyperparameters
    reference = "monopolar"
    l_freq = 80
    h_freq = 250

    # define bids path

    # define derivatives path

    # run HFO computation
    compute_hfos()
