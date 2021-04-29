from typing import List

import numpy as np
from sklearn.preprocessing import StandardScaler


def filter_outliers(
    deriv_mat, threshold_val: float = 1, sigma: float = 3, ignore_val: float = None
):
    """Obtain outlier boolean mask on the deriv_mat.

    Parameters
    ----------
    deriv_mat : np.ndarray (n_chs, n_times)
    threshold_val : float
    sigma : float

    Returns
    -------
    outlier_mask : np.ndarray (n_chs, n_times)
    masked_deriv_mat : np.ma.masked_array (n_chs, n_times)
    """
    # perform z-score
    scaler = StandardScaler(with_std=True, with_mean=True, copy=True)
    scaled_mat = scaler.fit_transform(deriv_mat)

    # if ignore_val is not None:
    # #     if not isinstance(ignore_vals, list):
    # #         ignore_vals = [ignore_vals]
    #
    #     # find indices to ignore
    #     ignore_idx = np.where(deriv_mat == ignore_val)[0]
    # else:
    #     ignore_idx = []

    # threshold values
    thresh_mat = scaled_mat.copy()
    thresh_mat[thresh_mat <= sigma] = 0
    thresh_mat[thresh_mat > sigma] = threshold_val

    # print(np.max(scaled_mat), np.min(scaled_mat))
    # print(np.where(thresh_mat))
    # create mask
    outlier_mask = np.ma.make_mask(thresh_mat, copy=False, shrink=False)

    masked_deriv_mat = np.ma.masked_array(deriv_mat, outlier_mask, fill_value=np.nan)

    return masked_deriv_mat
