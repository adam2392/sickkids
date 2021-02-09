import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.utils import resample


def compute_bootstrap_lr(dmd_modes, freqs, n_boot=500, random_state=None):
    # compute the modepower from the dmd modes
    modepower = np.linalg.norm(dmd_modes, ord=2, axis=0) ** 2

    # for anything on 0 frequencies, we will ignore
    zero_freq_inds = np.argwhere(np.abs(freqs) != 0)
    freqs = np.array(freqs)[zero_freq_inds]
    modepower = modepower[zero_freq_inds]

    # linear fit on the log of the frequencies
    lm = LinearRegression()
    lm.fit(np.log10(np.abs(freqs)).reshape(-1, 1), np.log10(modepower))
    dmd_int, dmd_slope = lm.intercept_, lm.coef_

    # perform bootstrapping to get the confidence interval of the power law fit
    intercepts = []
    slopes = []

    n_samples = freqs.size
    for iboot in range(n_boot):
        X_sample, y_sample = resample(
            freqs,
            modepower,
            replace=True,
            n_samples=n_samples,
            random_state=random_state,
        )
        # make sure frequencies are logged
        X_sample = np.log10(np.abs(X_sample))
        y_sample = np.log10(y_sample)

        # fit linear regression on bootstrapped samples
        lr = LinearRegression()
        lr.fit(X_sample, y_sample)
        int_, slope = lr.intercept_, lr.coef_

        # store samples
        intercepts.append(int_)
        slopes.append(slope)

    return dmd_int, intercepts, dmd_slope, slopes


def _compute_dist_from_boot(intercepts, slopes, freq):
    predicted = []
    for intercept, slope in zip(intercepts, slopes):
        pwr_predict = intercept + slope * np.log10(np.abs(freq))
        predicted.append(pwr_predict)
    return predicted


def compute_significant_freqs(dmd_modes, freqs, intercepts, slopes):
    # compute a 99% CI
    slope_mean = np.mean(slopes)
    int_mean = np.mean(intercepts)

    # compute the modepower from the dmd modes
    modepower = np.linalg.norm(dmd_modes, ord=2, axis=0) ** 2

    # for anything on 0 frequencies, we will ignore
    zero_freq_inds = np.argwhere(np.abs(freqs) != 0)
    freqs = np.array(freqs)[zero_freq_inds]
    modepower = modepower[zero_freq_inds]

    # compute 3 CI away
    significant_freqs = []
    for pwr, freq in zip(modepower, freqs):
        predicted_mean_pwr = int_mean + slope_mean * np.log10(np.abs(freq))

        # compute std from bootstrapped samples
        predicted_boot_pwr = _compute_dist_from_boot(intercepts, slopes, freq)
        predicted_std_pwr = np.std(predicted_boot_pwr)

        # store the significant frequencies
        if pwr > predicted_mean_pwr + 4 * predicted_std_pwr:
            significant_freqs.append(freq)

    return significant_freqs
