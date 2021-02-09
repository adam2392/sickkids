import numpy as np


def rescale_modes(snapshots, atilde, dmd):
    """Helper function to rescale and recompute DMD modes.

    Re-scales the DMD modes by the singular values from
    the data matrix SVD.

    Parameters
    ----------
    snapshots : np.ndarray
        The original snapshots matrix.
    atilde : np.ndarray
        The fitted rank-truncated linear operator from DMD.
    dmd : instance of pydmd.DMDBase
        ``dmd`` should already have been fitted.

    Returns
    -------
    modes : np.ndarray
        The re-scaled DMD modes.
    """
    Y = snapshots[:, 1:]

    # compute the SVD
    U, s, V = dmd._compute_svd(snapshots[:, :-1], dmd.svd_rank)

    # compute Ahat from Atilde
    sigma_inv_sqrt = np.diag(np.power(s.copy(), -0.5))
    sigma_sqrt = np.diag(np.power(s.copy(), 0.5))
    Ahat = sigma_inv_sqrt.dot(atilde).dot(sigma_sqrt)

    # compute eigenvector decomposition of Atilde
    lowrank_eigenvalues, lowrank_eigenvectors = np.linalg.eig(Ahat)

    # check norms
    #     print(np.linalg.norm(lowrank_eigenvectors, axis=0))

    # rescale eigenvectors by singular values
    lowrank_eigenvectors = sigma_sqrt.dot(lowrank_eigenvectors)

    #     print(Ahat.shape, lowrank_eigenvectors.shape)
    #     print(np.linalg.norm(lowrank_eigenvectors, axis=0))

    # compute DMD modes again
    modes = (Y.dot(V).dot(np.diag(np.reciprocal(s)))).dot(lowrank_eigenvectors)
    modes = modes[: dmd._snapshots.shape[0], :]
    return modes
