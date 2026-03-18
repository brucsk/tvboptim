import numpy as np
import jax.numpy as jnp

def z_score_per_region(bold_signal: np.ndarray | jnp.ndarray) -> jnp.ndarray:
    """
    Z-score the BOLD signal for each region independently.
    
    Parameters
    ----------
    bold_signal (jax.numpy.ndarray): BOLD signal, shape (time_points, n_regions).
    
    Returns
    ----------
    jax.numpy.ndarray: The z-scored BOLD signal with the same shape as input.
    """
    # Transform to jax array for compatibility if input is numpy array
    if type(bold_signal) is np.ndarray:
        bold_signal = jnp.array(bold_signal)

    # Compute mean and std for each region
    mean_per_region = jnp.mean(bold_signal, axis=0)
    std_per_region = jnp.std(bold_signal, axis=0, ddof=0)
    
    # Avoid division by zero
    std_per_region = jnp.where(std_per_region == 0, 1.0, std_per_region)
    
    # Z-score the signal
    z_scored_signal = (bold_signal - mean_per_region) / std_per_region
    
    return z_scored_signal

def zscore_check(x, axis=None, thres= 1e-10, verbose=False):
    """Check if the input array is z-scored (mean ~ 0 and std ~ 1) along the specified axis."""
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis, ddof=0)  # ddof=0 matches most z-score implementations
    if verbose:
        print(f"mean ~ 0? max|mean|={np.max(np.abs(mean)):.3e}")
        print(f"std ~ 1? max|std-1|={np.max(np.abs(std-1)):.3e}")
    if np.max(np.abs(mean)) > thres or np.max(np.abs(std-1)) > thres:
        if verbose:
            print(
            f"Signal is not z-scored: mean is not close to 0 or std is not close to 1 "
            f"(max|mean|={np.max(np.abs(mean)):.3e}); max|std-1|={np.max(np.abs(std-1)):.3e})"
                )
        return False 
    
    else: 
        if verbose:
            print(f"Signal is z-scored: mean is close to 0 and std is close to 1 (max|mean|={np.max(np.abs(mean)):.3e}); max|std-1|={np.max(np.abs(std-1)):.3e})")
        return True

def lagged_fc_matrices(X: np.ndarray | jnp.ndarray, n_tau: int = 2, diag_zero: bool = True, check_zscore: bool = False) -> np.ndarray:
    """ Compute lagged functional connectivity matrices from time series data.
    
    Parameters
    ----------
    X : np.ndarray | jnp.ndarray
        Z-scored BOLD time series data of shape (time_points, n_nodes).
    n_tau : int
        Number of time lags to compute (default is 2, which computes FC0 and FC1).
    diag_zero : bool
        Whether to set diagonal elements to zero (default is True).
    check_zscore : bool
        Whether to check if the input is z-scored (default is True).

    Returns
    -------
    Q_emp : np.ndarray
        Lagged FC matrices of shape (n_tau, n_nodes, n_nodes).
    """
    if check_zscore:
        # Check if input is z-scored (important to respect the formula for lagged FC)
        if zscore_check(X, axis=0) is False:
            raise ValueError("Input time series data must be z-scored (mean ~ 0 and std ~ 1) along the time axis.")
    # Transform to jax array for compatibility if input is numpy array
    if type(X) is np.ndarray:
        X = jnp.array(X)
    # Get dimensions
    n_T, n_nodes = X.shape
    # Lag (time-shifted) FC matrices
    Q_emp = np.zeros([n_tau, n_nodes, n_nodes], dtype=float)
    # Remove mean in the time series
    centered_X = X - X.mean(axis=0)
    # Calculate the lagged FC matrices
    n_T_span = n_T - n_tau + 1
    for i_tau in range(n_tau):
        Q_emp[i_tau] = np.tensordot(centered_X[0:n_T_span], \
                                    centered_X[i_tau:n_T_span+i_tau], \
                                    axes=(0,0))
    Q_emp /= float(n_T_span - 1)
    if diag_zero:
        for i_tau in range(n_tau):
            np.fill_diagonal(Q_emp[i_tau], 0.0)

    return Q_emp