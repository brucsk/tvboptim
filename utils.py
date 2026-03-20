import os
from pathlib import Path
import pandas as pd
import scipy.io as sio
import numpy as np
import jax.numpy as jnp
from typing import Callable
import optax
# Import from tvboptim
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.optim.callbacks import MultiCallback, DefaultPrintCallback, SavingLossCallback
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
# Observation functions
from tvboptim.observations.observation import compute_fc, fc_corr, rmse
from tvboptim.utils import set_cache_path, cache

# Set cache path for tvboptim
set_cache_path("ei_tuning")

def setup_directories(base_dir = "./"):
    if base_dir == None:
        base_dir = Path.cwd()
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    # Set path to save simulated BOLD signal
    path_simulated_bold = os.path.join(results_dir, "simulated_bold.npy")

    return results_dir, path_simulated_bold

def load_and_organize_bold(data_dir: str | None = None, 
                           cond0_filename: str | None = None, 
                           cond1_filename: str | None = None,
                           n_sub: int = 48, 
                           n_nodes: int = 68
                           ) -> np.ndarray:
    """
    Load and organize BOLD signal data for multiple subjects.
    
    Parameters
    ----------
    data_dir (str): Directory containing the BOLD signal files.
    cond0_filename (str): Filename for the control group BOLD signal.
    cond1_filename (str): Filename for the schizophrenic group BOLD signal.
    n_sub (int): Number of subjects.
    n_nodes (int): Number of nodes or regions.

    Returns
    -------
    np.ndarray: Organized BOLD signal data of shape (n_sub, n_time_points, n_regions, n_cond).
    """
    if data_dir is None or cond0_filename is None or cond1_filename is None:
        raise ValueError("data_dir, cond0_filename, and cond1_filename must be provided.")

    ## Load time-series bold data from two conditions, in this case, schizophrenic and control groups
    TS_CTR  = np.load(os.path.join(data_dir, cond0_filename))
    TS_SCZ  = np.load(os.path.join(data_dir, cond1_filename))

    ## Organize the data
    # Separate the participants by condition
    condition_0 = TS_CTR[0:n_sub, 0:n_nodes, :]  
    condition_1 = TS_SCZ[0:n_sub, 0:n_nodes, :]  

    # Determine the maximum number of participants in either condition (for alignment)
    max_participants = max(condition_0.shape[2], condition_1.shape[2])

    # Pad the smaller group to match the size of the larger one along the participant dimension
    condition_0_padded = np.pad(condition_0, ((0, 0), (0, 0), (0, max_participants - condition_0.shape[2])), mode='constant')
    condition_1_padded = np.pad(condition_1, ((0, 0), (0, 0), (0, max_participants - condition_1.shape[2])), mode='constant')

    # Stack the conditions along the fourth dimension
    new_array = np.stack((condition_0_padded, condition_1_padded), axis=3)
    
    return new_array

def load_structural_connectivity(sc_filepath: str | None = None,
                                 tl_filepath: str | None = None,
                                 centers_filepath: str | None = None) -> tuple[np.ndarray, pd.DataFrame, list]:
    """
    Load the structural connectivity matrix from a .npy file.
    
    Parameters
    ----------
    sc_filepath (str | None): Filepath for the structural connectivity matrix (.mat format expected).
    tl_filepath (str | None): Filepath for the tract lengths file.
    centers_filepath (str | None): Filepath for the region centers file.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame, list]: A tuple containing the normalized structural connectivity matrix, tract lengths DataFrame, and region labels.
    """
    if sc_filepath is None or tl_filepath is None or centers_filepath is None:
        raise ValueError("All filepaths (connectome, tract lengths, region centers) must be provided.")
    
    # Weights
    SCR = sio.loadmat(sc_filepath)['matrix']
    weights = SCR / np.max(SCR)

    # Delays
    lengths = pd.read_csv(tl_filepath)
    speed = 3.0
    delays = lengths / speed

    # Load region labels and coordinates
    df = pd.read_csv(
        centers_filepath,
        sep='\t',
        header=None,
        dtype={1: float, 2: float, 3: float},
        names=['label', 'x', 'y', 'z']
    )

    labels = df['label'].tolist()
    
    return weights, delays, labels

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

def lagged_fc_matrices(X: np.ndarray | jnp.ndarray, n_tau: int = 2, diag_zero: bool = True) -> np.ndarray:
    """ Compute lagged functional connectivity matrices from time series data.
    
    Parameters
    ----------
    X : np.ndarray | jnp.ndarray
        Z-scored BOLD time series data of shape (time_points, n_nodes).
    n_tau : int
        Number of time lags to compute (default is 2, which computes FC0 and FC1).
    diag_zero : bool
        Whether to set diagonal elements to zero (default is True).

    Returns
    -------
    Q : np.ndarray
        Lagged FC matrices of shape (n_tau, n_nodes, n_nodes).
    """
    # Transform to jax array for compatibility if input is numpy array
    if type(X) is np.ndarray:
        X = jnp.array(X)
    # Get dimensions
    n_T, n_nodes = X.shape
    # Lag (time-shifted) FC matrices
    #Q_emp = np.zeros([n_tau, n_nodes, n_nodes], dtype=float)
    # Remove mean in the time series
    centered_X = X - jnp.mean(X, axis=0)
    n_T_span = n_T - n_tau + 1
    
    def one_tau(i_tau):
        return jnp.tensordot(
            centered_X[0:n_T_span],
            centered_X[i_tau:n_T_span + i_tau],
            axes=(0, 0)
        )

    Q = jnp.stack([one_tau(i) for i in range(n_tau)], axis=0)
    Q = Q / (n_T_span - 1)

    if diag_zero:
        Q = Q * (1.0 - jnp.eye(n_nodes)[None, :, :])

    return Q

def make_loss(
    model_opt,
    bold_monitor_opt,
    Q0_emp,
    Q1_emp,
    target_fic,
    alpha_fc0=1.0,
    beta_fc1=2.0
) -> Callable:
    def loss(state):
        ts = model_opt(state)
        bold = bold_monitor_opt(ts)

        bold_signal = bold.data
        n_timepoints, n_nodes = bold_signal.shape[0], bold_signal.shape[-1]
        bold_signal = bold_signal.reshape(n_timepoints, n_nodes)
        bold_signal = bold_signal[5:, :]
        z_scored_bold = z_score_per_region(bold_signal)

        Qsim = lagged_fc_matrices(z_scored_bold, n_tau=2, diag_zero=True)
        Q0_sim, Q1_sim = Qsim[0], Qsim[1]

        loss_q0 = rmse(Q0_sim, Q0_emp)
        loss_q1 = rmse(Q1_sim, Q1_emp)

        mean_activity = jnp.mean(ts.data[-500:, 0, :], axis=0)
        activity_loss = jnp.mean((mean_activity - target_fic) ** 2)

        return alpha_fc0 * loss_q0 + beta_fc1 * loss_q1 + activity_loss

    return loss

# Define gradient optimization function 
@cache("gradient_optimization", redo=True)
def run_gradient_optimization(
    max_steps: int,
    learning_rate: float,
    loss: Callable,
    state_opt: Bunch,
    verbose: bool = True,
):
    """Run gradient-based optimization with optional LR scheduling.

    Parameters
    ----------
    max_steps : int
        Number of optimization steps.
    learning_rate : float
        Initial learning rate.
    verbose : bool
        Whether to print schedule information.
    """
    
    lr = learning_rate
    if verbose:
        print(f"LR: {learning_rate}")
    

    # Create optimizer
    optimizer = OptaxOptimizer(
        loss,
        optax.adamaxw(learning_rate=lr),
        callback=MultiCallback([DefaultPrintCallback(), SavingLossCallback()])
    )

    # Run optimization
    opt_state, opt_fitting_data = optimizer.run(state_opt, max_steps=max_steps)

    return opt_state, opt_fitting_data

# # Utils initially defined in the notebook below
# def setup_eval_model():
#     """Setup evaluation model for FC computation (called after initial simulation)."""
#     global model_eval, state_eval, _state
#     model_eval, state_eval = prepare(network, Heun(), t1=t1, dt=dt)
#     _state = copy.deepcopy(state_eval)

# def eval_fc(J_i, wLRE, wFFI):
#     """Evaluate FC for given parameters using a long simulation."""
#     _state.dynamics.J_i = J_i
#     _state.coupling.coupling.wLRE = wLRE
#     _state.coupling.coupling.wFFI = wFFI

#     # Run simulation
#     raw_result = model_eval(_state)

#     # Compute BOLD
#     bold_signal = bold_monitor(raw_result)

#     # Compute FC (skip initial transient)
#     fc = compute_fc(bold_signal, skip_t=20)
#     return fc