import os
from pathlib import Path
import pandas as pd
import scipy.io as sio
import numpy as np
import jax.numpy as jnp
from typing import Callable
import optax
import copy 
import matplotlib.pyplot as plt
# Import from tvboptim
from tqdm import tqdm
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.optim.callbacks import MultiCallback, DefaultPrintCallback, SavingLossCallback
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.solvers import Heun
# Observation functions
from tvboptim.observations.observation import compute_fc, fc_corr, rmse
from tvboptim.experimental.network_dynamics import prepare
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

def setup_eval_model(t1, dt, network):
    """Setup evaluation model for FC computation (called after initial simulation)."""
    model_eval, state_eval = prepare(network, Heun(), t1=t1, dt=dt)
    state_copy = copy.deepcopy(state_eval)
    return model_eval, state_eval, state_copy

def compute_z_scored_bold(model, state, bold_monitor, n_transient=5):
    # Run simulation
    raw_result = model(state)

    # Compute BOLD
    bold_result = bold_monitor(raw_result)
    bold_signal = bold_result.data
    # Reshape from (time points, 1, regions) to (time points, regions) 
    n_timepoints, n_nodes = bold_signal.shape[0], bold_signal.shape[-1]
    bold_signal = bold_signal.reshape(n_timepoints, n_nodes)
    # Remove transient (n_transient time points = n_transient * dt ms, given dt=4 ms)
    bold_signal = bold_signal[n_transient:, :]
    z_scored_bold = z_score_per_region(bold_signal)
    return z_scored_bold

def eval_Q0_Q1(model, state, bold_monitor):
    """Evaluate FC0 and FC1 for given parameters using a long simulation."""
    z_scored_bold = compute_z_scored_bold(model=model, 
                                          state=state, 
                                          bold_monitor=bold_monitor)

    Q_sim = lagged_fc_matrices(z_scored_bold, n_tau=2, diag_zero=True)
    Q0_sim = Q_sim[0]  # Extract FC0 (lag-0)
    Q1_sim = Q_sim[1]  # Extract FC1 (lag-1)
    
    return Q0_sim, Q1_sim

def plot_gradient_descent_results(optimized_fit, optimized_state, participant_idx, condition_idx,
                                   Q0_emp, Q1_emp, Q0_pre_gd, Q1_pre_gd, Q0_sim, Q1_sim, Q0_corr_pre, Q1_corr_pre, Q0_corr_opt, Q1_corr_opt,
                                   conds=["CTR", "SCZ"]):
    # Extract loss values
    loss_values = optimized_fit["loss"].save
    n_steps = len(loss_values)

    # Define consistent color palette derived from cividis
    cividis_cmap = plt.cm.cividis
    cividis_colors = cividis_cmap(np.linspace(0, 1, 256))
    accent_blue = cividis_cmap(0.3)  # Dark blue from cividis
    accent_gold = cividis_cmap(0.85)  # Gold/yellow from cividis
    accent_mid = cividis_cmap(0.6)   # Mid-tone

    fig = plt.figure(figsize=(8.1, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.6)

    # Top left: Loss trajectory - use cividis-derived colors
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(loss_values, linewidth=2, color="k", alpha=0.9)
    ax1.scatter(0, loss_values[0], s=80, color=accent_blue, zorder=5)
    ax1.scatter(n_steps-1, loss_values.array[-1], s=80, color=accent_gold, zorder=5)
    ax1.set_xlabel('Optimization Step')
    ax1.set_ylabel('Combined Loss')
    ax1.set_title('Loss Convergence')
    ax1.grid(True, alpha=0.3)

    # Top middle: wFFI matrix - use cividis
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(optimized_state.coupling.coupling.wFFI, vmin=0, vmax=2, cmap='cividis')
    ax2.set_title('Optimized wFFI')
    ax2.set_xlabel('Source', labelpad=0.5)
    ax2.set_ylabel('Target', labelpad=0.5)
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # Top right: wLRE matrix - use cividis
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(optimized_state.coupling.coupling.wLRE, vmin=0, vmax=2, cmap='cividis')
    ax3.set_title('Optimized wLRE')
    ax3.set_xlabel('Source', labelpad=0.5)
    ax3.set_ylabel('Target', labelpad=0.5)
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Bottom row: FC comparison - use cividis
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(Q0_emp, vmin=-1, vmax=1.0, cmap='cividis')
    ax4.set_title('Target FC0')
    ax4.set_xlabel('Region', labelpad=0.5)
    ax4.set_ylabel('Region', labelpad=0.5)
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(Q0_pre_gd, vmin=-1, vmax=1.0, cmap='cividis')
    ax5.set_title(f'Pre-Opt FC0\nCorr: {Q0_corr_pre:.3f}')
    ax5.set_xlabel('Region', labelpad=0.5)
    ax5.set_ylabel('Region', labelpad=0.5)
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(Q0_sim, vmin=-1, vmax=1.0, cmap='cividis')
    ax6.set_title(f'Post-Opt FC0\nCorr: {Q0_corr_opt:.3f}')
    ax6.set_xlabel('Region', labelpad=0.5)
    ax6.set_ylabel('Region', labelpad=0.5)
    plt.colorbar(im6, ax=ax6, fraction=0.046)

    ax7 = fig.add_subplot(gs[2, 0])
    im7 = ax7.imshow(Q1_emp, vmin=-1, vmax=1.0, cmap='cividis')
    ax7.set_title('Target FC1')
    ax7.set_xlabel('Region', labelpad=0.5)
    ax7.set_ylabel('Region', labelpad=0.5)
    plt.colorbar(im7, ax=ax7, fraction=0.046)

    ax8 = fig.add_subplot(gs[2, 1])
    im8 = ax8.imshow(Q1_pre_gd, vmin=-1, vmax=1.0, cmap='cividis')
    ax8.set_title(f'Pre-Opt FC1\nCorr: {Q1_corr_pre:.3f}')
    ax8.set_xlabel('Region', labelpad=0.5)
    ax8.set_ylabel('Region', labelpad=0.5)
    plt.colorbar(im8, ax=ax8, fraction=0.046)

    ax9 = fig.add_subplot(gs[2, 2])
    im9 = ax9.imshow(Q1_sim, vmin=-1, vmax=1.0, cmap='cividis')
    ax9.set_title(f'Post-Opt FC1\nCorr: {Q1_corr_opt:.3f}')
    ax9.set_xlabel('Region', labelpad=0.5)
    ax9.set_ylabel('Region', labelpad=0.5)
    plt.colorbar(im9, ax=ax9, fraction=0.046)

    plt.suptitle(f'Gradient Descent Optimization Results: Subject {participant_idx}, Condition {conds[condition_idx]}', fontsize=14, y=0.94)

    #plt.show()
    return fig

def compute_quality_metrics(t1, bold_TR, transient_lim, n_nodes, n_sub, n_cond, 
                            Q0_emp_all, Q1_emp_all, Q0_pre_gd, Q1_pre_gd, 
                            model_opt, optimized_states, optimized_fits,  bold_monitor, 
                            result_dir = "./results", conds = ["CTR", "SCZ"], verbose=False):
    """Compute quality metrics (e.g., FC correlations) for pre- and post-optimization."""
    
    # Initialize variable to store z-scored BOLD signals for all participants and conditions after gradient descent optimization
    n_timepoints = int(t1 / bold_TR) - transient_lim  # Number of time points after removing transient
    z_scored_gd = np.zeros((n_sub, n_timepoints, n_nodes, n_cond))

    # Initialize variables to store pre-optimization quality metrics for all participants and conditions
    Q0_corr_pre = np.empty((n_sub, n_cond))
    Q0_rmse_pre = np.empty((n_sub, n_cond))
    Q1_corr_pre = np.empty((n_sub, n_cond))
    Q1_rmse_pre = np.empty((n_sub, n_cond))

    # Initialize simulated lagged FC matrices 
    Q0_sim = np.empty((n_sub, n_nodes, n_nodes, n_cond))
    Q1_sim = np.empty((n_sub, n_nodes, n_nodes, n_cond))

    # Initialize variables to store post-optimization quality metrics for all participants and conditions
    Q0_corr_opt = np.empty((n_sub, n_cond))
    Q0_rmse_opt = np.empty((n_sub, n_cond))
    Q1_corr_opt = np.empty((n_sub, n_cond))
    Q1_rmse_opt = np.empty((n_sub, n_cond))

    # Initialize dataframe to store all results
    results_df = pd.DataFrame(columns=["Participant", "Condition", "Q0_Corr_Pre", "Q0_RMSE_Pre", "Q1_Corr_Pre", "Q1_RMSE_Pre",
                                        "Q0_Corr_Opt", "Q0_RMSE_Opt", "Q1_Corr_Opt", "Q1_RMSE_Opt"])
    
    participant_range = range(n_sub)
    cond_range = range(n_cond)

    for participant_idx in tqdm(participant_range, desc="Participants", leave=True):
        for condition_idx in tqdm(cond_range, desc="Conditions", leave=False):
            if verbose:
                print(f"Participant {participant_idx}, Condition {conds[condition_idx]}\n", "-"*14)

            # Get FC0, FC1 for the current participant and condition
            Q0_emp = Q0_emp_all[participant_idx, :, :, condition_idx]  
            Q1_emp = Q1_emp_all[participant_idx, :, :, condition_idx]

            # Compute pre-optimization quality metrics
            Q0_corr_pre[participant_idx, condition_idx] = fc_corr(Q0_pre_gd, Q0_emp)
            Q0_rmse_pre[participant_idx, condition_idx] = jnp.sqrt(jnp.mean((Q0_pre_gd - Q0_emp)**2))
            Q1_corr_pre[participant_idx, condition_idx] = fc_corr(Q1_pre_gd, Q1_emp)
            Q1_rmse_pre[participant_idx, condition_idx] = jnp.sqrt(jnp.mean((Q1_pre_gd - Q1_emp)**2))

            # Compute z-scored BOLD signals after gradient descent optimization 
            z_scored_gd[participant_idx, :, :, condition_idx] = compute_z_scored_bold(
            model_opt,
            optimized_states[participant_idx, condition_idx],
            bold_monitor
            )
            
            # Compute simulated lagged FC matrices
            Q_sim = lagged_fc_matrices(z_scored_gd[participant_idx, :, :, condition_idx], n_tau=2, diag_zero=True)
            Q0_sim[participant_idx, :, :, condition_idx] = Q_sim[0]  # Simulated FC0
            Q1_sim[participant_idx, :, :, condition_idx] = Q_sim[1]  # Simulated FC1

            # Compute post-optimization quality metrics
            Q0_corr_opt[participant_idx, condition_idx] = fc_corr(Q0_sim[participant_idx, :, :, condition_idx], Q0_emp)
            Q1_corr_opt[participant_idx, condition_idx] = fc_corr(Q1_sim[participant_idx, :, :, condition_idx], Q1_emp)
            Q0_rmse_opt[participant_idx, condition_idx] = jnp.sqrt(jnp.mean((Q0_sim[participant_idx, :, :, condition_idx] - Q0_emp)**2))
            Q1_rmse_opt[participant_idx, condition_idx] = jnp.sqrt(jnp.mean((Q1_sim[participant_idx, :, :, condition_idx] - Q1_emp)**2))

            if verbose:
                print(f"\nOptimization Results for FC0:")
                print(f"  Pre-Optimization  - Correlation: {Q0_corr_pre[participant_idx, condition_idx]:.4f}, RMSE: {Q0_rmse_pre[participant_idx, condition_idx]:.4f}")

                print(f"  Post-Optimization - Correlation: {Q0_corr_opt[participant_idx, condition_idx]:.4f}, RMSE: {Q0_rmse_opt[participant_idx, condition_idx]:.4f}")
                print(f"  Improvement: Δcorr = {Q0_corr_opt[participant_idx, condition_idx] - Q0_corr_pre[participant_idx, condition_idx]:+.4f}, ΔRMSE = {Q0_rmse_opt[participant_idx, condition_idx] - Q0_rmse_pre[participant_idx, condition_idx]:+.4f}")

                print(f"\nOptimization Results for FC1:")
                print(f"  Pre-Optimization  - Correlation: {Q1_corr_pre[participant_idx, condition_idx]:.4f}, RMSE: {Q1_rmse_pre[participant_idx, condition_idx]:.4f}")
                print(f"  Post-Optimization - Correlation: {Q1_corr_opt[participant_idx, condition_idx]:.4f}, RMSE: {Q1_rmse_opt[participant_idx, condition_idx]:.4f}")
                print(f"  Improvement: Δcorr = {Q1_corr_opt[participant_idx, condition_idx] - Q1_corr_pre[participant_idx, condition_idx]:+.4f}, ΔRMSE = {Q1_rmse_opt[participant_idx, condition_idx] - Q1_rmse_pre[participant_idx, condition_idx]:+.4f}")

            # Create new row to insert into results dataframe
            data_temp = pd.DataFrame({
                "Participant": [participant_idx],
                "Condition": [conds[condition_idx]],
                "Q0_Corr_Pre": [Q0_corr_pre[participant_idx, condition_idx]],
                "Q0_RMSE_Pre": [Q0_rmse_pre[participant_idx, condition_idx]],
                "Q1_Corr_Pre": [Q1_corr_pre[participant_idx, condition_idx]],
                "Q1_RMSE_Pre": [Q1_rmse_pre[participant_idx, condition_idx]],
                "Q0_Corr_Opt": [Q0_corr_opt[participant_idx, condition_idx]],
                "Q0_RMSE_Opt": [Q0_rmse_opt[participant_idx, condition_idx]],
                "Q1_Corr_Opt": [Q1_corr_opt[participant_idx, condition_idx]],
                "Q1_RMSE_Opt": [Q1_rmse_opt[participant_idx, condition_idx]]
            })

            # Save row corresponding to one participant and condition into results dataframe
            results_df = pd.concat([results_df, data_temp], ignore_index=True)

            fig = plot_gradient_descent_results(optimized_fit= optimized_fits[participant_idx, condition_idx],
                                optimized_state= optimized_states[participant_idx, condition_idx],
                                participant_idx = participant_idx,
                                condition_idx = condition_idx,
                                Q0_emp=Q0_emp,
                                Q1_emp=Q1_emp,
                                Q0_pre_gd=Q0_pre_gd,
                                Q1_pre_gd=Q1_pre_gd,
                                Q0_sim=Q0_sim[participant_idx, :, :, condition_idx],
                                Q1_sim=Q1_sim[participant_idx, :, :, condition_idx],
                                Q0_corr_pre=Q0_corr_pre[participant_idx, condition_idx],
                                Q1_corr_pre=Q1_corr_pre[participant_idx, condition_idx],
                                Q0_corr_opt=Q0_corr_opt[participant_idx, condition_idx],
                                Q1_corr_opt=Q1_corr_opt[participant_idx, condition_idx]
                                )
            
            plt.savefig(os.path.join(result_dir, f"gd_results_participant_{participant_idx}_condition_{conds[condition_idx]}.png"), dpi=300)
            plt.close(fig)

    # Save dataframe into csv file
    results_csv_path = os.path.join(result_dir, "optimization_quality_metrics.csv")
    results_df.to_csv(results_csv_path, index=False)
    if verbose:
        print(f"\nAll quality metrics saved to {results_csv_path}")

    # Save z-scored BOLD signals after gradient descent optimization into a .npy file
    z_scored_gd_path = os.path.join(result_dir, "z_scored_bold_gd.npy")
    np.save(z_scored_gd_path, z_scored_gd)
    if verbose:
        print(f"Z-scored BOLD signals after gradient descent optimization saved to {z_scored_gd_path}")
    
    # Save simulated lagged FC matrices into .npy files
    Q0_sim_path = os.path.join(result_dir, "Q0_sim.npy")
    Q1_sim_path = os.path.join(result_dir, "Q1_sim.npy")
    np.save(Q0_sim_path, Q0_sim)
    np.save(Q1_sim_path, Q1_sim)
    if verbose:
        print(f"Simulated FC0 matrices saved to {Q0_sim_path}")
        print(f"Simulated FC1 matrices saved to {Q1_sim_path}")