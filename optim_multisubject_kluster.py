## Imports ====================
import cloudpickle as pickle
from datetime import datetime
import jax
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

from utils import *
from model_definition.network_model_utils import *
from tvboptim.types import Parameter, BoundedParameter

jax.config.update("jax_enable_x64", True)

## Set params ====================
# Set directory information
data_dir = "./"
cond0_filename = "TS_Control.npy"
cond1_filename = "TS_Schizo.npy"
result_dir = "./results/"

# Set dataset parameters
n_sub = 48
n_nodes = 68 # size of network for AAL90
conds = ['CTR', 'SCZ']
n_cond = len(conds) # number of conditions

# Simulation parameters
t1 = 314_000   # Simulation duration (ms) matching empirical data (=304_000) + transient time (~10_000 ms)
dt = 4.0      # Integration timestep (ms) matching original script
bold_TR = 2000.0 # BOLD sampling period (ms)
transient_lim = 5 # Number of time points to remove as transient (transient_lim * dt ms)
target_fic = 0.25  # FIC tuning parameter: Target excitatory activity level

# Gradient descent parameters
learning_rate = 0.0325
max_steps = 200

# Other parameters
n_tau = 2 # number of lags for lagged FC

## Load time-series BOLD data =====================
new_array = load_and_organize_bold(data_dir = data_dir, cond0_filename = cond0_filename, cond1_filename = cond1_filename,
                                   n_sub = n_sub, n_nodes = n_nodes)

## Compute time-lagged matrices for empirical data =====================
Q0_emp_all = np.zeros((n_sub, n_nodes, n_nodes, n_cond))  # shape: (n_sub, n_tau, n_nodes, n_nodes, n_cond)
Q1_emp_all = np.zeros((n_sub, n_nodes, n_nodes, n_cond))

for participant_idx in range(n_sub):
    for condition_idx in range(n_cond):
        # Get empirical time series of interest 
        ts = new_array[participant_idx,:,:,condition_idx]
        # Take the transpose for the lagged FC matrices computation
        X_emp = ts.T
    
        # Z-score the empirical time series per region
        z_scored_emp = z_score_per_region(X_emp)

        # Compute empirical lagged FC matrices
        Q_emp_single = lagged_fc_matrices(z_scored_emp, n_tau=n_tau, diag_zero=True)
        Q0_emp_single = Q_emp_single[0]  # FC0 (zero-lag)
        Q1_emp_single = Q_emp_single[1]  # FC1 (lag-1)

        Q0_emp_all[participant_idx, :, :, condition_idx] = Q0_emp_single
        Q1_emp_all[participant_idx, :, :, condition_idx] = Q1_emp_single
    
print("Empirical time series shape (time points x regions):", X_emp.shape)
print("Empirical FC0 shape (regions x regions):", Q0_emp_all.shape)
print("Empirical FC1 shape (regions x regions):", Q1_emp_all.shape)

## Load structural connectivity data =====================
sc_path = 'SC_EnigmadK68.mat'
tl_path = 'tract_lengths.csv'
centers_path = 'centers.txt'

weights, delays, labels = load_structural_connectivity(sc_filepath=sc_path, tl_filepath=tl_path, centers_filepath=centers_path)

## Build model to optimize =====================
# Build a single network model using the structural connectivity and region labels
network = build_network_model(weights=weights, labels=labels)

## Run initial simulation and set up BOLD monitor ======================
model, state, result_init = run_initial_simulation(t1=t1, dt=dt, network=network)
bold_monitor_opt = setup_bold_monitor(bold_TR = bold_TR, result_init = result_init)
network.update_history(result_init)
model_opt, state_opt, _ = run_initial_simulation(t1=t1, dt=dt, network = network, verbose=False)

## Set up evaluation model =======================
# Will be populated after initial simulation completes
model_eval, state_eval, _state = None, None, None
model_eval, state_eval, _state = setup_eval_model(t1=t1, dt=dt, network=network)
# Compute Q before gradient descent optimization
print("Computing pre-gradient descent functional connectivity...")
Q0_pre_gd, Q1_pre_gd = eval_Q0_Q1(
    model_eval, state_eval, bold_monitor_opt
)

## Main pipeline ========================
# Test for scaling up - later substitute with n_sub and n_cond defined at the beggining of script
n_sub_test = 3
n_cond_test = 2

# Define ranges for participants and conditions for testing
participant_range_test= range(n_sub_test)
cond_range_test = range(n_cond_test)

optimized_states_test = np.empty((n_sub_test, n_cond_test), dtype=object)
optimized_fits_test = np.empty((n_sub_test, n_cond_test), dtype=object)

for participant_idx in participant_range_test:
    for condition_idx in cond_range_test:
        print(f"Testing participant {participant_idx}, condition {condition_idx}")

        Q0_emp = Q0_emp_all[participant_idx, :, :, condition_idx]  # FC0 for the first participant and first condition
        Q1_emp = Q1_emp_all[participant_idx, :, :, condition_idx]

        print(f"Empirical FC0 shape: {Q0_emp.shape}, Empirical FC1 shape: {Q1_emp.shape}")

        loss = make_loss(
            model_opt=model_opt,
            bold_monitor_opt=bold_monitor_opt,
            Q0_emp=Q0_emp,
            Q1_emp=Q1_emp,
            target_fic=target_fic,
            alpha_fc0=1.0,
            beta_fc1=2.0
        )

        # Evaluate initial loss
        initial_loss = loss(state_opt)
        print(f"Initial loss: {initial_loss:.4f}")

        # Mark parameters for optimization (J_i, wLRE, wFFI) with appropriate constraints
        state_opt.dynamics.J_i = Parameter(state_opt.dynamics.J_i)
        state_opt.coupling.coupling.wLRE = BoundedParameter(jnp.ones((n_nodes, n_nodes)), low=0.0, high=jnp.inf)
        state_opt.coupling.coupling.wFFI = BoundedParameter(jnp.ones((n_nodes, n_nodes)), low=0.0, high=jnp.inf)

        optimized_state_temp, optimized_fit_temp = run_gradient_optimization(max_steps, learning_rate, loss, state_opt)
        optimized_states_test[participant_idx, condition_idx] = optimized_state_temp
        optimized_fits_test[participant_idx, condition_idx] = optimized_fit_temp

## Save results =====================
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
pikl_name = f"part2_saved_state_{timestamp}.pkl"
save_path = Path(os.path.join(result_dir, pikl_name))

to_save = {
    "model_eval": model_eval,
    "state_eval": state_eval,
    "model_opt": model_opt,
    "optimized_states": optimized_states_test,
    "optimized_fits": optimized_fits_test,
}

with save_path.open("wb") as f:
    pickle.dump(to_save, f)

print(f"Saved variables to {save_path.resolve()}")

## Compute and save quality metrics and plots =====================
new_result_dir = f"./results/{learning_rate}_{max_steps}/"
compute_quality_metrics(t1, bold_TR, transient_lim, n_nodes, n_sub_test, n_cond_test, 
                            Q0_emp_all, Q1_emp_all, Q0_pre_gd, Q1_pre_gd, 
                            model_opt, optimized_states_test, optimized_fits_test,  bold_monitor_opt, 
                            result_dir = new_result_dir, conds = ["CTR", "SCZ"], verbose=False)