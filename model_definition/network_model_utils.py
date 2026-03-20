import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Callable, Any
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from model_definition.rww_dynamics import ReducedWongWangEIB
from model_definition.eib_linear_coupling import EIBLinearCoupling
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics import Network, solve, prepare
from tvboptim.experimental.network_dynamics.solvers import Heun, BoundedSolver
from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.experimental.network_dynamics.core.bunch import Bunch


from tvboptim.observations.tvb_monitors.bold import Bold

def build_network_model(weights: np.ndarray, labels: list[str], sigma: float = 0.01, verbose: bool = True) -> Network:  
    """
    Build a TVB network model using the provided structural connectivity weights and region labels.
    Parameters
    ----------
    weights (np.ndarray): Structural connectivity matrix (n_nodes x n_nodes).
    labels (list[str]): List of region labels corresponding to the nodes in the connectivity matrix.
    sigma (float): Standard deviation of the additive noise.
    verbose (bool): Whether to print verbose output.
    Returns
    -------
    Network: A TVB Network object configured with the specified dynamics, coupling, graph, and noise.
    """
    n_nodes = weights.shape[0]

    # Create network components
    graph = DenseGraph(weights, region_labels=labels)
    dynamics = ReducedWongWangEIB(J_i = jnp.ones((n_nodes)))

    # Initialize EIB coupling with dual weight matrices
    # wLRE and wFFI start as copies of structural connectivity
    coupling = EIBLinearCoupling(incoming_states=["S_e"])

    # Set the weight matrices to the proper shape based on structural connectivity
    # Both start as scaled versions of structural connectivity
    coupling.params.wLRE = jnp.ones((n_nodes, n_nodes)) #+ 0.8*fc_target  # [n_nodes, n_nodes]
    coupling.params.wFFI = jnp.ones((n_nodes, n_nodes)) #- 0.8*fc_target  # [n_nodes, n_nodes]

    # Small noise to break symmetry
    noise = AdditiveNoise(sigma=sigma, apply_to="S_e")

    # Assemble the network
    network = Network(
        dynamics=dynamics,
        coupling={'coupling': coupling},  # Both use same coupling but produce different outputs
        graph=graph,
        noise=noise
    )

    if verbose:
        print(f"Network created with {n_nodes} nodes")

    return network

def run_initial_simulation(t1: float, dt: float, network: Network, verbose: bool = True) -> tuple[Callable[..., Any], Bunch, NativeSolution]:
    """
    Run an initial simulation of the network to verify that it produces reasonable dynamics.
    Parameters
    ----------
    t1 (float): Simulation duration (ms)
    dt (float): Integration timestep (ms))
    network (Network): The TVB Network object to simulate.
    verbose (bool): Whether to print verbose output.
    Returns
    -------
    tuple: A tuple containing the compiled model, initial state, and simulation results.
    """
    # Prepare simulation: compile model and initialize state
    solver = BoundedSolver(Heun(), low=0.0, high=1.0)
    model, state = prepare(network, solver, t1=t1, dt=dt)

    # Run initial transient to reach quasi-stationary state
    if verbose:
        print("Running initial transient simulation...")

    result_init = jax.block_until_ready(model(state))

    if verbose:
        print(f"Initial simulation complete. Final S_e mean: {result_init.data[-1, 0, :].mean():.3f}")
        print(f"Initial simulation complete. Final S_i mean: {result_init.data[-1, 1, :].mean():.3f}")

    return model, state, result_init

def setup_bold_monitor(bold_TR: float = 2000.0, result_init: NativeSolution = None, verbose: bool = True) -> Bold:
    """
    Set up a BOLD monitor for the network simulation.
    Parameters
    ----------
    TR (float): Repetition time for BOLD sampling (ms).
    n_nodes (int): Number of nodes in the network.
    Returns
    -------
    """    
    # The BOLD period is bold_TR
    bold_monitor = Bold(
        period=bold_TR,           # BOLD sampling period (TR = 2000 ms)
        downsample_period=4.0,  # Intermediate downsampling matches dt
        voi=0,                  # Monitor first state variable (S_e)
        history=result_init     # Use initial state as warm start for BOLD history
    )

    if verbose:
        print("BOLD monitor initialized")

    return bold_monitor
 