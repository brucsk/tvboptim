from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
import jax.numpy as jnp
from typing import Tuple

class ReducedWongWangEIB(AbstractDynamics):
    """Two-population Reduced Wong-Wang model with E-I balance support"""

    STATE_NAMES = ('S_e', 'S_i')
    INITIAL_STATE = (0.001, 0.001)
    AUXILIARY_NAMES = ('H_e', 'H_i')
    g_prop = 0  # awake

    DEFAULT_PARAMS = Bunch(
        # Excitatory population parameters
        a_e=310.0,         # Input gain parameter
        b_e=125.0,         # Input shift parameter [Hz]
        d_e=0.160,         # Input scaling parameter [s]
        gamma_e=0.641/1000,  # Kinetic parameter
        tau_e=100.0,       # NMDA decay time constant [ms]
        w_p   = 1.4 * (1 - 0.2*g_prop),           # Excitatory recurrence weight
        W_e=1.0,           # External input scaling weight

        # Inhibitory population parameters
        a_i=615.0,         # Input gain parameter
        b_i=177.0,         # Input shift parameter [Hz]
        d_i=0.087,         # Input scaling parameter [s]
        gamma_i=1.0/1000,  # Kinetic parameter
        tau_i = 10  * (1 + 0.8*g_prop),        # NMDA decay time constant [ms]
        W_i=0.7,           # External input scaling weight

        # Synaptic weights
        J_N=0.15,          # NMDA current [nA]
        J_i   = 1.0 * (1 + 0.6*g_prop),          # Inhibitory synaptic weight

        # External inputs
        I_o=0.382,         # Background input current
        I_ext=0.0,         # External stimulation current

        # Coupling parameters
        lamda=1.0,         # Lambda: inhibitory coupling scaling

    )

# Primary GABA effect
#J_i = 1.0 → 1.4
# Slower inhibition
#tau_i = 10.0 → 18.0
# Slightly reduced excitation
#w_p = 1.4 → 1.2

    COUPLING_INPUTS = {
        'coupling': 2,  # Long-range excitation and Feedforward inhibition
    }

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute two-population Wong-Wang dynamics with dual coupling."""

        # Unpack state variables
        S_e = state[0]  # Excitatory synaptic gating
        S_i = state[1]  # Inhibitory synaptic gating

        # Unpack coupling inputs
        c_lre = params.J_N * coupling.coupling[0]  # Long-range excitation
        c_ffi = params.J_N * coupling.coupling[1]  # Feedforward inhibition

        # Excitatory population input
        J_N_S_e = params.J_N * S_e
        x_e_pre = (params.w_p * J_N_S_e - params.J_i * S_i +
                   params.W_e * params.I_o + c_lre + params.I_ext)

        # Excitatory transfer function
        x_e = params.a_e * x_e_pre - params.b_e
        H_e = x_e / (1.0 - jnp.exp(-params.d_e * x_e))

        # Excitatory dynamics
        dS_e_dt = -(S_e / params.tau_e) + (1.0 - S_e) * H_e * params.gamma_e

        # Inhibitory population input
        x_i_pre = J_N_S_e - S_i + params.W_i * params.I_o + params.lamda * c_ffi

        # Inhibitory transfer function
        x_i = params.a_i * x_i_pre - params.b_i
        H_i = x_i / (1.0 - jnp.exp(-params.d_i * x_i))

        # Inhibitory dynamics
        dS_i_dt = -(S_i / params.tau_i) + H_i * params.gamma_i

        # Package results
        derivatives = jnp.array([dS_e_dt, dS_i_dt])
        auxiliaries = jnp.array([H_e, H_i])

        return derivatives, auxiliaries