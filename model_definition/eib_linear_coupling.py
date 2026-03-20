from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
import jax.numpy as jnp

class EIBLinearCoupling(InstantaneousCoupling):
    """EIB Linear coupling with separate excitatory and inhibitory weight matrices.

    This coupling produces two outputs:
        c_lre: Long-range excitation (wLRE * S_e)
        c_ffi: Feedforward inhibition (wFFI * S_e)

    Both couplings are driven by the excitatory activity (S_e) from other regions.
    """

    N_OUTPUT_STATES = 2  # Produces two coupling outputs

    DEFAULT_PARAMS = Bunch(
        wLRE = 1.0,  # Long-range excitation weight matrix
        wFFI = 1.0,  # Feedforward inhibition weight matrix
    )

    def pre(
        self,
        incoming_states: jnp.ndarray,
        local_states: jnp.ndarray,
        params: Bunch
    ) -> jnp.ndarray:
        """Pre-synaptic transformation: multiply S_e with wLRE and wFFI."""
        # incoming_states[0] is S_e from all source nodes
        S_e = incoming_states[0]  # [n_target, n_source]
        # Apply weights: element-wise multiply S_e with each weight matrix
        # params.wLRE and params.wFFI have shape [n_nodes, n_nodes]
        c_lre = S_e * params.wLRE  # [n_target, n_source]
        c_ffi = S_e * params.wFFI  # [n_target, n_source]

        # Stack into [2, n_target, n_source]
        return jnp.stack([c_lre, c_ffi], axis=0)

    def post(
        self,
        summed_inputs: jnp.ndarray,
        local_states: jnp.ndarray,
        params: Bunch
    ) -> jnp.ndarray:
        """Post-synaptic transformation: pass through without scaling."""
        return summed_inputs
    