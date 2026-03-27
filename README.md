### General description

This repository corresponds to an optimization project based on [TVBOptim](https://github.com/virtual-twin/tvboptim), a jax-powered optimization framework for neural network simulations. In particular, 
we use the Reduced Wong-Wang neural mass model to model a 68-region network according to the Desikan-Killiani (DK) human connectome, by fitting empirical functional connectivity matrices with gradient descent.

### Data
The time-series fMRI data was retrived from the [Zenodo repository](https://zenodo.org/records/10431855) supporting [Bryant et al., 2024](https://doi.org/10.1371/journal.pcbi.1012692). We used N = 48 subjects of the schizophrenia condition, and an equal amount of controls. 
