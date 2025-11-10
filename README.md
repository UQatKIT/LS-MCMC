![CI](https://img.shields.io/github/actions/workflow/status/UQatKIT/LS-MCMC/ci.yml?label=CI)
![Docs](https://img.shields.io/github/actions/workflow/status/UQatKIT/LS-MCMC/docs.yml?label=Docs)
![License](https://img.shields.io/github/license/UQatKIT/LS-MCMC)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

# Large Scale MCMC Library
A lightweight, modular library developed for large-scale Markov Chain Monte Carlo.
In particular, it implements algorithms that are infinite-dimensionally consistent to ensure discretization independent
acceptance probabilities.
The library is lightweight in the sense, that it tries to keep the overhead for the user as small as possible and relies heavily on modularization, so extensions can be easily implemented.
It focuses on large-scale problems by supporting disk-based storage and reusing computed results as much as possible.
It is most likely not the fastest MCMC library out there, but it is quick to use and supports most features that are essential for MCMC on function spaces.


### Key Features
- **pCN and MALA support** <br>
- **On-Disk sample storage** <br>
- **Graceful exit on errors** <br>


Markov-Chain-Monte-Carlo Methods aim to sample from an unknown probability measure $\mu$ that is known up to a normalization constant.
We assume the density is known w.r.t. a reference Gaussian measure $\mu_0 = N(0, C)$, i.e.

$$
\frac{d\mu}{d\mu_0} \propto L(u) = \exp(-\Phi(u)).
$$

In this case the user only needs to supply the covariance operator of the reference measure $C$ and the potential function $\Phi$.


## Installation and Development
Releases of this version can be downloaded from pip directly:

```sh
pip install ls-mcmc
```

This library relies on [uv](https://docs.astral.sh/uv/) for packaging.
To install a development environment, clone the Github repository and run

```sh
uv sync --all-groups
```

in the projects root folder.

## Documentation

### Usage
The usage section describes a basic setup for sampling from a distribution, configuring logging output, disk-storage and restarting of chains.

### API Reference
The API reference contains documentation for all parts of the library.

## Examples
Other examples can be found [here](https://github.com/UQatKIT/LS-MCMC/tree/main/examples).

## Acknowledgement and License
This package is developed by the [Uncertainty Quantification Research Group at KIT](https://www.scc.kit.edu/forschung/uq.php).
It is distributed under the MIT License.
