# Usage
This guide aims to give a minimal working example to generate samples using this MCMC Library.
While the algorithms in this library can be used for any kind of MCMC Sampling, it is especially built toe be efficient for distributions, where the realizations are infinite-dimensional objects like distributions on function spaces.
For examples and a proper introduction of the mathematical background in this library we recommend reading [1].
Throughout this usage guide we will look at the toy example where we want to sample from a multivariate Gaussian $\mathcal N(m, \Sigma)$.

## Math background
Nonetheless we will give a quick introduction of the notation used in [1].
In general, we are interested in sampling from an arbitrary distribution $\mu$.
This distribution is assumed to be absolutely continuous w.r.t. to a gaussian reference measure $\mu_0$.
We assume to know the Radon-Nikodym derivative

$$
\frac{d\mu}{d\mu_0} \propto \exp(-\Phi(u)),
$$

where $\Phi$ is called *potential*.
The reference measure is assumed to be a zero-mean Gaussian with covariance operator $C$, i.e.
$\mu_0 = \mathcal N(0, C)$. The reference measure can often be chosen as the prior.

## Model description
As a user you need to define the reference measure $\mu_0$ and the potential $\Phi$ to for the library to use. This is done via the [`MCMCModel`][ls_mcmc.model.MCMCModel] or [`DifferentiableMCMCModel`][ls_mcmc.model.DifferentiableMCMCModel] classes.
For our toy model we will consider the reference Gaussian measure $\mathcal N(0, I)$.
The potential is then given by

$$
\Phi(u) = \frac{1}{2} \langle u-m, \Sigma^{-1}(m-u)\rangle - \frac{1}{2} \langle u,u\rangle
$$

$$
\nabla\Phi(u) = \Sigma^{-1}(u-m)-u.
$$

For the covariance operator, we only need to evaluate it's action on a state $u$.
More precisely the action of $C^{\frac{1}{2}}$ (and of $C$ for MALA aswell).
The covariance operator is implemented via it's action in [`compute_preconditioner_sqrt_action`][ls_mcmc.model.MCMCModel.compute_preconditioner_sqrt_action]

Implemented as a model this would read:
```py
import numpy as np
from ls_mcmc import model

class GaussModel(model.DifferentiableMCMCModel):
    r"""Gaussian target with given mean and covariance, exposing $\Phi(x)$ and $\nabla\Phi(x)$"""

    def __init__(self, mean, cov) -> None:
        mean = np.asarray(mean)
        cov = np.asarray(cov)
        self._dim = mean.shape[0]
        self._reference_point = np.zeros(self._dim)
        self._preconditioner_sqrt_matrix = np.identity(self._dim) # covariance operator
        self._mean = mean
        self._cov = cov
        self._prec = np.linalg.inv(cov)

    def evaluate_potential(self, state: np.ndarray) -> float:
        potential = (
            1 / 2 * np.dot(state - self._mean, self._prec @ (state - self._mean))
            - 1 / 2 * state @ state
        )
        return potential

    def evaluate_gradient_of_potential(self, state: np.ndarray) -> np.ndarray:
        return self._prec @ (state - self._mean) - state

    def compute_preconditioner_sqrt_action(self, state: np.ndarray) -> np.ndarray:
        action = self._preconditioner_sqrt_matrix @ state
        return action

    @property
    def reference_point(self) -> np.ndarray:
        return self._reference_point

```

## Sampling
To start the sampling process, we need 4 other components.
- A choice of a [`MCMCAlgorithm`][ls_mcmc.algorithms.MCMCAlgorithm].
- A storage for the samples via [`MCMCStorage`][ls_mcmc.storage.MCMCStorage] that stores the samples,
this library provides a simple in memory storage and one storage via Zarr that can be used to store samples to the disk automatically.
- A list of [`MCMCOutput`][ls_mcmc.output.MCMCOutput]s for logging purpose, e.g. acceptance rate, component mean etc.
- An instance of a [`MCMCLogger`][ls_mcmc.logging.MCMCLogger] that deals with the actual logging.

A minimal working example could look like

```py
from ls_mcmc import algorithms, logging, output, sampling, storage

model = GaussModel(np.array([0, 0]), np.array([[1,0],[0,1]])) # N(0, I) 2D-Gaussian

algorithm = algorithms.pCNAlgorithm(model=model, step_width=0.1)
sample_storage = storage.NumpyStorage()
logger = logging.MCMCLogger(logging.LoggerSetings()) # Only print output by default
outputs = [output.Acceptance()]
sampler = sampling.Sampler(algorithm, sample_storage, outputs, logger)

sampler_settings = sampling.SamplerRunSettings(
    num_samples=5,
    initial_state=np.array([0, 0]),
)

samples, outputs = sampler.run(sampler_settings)
print(50*"-")
print(samples.values)
```
yields the output
```
| Iteration   | Time        | Acceptance  | 
-------------------------------------------
| 0.000e+00   | 8.845e-05   | 1.0         | 
| 1.000e+00   | 2.232e-04   | 1.0         | 
| 2.000e+00   | 2.806e-04   | 1.0         | 
| 3.000e+00   | 3.264e-04   | 1.0         | 
| 4.000e+00   | 3.655e-04   | 1.0         | 
--------------------------------------------------
[[ 0.          0.        ]
 [ 0.10058418 -0.10568389]
 [ 0.1442706  -0.49194583]
 [ 1.1297624   0.46249727]
 [-0.33447974 -0.22112121]]
```

## Disk Storage and graceful shutdown
For actual applications it is often not possible to rely on a purely In-Memory storage.
First, the available memory might not be enough and storing them on a hard drive might be the only possibility.
Secondly, if the sampler crashes or computation time runs out, one wants the samples to be stored, so they can be used later.
For this use we use [`Zarr`](https://github.com/zarr-developers/zarr-python).

The ZarrStorage interface can be used as follow:
```py
sample_storage = storage.ZarrStorage(Path("./storage"), chunk_size=10)
```
The `chunk_size` denotes how many samples are stored In-Memory before they are being flushed to the disk.
The [`ZarrStorage`][ls_mcmc.storage.ZarrStorage] can be used in the sampler the same way as the [`NumpyStorage`][ls_mcmc.storage.NumpyStorage].

The [`ZarrStorage`][ls_mcmc.storage.ZarrStorage] has the advantage, that it supports a graceful exit and tries to save all samples if an exit signal is received.

### Restarting a chain
If your program exits unexpectedly, and you used the [`ZarrStorage`][ls_mcmc.storage.ZarrStorage] the chain can be restarted.
If a termination signal (e.g. SIGTERM or SIGINT) is received, the sampler tries to store all samples that are not yet stored to the disk, as well as the state of the sampler including rng.
Restarting a chain can look is similar to the initial start.
```py
from ls_mcmc import algorithms, logging, output, sampling, storage

model = GaussModel(np.array([0, 0]), np.array([[1,0],[0,1]])) # N(0, I) 2D-Gaussian

algorithm = algorithms.pCNAlgorithm(model=model, step_width=0.1)
sample_storage = storage.ZarrStorage.initialize_from_disk(
    Path("./storage"), 10
)
logger = logging.MCMCLogger(logging.LoggerSetings()) # Only print output by default

samples, outputs = sampling.Sampler.resume_from_checkpoint(algorithm, sample_storage, logger)
```


## References
[1]: Cotter, Roberts, Stuart, White (2013). *MCMC Methods for Functions: Modifying Old
Algorithms to Make Them Faster.* Statistical Science 28(3).