# Bayesian Inverse Problem
This example shows how one can sample from the posterior of a toy Bayesian Inverse Problem (BIP) with this library.

# Setup
Consider the 1D Poisson Equation:

$$
\begin{gather*}
    \text{div}(a(x) \nabla u(x)) &= 1 &\text{on $\Omega$} \\
    u(x) &= 0 &\text{on $\partial\Omega$} \\
\end{gather*}
$$


The PDE defines the PDE solution operator $S: L^2((0, 1)) \to H^1_0((0, 1)): a(x) \to u(x)$.
In most applications we don't have access to the whole PDE solution, but just some observed data at discrete points,
so we consider the observation operator: $G: L^2((0,1)) \to \mathbb R^d: a(x) \to u(x)\mid_{x_1, \ldots x_d}$
The inverse problem is now:

We assume we have some noisy observation $y=G(a^*) + \eta, \;\eta\sim N(0, \delta I_d)$ and want to know the distribution of $a$ given data $y$. Additionally, we assume a Gaussian prior distribution $\mu_0 \sim N(0, C)$, where the covariance $C$ is given by the [Matérn](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function) kernel.

The Bayes theorem then states, that the distribution of $a$ given $y$ is given by

$$
\frac{d\mu^y}{d\mu_0} \propto \exp(-\Phi(a)) = \exp(-\|G(a) - y\|_{(\delta I_d)^{-1}}).
$$

With this we can use MCMC to sample from $\mu^y$ given the forward operator.

For simplicity, we implemented the PDE solve using [scikit-fem](https://github.com/kinnala/scikit-fem),
of course you can also use any other PDE library.

The forward operator could then look like this:

```py
from skfem import MeshLine, Basis, ElementLineP1, BilinearForm, LinearForm, enforce, solve
from skfem.helpers import dot, grad
import numpy as np
from sklearn.gaussian_process.kernels import Matern

class Forward_operator:
    def __init__(self, points_pde_solve: np.ndarray, points_observation: np.ndarray) -> None:
        self._points_observation = points_observation
        self._points = points_pde_solve
        N = self._points.shape[0]
        self._elements = np.vstack([np.arange(N - 1), np.arange(1, N)])
        self._mesh = MeshLine(self._points.reshape(1, -1), self._elements)
        self._basis = Basis(self._mesh, ElementLineP1())

    def _evaluate(self, state: np.ndarray) -> np.ndarray:
        def A(x):
            flat = x.ravel()
            vals = np.interp(flat, self._points, state)
            return vals.reshape(x.shape)

        @BilinearForm
        def laplace(u, v, w):
            return A(w.x[0]) * dot(grad(u), grad(v))

        @LinearForm
        def rhs(v, _):
            return 1.0 * v

        A = laplace.assemble(self._basis)
        b = rhs.assemble(self._basis)

        # Dirichlet boundary conditions
        A, b = enforce(A, b, D=self._mesh.boundary_nodes())

        # solve the linear system
        x = solve(A, b)
        return x

    def __call__(self, state: np.ndarray) -> np.ndarray:
        pde_sol = self._evaluate(state)
        observed = np.interp(self._points_observation, self._points, pde_sol)
        return observed
```

We initialize the forward operator with a set of points on which we observe the solution `points_observation` and a set
of points that is used to build the FE-mesh `points_pde_solve`.

The last thing missing is the implementation of the [`MCMCModel`][ls_mcmc.model.MCMCModel] base class.
To force positivity on $a(x)$, we draw samples for $\log(a(x))$, which takes values on the whole
real line and use the exponential of the samples for the PDE.

```py
import ls_mcmc.model as model

class PoissonModel(model.MCMCModel):
    def __init__(
        self, points: np.ndarray, G: Forward_operator, data: np.ndarray, noise_std: float
    ) -> None:
        N = points.shape[0]
        self._data = data
        self._G = G
        self._reference_point = np.zeros((N,))
        self._points = points
        self._noise_std = noise_std
        self._kernel = Matern(length_scale=0.3, nu=2)
        self._covariance = self._kernel(self._points.reshape(-1, 1)) + 1e-10*np.eye(N) # spd
        self._preconditioner_sqrt_matrix = np.linalg.cholesky(self._covariance)

    def evaluate_potential(self, state: np.ndarray) -> float:
        potential = 1 / 2 * np.linalg.norm((self._G(np.exp(state)) - self._data)/self._noise_std) ** 2
        return potential

    def compute_preconditioner_sqrt_action(self, state: np.ndarray) -> np.ndarray:
        action = self._preconditioner_sqrt_matrix @ state
        return action

    @property
    def reference_point(self) -> np.ndarray:
        return self._reference_point
```

Since the noise is i.i.d. Gaussian on the observation mesh, the implemented MCMC methods are dependent on
the observation points, but due to the structure of pCN (and also MALA, which is not used in this tutorial)
the acceptance rate does not deteriorate, if we increase the dimension of the input function space, i.e.
a finer mesh for the PDE solve.


We can check the mesh independence numerically by sampling with different discretization.

```py
# check mesh independence
import matplotlib.pyplot as plt
import numpy as np

from ls_mcmc import algorithms, logging, output, sampling, storage
def coef(x):
    return 1.0 + 0.8 * np.sin(2 * np.pi * x)


np.random.seed(0)
acceptance_rates = []
N_pdes = (10, 20, 30, 50, 100, 200, 300, 400)
N_obs = 10
observation_points = np.linspace(0, 1, N_obs)
for N_pde in N_pdes:
    points = np.linspace(0, 1, N_pde)
    delta = 0.01
    ground_truth = coef(points)
    G = Forward_operator(points, observation_points)
    data = G(ground_truth) + np.random.normal(scale=delta, size=(N_obs,))
    posterior_model = PoissonModel(points, G, data, delta)

    acceptance_rate_output = output.Acceptance()
    outputs = (acceptance_rate_output, )
    logger_settings = logging.LoggerSettings()
    sampler_settings = sampling.SamplerRunSettings(
        num_samples=10000,
        initial_state=np.zeros((N_pde,)),
        print_interval=2000,
        store_interval=1,
    )

    sample_storage = storage.NumpyStorage()
    logger = logging.MCMCLogger(logger_settings)
    algorithm = algorithms.pCNAlgorithm(posterior_model, step_width=0.001)
    sampler = sampling.Sampler(algorithm, sample_storage, outputs, logger)
    samples, outputs = sampler.run(sampler_settings)
    acceptance_rates.append(outputs[0].value)
for N_pde, acceptance_rate in zip(N_pdes, acceptance_rates):
    print(f"Number of discretization points: {N_pde}, acceptance rate: {acceptance_rate}")
```

```
Number of discretization points: 10, acceptance rate: 0.6817999999999997
Number of discretization points: 20, acceptance rate: 0.6626999999999995
Number of discretization points: 30, acceptance rate: 0.6769000000000008
Number of discretization points: 50, acceptance rate: 0.6693000000000014
Number of discretization points: 100, acceptance rate: 0.6869000000000001
Number of discretization points: 200, acceptance rate: 0.6705999999999993
Number of discretization points: 300, acceptance rate: 0.6877000000000008
Number of discretization points: 400, acceptance rate: 0.690600000000001
```

The output indeed shows, that the acceptance rate is independent of the underlying discretization.

The full example code can be found under [examples/example_bip.ipynb](https://github.com/UQatKIT/LS-MCMC/tree/main/examples/example_bip.ipynb).
