r"""Abstract Base classes that models should implement to be used for MCMC sampling.

The implementations of MCMC algorithms are based on (Cotter et al., 2013),
We sample from a measure $\mu$ that is defined via it's Radon-Nikodym derivative w.r.t. to a
Gaussian reference measure.

$$
\frac{d\mu}{d\mu_0} \propto \exp(-\Phi(u))
$$

where $\mu_0 = \mathcal{N}(0, C)$ is a centered Gaussian reference measure
with covariance operator $C$.
The likelihood is then given via the potential $\Phi(u)$.
Models should at least implement calculating the potential $\Phi(u)$ and the action
(i.e. matrix vector product in the finite dimensional setting)
of the covariance operator $C$ on a vector.
The covariace operator is also sometimes called preconditioner.

Classes:
    MCMCModel: Abstract base class for an MCMC model that evaluates the likelihood
    DifferentiableMCMCModel: Abstract base class for an MCMC model that evaluates the likelihood
        and also derivative information on the potential of the likelihood
References:
    Cotter, Roberts, Stuart, White (2013). *MCMC Methods for Functions: Modifying Old
    Algorithms to Make Them Faster.* Statistical Science 28(3).
"""

from abc import ABC, abstractmethod

import numpy as np


# ==================================================================================================
class MCMCModel(ABC):
    """Base interface for MCMC target models.

    Methods:
        evaluate_potential: Compute potential of likelihood
        compute_preconditioner_sqrt_action: Apply preconditioner on a state
        reference_point: Reference point that can be used to make sampling more effective
    """

    @abstractmethod
    def evaluate_potential(self, state: np.ndarray[tuple[int], np.dtype[np.floating]]) -> float:
        r"""Return potential energy $\Phi(u)$.

        Args:
            state: state $u$.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_preconditioner_sqrt_action(
        self, state: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """Apply square-root preconditioner to state.

        Args:
            state: state $u$.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def reference_point(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """Return a reference point."""
        raise NotImplementedError


class DifferentiableMCMCModel(MCMCModel):
    """Extension of MCMCModel with differentiable potential.

    Methods:
        evaluate_potential: Compute potential of likelihood
        compute_preconditioner_sqrt_action: Apply square root of preconditioner on a state
        reference_point: Reference point that can be used to make sampling more effective
        evaluate_gradient_of_potential: Compute the gradient of the potential of the likelihood
        compute_preconditioner_action: Apply preconditioner on a state
    """

    @abstractmethod
    def evaluate_gradient_of_potential(
        self, state: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        r"""Return gradient $\nabla\Phi(u)$.

        Args:
            state: state $u$.
        """
        raise NotImplementedError

    def compute_preconditioner_action(
        self, state: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """Apply full preconditioner.

        Args:
            state: state $u$.
        """
        sqrt_action = self.compute_preconditioner_sqrt_action(state)
        return self.compute_preconditioner_sqrt_action(sqrt_action)
