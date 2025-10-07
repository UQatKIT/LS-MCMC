"""Implementation of core MCMC algorithms.

Classes:
    MCMCAlgorith: abstract base class used for all MCMCAlgorithms
    pCNAlgorithm: Implementation of the pCN algorithm
    MALAAlgorithm: Implementation of the MALA (pCNL) algorithm
"""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from . import model


# ==================================================================================================
@dataclass
class _CachedArgs:
    """Base class for argument caches used in MCMCAlgorithm."""

    def clear(self) -> None:
        """Clear all cached arguments."""
        for field in dataclasses.fields(self):
            self.__setattr__(field.name, None)


class MCMCAlgorithm(ABC):
    """Abstract base class for MCMC algorithms.

    Subclasses must implement proposal generation and the acceptance probability.
    This class handles the generic propose/accept-reject step and caching handoff.

    Methods:
        compute_step: Compute one step of MCMC
    """

    def __init__(self, model: model.MCMCModel, step_width: float) -> None:
        r"""Initialize MCMC algorithm.

        Args:
            model (MCMCModel): Provides potential $\Phi$ and preconditioner actions
                encoding the covariance operator $C$ and reference point $u_0$.
            step_width (float): The step scaling $\delta \in (0,2)$. Smaller values
                give higher acceptance but slower exploration; larger values explore faster until
                acceptance deteriorates.
        """
        if step_width <= 0:
            raise ValueError("Step width must be greater than zero.")
        self._step_width = step_width
        self._model = model
        self._cached_args_current: _CachedArgs | None = None  # derived class should set this
        self._cached_args_proposal: _CachedArgs | None = None  # derived class should set this

    def compute_step(
        self, current_state: np.ndarray[tuple[int], np.dtype[np.floating]], rng: np.random.Generator
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], bool]:
        """Advance the Markov chain by one step.

        1. Create a proposal from the current state.
        2. Perform accept-reject according to the subclass criterion.
        3. Swap caches on accept and clear the proposal cache.

        Args:
            current_state (np.ndarray): Current chain state.
            rng (np.random.Generator): Random number generator.

        Returns:
            tuple[np.ndarray, bool]: New state and a flag indicating acceptance.
        """
        # Create a proposal
        proposal = self._create_proposal(current_state, rng)
        new_state, accepted = self._perform_accept_reject(current_state, proposal, rng)
        if accepted:
            self._cached_args_current, self._cached_args_proposal = (
                self._cached_args_proposal,
                self._cached_args_current,
            )
        self._cached_args_proposal.clear()
        return new_state, accepted

    def _perform_accept_reject(
        self,
        current_state: np.ndarray[tuple[int], np.dtype[np.floating]],
        proposal: np.ndarray[tuple[int], np.dtype[np.floating]],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], bool]:
        """Perform a standard Metropolis-Hastings accept/reject step.

        Args:
            current_state (np.ndarray): Current state.
            proposal (np.ndarray): Proposed state.
            rng (np.random.Generator): Random number generator.

        Returns:
            tuple[np.ndarray, bool]: Accepted state (either proposal or current) and flag.
        """
        assert current_state.shape == proposal.shape, (
            f"Current state and proposal must have the same shape, but they have shapes"
            f"{current_state.shape} and {proposal.shape}, respectively."
        )
        acceptance_probability = self._evaluate_acceptance_probability(current_state, proposal)
        random_draw = rng.uniform()
        if random_draw < acceptance_probability:
            new_state = proposal
            accepted = True
        else:
            new_state = current_state
            accepted = False
        return new_state, accepted

    @abstractmethod
    def _create_proposal(
        self, state: np.ndarray[tuple[int], np.dtype[np.floating]], rng: np.random.Generator
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """Create a proposal given the current state and RNG.

        Args:
            state (np.ndarray): Current state.
            rng (np.random.Generator): Random number generator.

        Returns:
            np.ndarray: Proposed next state with same shape as ``state``.
        """
        raise NotImplementedError

    @abstractmethod
    def _evaluate_acceptance_probability(
        self,
        current_state: np.ndarray[tuple[int], np.dtype[np.floating]],
        proposal: np.ndarray[tuple[int], np.dtype[np.floating]],
    ) -> float:
        """Evaluate acceptance probability for a proposal in [0, 1]."""
        raise NotImplementedError


# ==================================================================================================
@dataclass
class _pCNCachedArgs(_CachedArgs):
    """Argument cache for pCN algorithm."""

    potential: float | None = None


class pCNAlgorithm(MCMCAlgorithm):
    r"""Preconditioned Crank-Nicolson (pCN) sampler.

    Implements the function-space pCN proposal for sampling from a target measure
    with (unnormalized) density proportional to $\exp(-\Phi(u))$ with respect to a
    Gaussian reference measure $\mu_0 = \mathcal N(u_0, C)$.

    Proposal (given current state $u$):

    $$
    v = u_0 + \frac{2-\delta}{2+\delta}\, (u - u_0) + \frac{\sqrt{8\delta}}{2+\delta}\,
    w, \qquad w \sim \mathcal N(0, C),
    $$

    where ``step_width`` corresponds to the parameter $\delta \in (0,2)$. The move
    preserves the prior (reference) measure and therefore yields a simple Metropolis-Hastings
    acceptance probability

    $$
    \alpha(u,v) = 1 \wedge \exp\big( -\Phi(v) + \Phi(u) \big).
    $$

    Methods:
        compute_step: Compute one step of MCMC

    Notes:
        The proposal leaves the Gaussian reference measure invariant; only the potential
        difference appears in the acceptance probability (no proposal density correction term).

    References:
        Cotter, Roberts, Stuart, White (2013). *MCMC Methods for Functions: Modifying Old
        Algorithms to Make Them Faster.* Statistical Science 28(3).
    """

    def __init__(self, model: model.MCMCModel, step_width: float) -> None:
        r"""Initialize pCN algorithm.

        Args:
            model (MCMCModel): Provides potential $\Phi$ and preconditioner actions
                encoding the covariance operator $C$ and reference point $u_0$.
            step_width (float): The proposal scaling $\delta \in (0,1)$. Smaller values
                give higher acceptance but slower exploration; larger values explore faster until
                acceptance deteriorates.
        """
        super().__init__(model, step_width)
        self._cached_args_current: _pCNCachedArgs = _pCNCachedArgs()
        self._cached_args_proposal: _pCNCachedArgs = _pCNCachedArgs()

    def _cache_args(
        self, potential_current: float | None = None, potential_proposal: float | None = None
    ) -> None:
        """Cache potential values for reuse if already computed.

        Args:
            potential_current (float | None): Potential at current state.
            potential_proposal (float | None): Potential at proposed state.
        """
        if potential_current:
            self._cached_args_current.potential = potential_current
        if potential_proposal:
            self._cached_args_proposal.potential = potential_proposal

    def _create_proposal(
        self, state: np.ndarray[tuple[int], np.dtype[np.floating]], rng: np.random.Generator
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """Generate a pCN proposal state.

        The proposal preserves the Gaussian reference measure. Implementation uses the
        model's preconditioner square root action to draw correlated noise.

        Args:
            state (np.ndarray): Current state.
            rng (np.random.Generator): Random number generator.

        Returns:
            np.ndarray: Proposed state with same shape as ``state``.
        """
        random_increment = rng.normal(size=state.shape)
        random_increment = self._model.compute_preconditioner_sqrt_action(random_increment)
        proposal = self._model.reference_point + (
            (2 - self._step_width) / (2 + self._step_width) * (state - self._model.reference_point)
            + np.sqrt(8 * self._step_width) / (2 + self._step_width) * random_increment
        )
        return proposal

    def _evaluate_acceptance_probability(
        self,
        current_state: np.ndarray[tuple[int], np.dtype[np.floating]],
        proposal: np.ndarray[tuple[int], np.dtype[np.floating]],
    ) -> float:
        """Compute acceptance probability for pCN.

        For pCN the proposal ratio cancels, yielding ``min(1, exp(-Phi(v) + Phi(u)))``.

        Args:
            current_state (np.ndarray): Current state ``u``.
            proposal (np.ndarray): Proposed state ``v``.

        Returns:
            float: Acceptance probability in ``[0, 1]``.
        """
        assert current_state.shape == proposal.shape, (
            f"Current state and proposal must have the same shape, but they have shapes"
            f"{current_state.shape} and {proposal.shape}, respectively."
        )
        if self._cached_args_current.potential is None:
            potential_current = self._model.evaluate_potential(current_state)
        else:
            potential_current = self._cached_args_current.potential
        potential_proposal = self._model.evaluate_potential(proposal)
        acceptance_probability = np.min((1, np.exp(-potential_proposal + potential_current)))

        self._cache_args(potential_current=potential_current, potential_proposal=potential_proposal)
        return acceptance_probability


# ==================================================================================================
@dataclass
class _MALACachedArgs(_CachedArgs):
    r"""Cache container for MALA intermediate quantities.

    Attributes:
        potential (float | None): Potential value $\Phi(u)$.
        gradient (np.ndarray | None): Gradient $\nabla\Phi(u)$.
        sqrt_action_gradient (np.ndarray | None): Action of preconditioner square root on gradient.
        action_gradient (np.ndarray | None): Action of full preconditioner on gradient.
    """

    potential: float | None = None
    gradient: np.ndarray[tuple[int], np.dtype[np.floating]] | None = None
    sqrt_action_gradient: np.ndarray[tuple[int], np.dtype[np.floating]] | None = None
    action_gradient: np.ndarray[tuple[int], np.dtype[np.floating]] | None = None


class MALAAlgorithm(MCMCAlgorithm):
    r"""Preconditioned Crank-Nicolson Langevin (pCNL / MALA) sampler.

    Implements the function-space MALA variant (Cotter et al., 2013) using a
    Gaussian reference prior with covariance operator $C$.

    Proposal (given current state $u$):

    $$
    v = \frac{2-\delta}{2+\delta}\,u
        - \frac{2\delta}{2+\delta}\,C \nabla \Phi(u)
        + \frac{\sqrt{8\delta}}{2+\delta}\,w,\qquad
    w \sim \mathcal N(0, C).
    $$

    Methods:
        compute_step: Compute one step of MCMC

    Notes:
        The preconditioner implicitly encodes the covariance operator $C$ of
        the reference Gaussian prior.

    References:
        Cotter, Roberts, Stuart, White (2013). "MCMC Methods for Functions:
        Modifying Old Algorithms to Make Them Faster." Statistical Science 28(3).
    """

    def __init__(self, model: model.DifferentiableMCMCModel, step_width: float) -> None:
        r"""Initialize MALA Sampling.

        Args:
            model (DifferentiableMCMCModel): Provides potential $\Phi$, its gradient,
                and preconditioner actions (embodying $C$).
            step_width (float): Step size $\delta > 0$. Smaller values increase
                acceptance; larger values explore faster until stability or
                acceptance deteriorates.
        """
        super().__init__(model, step_width)
        self._model = model
        self._cached_args_current: _MALACachedArgs = _MALACachedArgs()
        self._cached_args_proposal: _MALACachedArgs = _MALACachedArgs()

    def _populate_cache(
        self, cache: _MALACachedArgs, state: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> None:
        """Populates cache with necessary arguments for the proposal and accept-reject step.

        Args:
            cache (MALACachedArgs): _description_
            state (np.ndarray[tuple[int], np.dtype[np.floating]]): _description_
        """
        if cache.potential is None:
            cache.potential = self._model.evaluate_potential(state)
        if cache.gradient is None:
            cache.gradient = self._model.evaluate_gradient_of_potential(state)
        if cache.sqrt_action_gradient is None:
            cache.sqrt_action_gradient = self._model.compute_preconditioner_sqrt_action(
                cache.gradient
            )

    def _create_proposal(
        self, state: np.ndarray[tuple[int], np.dtype[np.floating]], rng: np.random.Generator
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        r"""Creates new proposal based on the current state.

        Args:
            state (np.ndarray[tuple[int], np.dtype[np.floating]]): current state
            rng (np.random.Generator): random number generator used for random increment

        Returns:
            np.ndarray: proposal
        """
        self._populate_cache(self._cached_args_current, state)
        if self._cached_args_current.action_gradient is None:
            self._cached_args_current.action_gradient = self._model.compute_preconditioner_action(
                self._cached_args_current.gradient
            )
        action_gradient = self._cached_args_current.action_gradient
        random_increment = rng.normal(size=state.shape)
        random_increment = self._model.compute_preconditioner_sqrt_action(random_increment)
        proposal = (
            (2 - self._step_width) * state
            - 2 * self._step_width * action_gradient
            + np.sqrt(8 * self._step_width) * random_increment
        ) / (2 + self._step_width)
        return proposal

    def _evaluate_acceptance_probability(
        self,
        current_state: np.ndarray[tuple[int], np.dtype[np.floating]],
        proposal: np.ndarray[tuple[int], np.dtype[np.floating]],
    ) -> float:
        """Calculates the acceptance probability for the proposal based on the current state.

        Args:
            current_state (np.ndarray[tuple[int], np.dtype[np.floating]]): current state
            proposal (np.ndarray[tuple[int], np.dtype[np.floating]]): proposed state

        Returns:
            float: acceptance probability
        """
        assert current_state.shape == proposal.shape, (
            f"Current state and proposal must have the same shape, but they have shapes"
            f"{current_state.shape} and {proposal.shape}, respectively."
        )
        self._populate_cache(self._cached_args_current, current_state)
        self._populate_cache(self._cached_args_proposal, proposal)
        p_u_v = (
            self._cached_args_current.potential
            + 1 / 2 * np.dot(proposal - current_state, self._cached_args_current.gradient)
            + self._step_width
            / 4
            * np.dot(current_state + proposal, self._cached_args_current.gradient)
            + self._step_width
            / 4
            * np.dot(
                self._cached_args_current.sqrt_action_gradient,
                self._cached_args_current.sqrt_action_gradient,
            )
        )

        p_v_u = (
            self._cached_args_proposal.potential
            + 1 / 2 * np.dot(current_state - proposal, self._cached_args_proposal.gradient)
            + self._step_width
            / 4
            * np.dot(proposal + current_state, self._cached_args_proposal.gradient)
            + self._step_width
            / 4
            * np.dot(
                self._cached_args_proposal.sqrt_action_gradient,
                self._cached_args_proposal.sqrt_action_gradient,
            )
        )

        acceptance_probability = np.min((1, np.exp(p_u_v - p_v_u)))

        return acceptance_probability
