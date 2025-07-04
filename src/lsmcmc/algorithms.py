from abc import ABC, abstractmethod

import numpy as np

from . import model


# ==================================================================================================
class MCMCAlgorithm(ABC):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, model: model.MCMCModel, step_width: float) -> None:
        if step_width <= 0:
            raise ValueError("Step width must be greater than zero.")
        self._step_width = step_width
        self._model = model
        self._cached_args = {}

    # ----------------------------------------------------------------------------------------------
    def compute_step(
        self, current_state: np.ndarray[tuple[int], np.dtype[np.floating]], rng: np.random.Generator
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], bool]:
        proposal, computed_args = self._create_proposal(current_state, rng)
        self._cache_args(computed_args)
        new_state, computed_args, accepted = self._perform_accept_reject(
            current_state, proposal, rng
        )
        computed_args = self._choose_args_to_cache(computed_args, accepted)
        self._cache_args(computed_args)
        return new_state, accepted

    # ----------------------------------------------------------------------------------------------
    def _perform_accept_reject(
        self,
        current_state: np.ndarray[tuple[int], np.dtype[np.floating]],
        proposal: np.ndarray[tuple[int], np.dtype[np.floating]],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], dict, bool]:
        assert current_state.shape == proposal.shape, (
            f"Current state and proposal must have the same shape, but they have shapes"
            f"{current_state.shape} and {proposal.shape}, respectively."
        )
        acceptance_probability, computed_args = self._evaluate_acceptance_probability(
            current_state, proposal
        )
        random_draw = rng.uniform()
        if random_draw < acceptance_probability:
            new_state = proposal
            accepted = True
        else:
            new_state = current_state
            accepted = False
        return new_state, computed_args, accepted

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _choose_args_to_cache(computed_args: dict[str, tuple], accepted: bool) -> dict[str, float]:
        valid_arg_ind = 0 if accepted else 1
        computed_args_to_cache = {arg: computed_args[arg][valid_arg_ind] for arg in computed_args}
        return computed_args_to_cache

    # ----------------------------------------------------------------------------------------------
    def _cache_args(self, computed_args: dict[str, tuple]) -> None:
        for arg in computed_args:
            self._cached_args[arg] = computed_args[arg]

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def _create_proposal(
        self, state: np.ndarray[tuple[int], np.dtype[np.floating]], rng: np.random.Generator
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def _evaluate_acceptance_probability(
        self,
        current_state: np.ndarray[tuple[int], np.dtype[np.floating]],
        proposal: np.ndarray[tuple[int], np.dtype[np.floating]],
    ) -> float:
        raise NotImplementedError


# ==================================================================================================
class pCNAlgorithm(MCMCAlgorithm):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, model: model.MCMCModel, step_width: float) -> None:
        super().__init__(model, step_width)
        self._cached_args = {"potential": None}

    # ----------------------------------------------------------------------------------------------
    def _create_proposal(
        self, state: np.ndarray[tuple[int], np.dtype[np.floating]], rng: np.random.Generator
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.floating]], dict]:
        random_increment = rng.normal(size=state.shape)
        random_increment = self._model.compute_preconditioner_sqrt_action(random_increment)
        proposal = (
            self._model.reference_point
            + np.sqrt(1 - self._step_width**2) * (state - self._model.reference_point)
            + self._step_width * random_increment
        )
        computed_args = {}
        return proposal, computed_args

    # ----------------------------------------------------------------------------------------------
    def _evaluate_acceptance_probability(
        self,
        current_state: np.ndarray[tuple[int], np.dtype[np.floating]],
        proposal: np.ndarray[tuple[int], np.dtype[np.floating]],
    ) -> tuple[float, dict]:
        assert current_state.shape == proposal.shape, (
            f"Current state and proposal must have the same shape, but they have shapes"
            f"{current_state.shape} and {proposal.shape}, respectively."
        )
        if self._cached_args["potential"] is None:
            potential_current = self._model.evaluate_potential(current_state)
        else:
            potential_current = self._cached_args["potential"]
        potential_proposal = self._model.evaluate_potential(proposal)
        acceptance_probability = np.min((1, np.exp(-potential_proposal + potential_current)))

        computed_args = {"potential": (potential_proposal, potential_current)}
        computed_args = {}
        return acceptance_probability, computed_args
