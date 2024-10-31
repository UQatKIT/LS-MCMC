from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from . import model


# ==================================================================================================
@dataclass
class AlgorithmSettings:
    step_width: float
    proposal_rng: np.random.Generator
    accept_reject_rng: np.random.Generator


# ==================================================================================================
class MCMCAlgorithm(ABC):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: AlgorithmSettings, model: model.MCMCModel) -> None:
        self._proposal_rng = settings.proposal_rng
        self._accept_reject_rng = settings.accept_reject_rng
        self._step_width = settings.step_width
        self._model = model
        self._cached_args = {}

    # ----------------------------------------------------------------------------------------------
    def compute_step(self, current_state: np.ndarray) -> tuple[np.ndarray, bool]:
        proposal, computed_args = self._create_proposal(current_state)
        self._cache_args(computed_args)
        new_state, computed_args, accepted = self._perform_accept_reject(current_state, proposal)
        computed_args = self._choose_args_to_cache(computed_args, accepted)
        self._cache_args(computed_args)
        return new_state, accepted

    # ----------------------------------------------------------------------------------------------
    def _perform_accept_reject(
        self, current_state: np.ndarray, proposal: np.ndarray
    ) -> tuple[np.ndarray, dict, bool]:
        acceptance_probability, computed_args = self._evaluate_acceptance_probability(
            current_state, proposal
        )
        random_draw = self._accept_reject_rng.uniform()
        if random_draw < acceptance_probability:
            new_state = proposal
            accepted = True
        else:
            new_state = current_state
            accepted = False
        return new_state, computed_args, accepted

    # ----------------------------------------------------------------------------------------------
    @property
    def rngs(self) -> tuple[np.random.Generator, np.random.Generator]:
        return self._proposal_rng, self._accept_reject_rng

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
    def _create_proposal(self, state: np.ndarray) -> np.ndarray:
        pass

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def _evaluate_acceptance_probability(
        self, current_state: np.ndarray, proposal: np.ndarray
    ) -> float:
        pass


# ==================================================================================================
class pCNAlgorithm(MCMCAlgorithm):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: AlgorithmSettings, model: model.MCMCModel) -> None:
        super().__init__(settings, model)
        self._cached_args = {"potential": None}

    # ----------------------------------------------------------------------------------------------
    def _create_proposal(self, state: np.ndarray) -> tuple[np.ndarray, dict]:
        random_increment = self._proposal_rng.normal(size=state.shape)
        random_increment = self._model.compute_preconditioner_sqrt_action(random_increment)
        ref_point = self._model.reference_point
        proposal = (
            ref_point
            + np.sqrt(1 - self._step_width**2) * (state - ref_point)
            + self._step_width * random_increment
        )
        computed_args = {}
        return proposal, computed_args

    # ----------------------------------------------------------------------------------------------
    def _evaluate_acceptance_probability(
        self, current_state: np.ndarray, proposal: np.ndarray
    ) -> tuple[float, dict]:
        if self._cached_args["potential"] is None:
            potential_current = self._model.evaluate_potential(current_state)
        else:
            potential_current = self._cached_args["potential"]
        potential_proposal = self._model.evaluate_potential(proposal)
        acceptance_probability = np.min((1, np.exp(-potential_proposal + potential_current)))

        computed_args = {"potential": (potential_proposal, potential_current)}
        computed_args = {}
        return acceptance_probability, computed_args
