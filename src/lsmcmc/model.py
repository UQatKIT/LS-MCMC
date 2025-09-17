from abc import ABC, abstractmethod

import numpy as np


# ==================================================================================================
class MCMCModel(ABC):
    @abstractmethod
    def evaluate_potential(self, state: np.ndarray[tuple[int], np.dtype[np.floating]]) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_preconditioner_sqrt_action(
        self, state: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def reference_point(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        raise NotImplementedError


class DifferentiableMCMCModel(MCMCModel):
    @abstractmethod
    def evaluate_gradient_of_potential(
        self, state: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        raise NotImplementedError
    
    def compute_preconditioner_action(
        self, state: np.ndarray[tuple[int], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        sqrt_action =  self.compute_preconditioner_sqrt_action(state)
        return self.compute_preconditioner_sqrt_action(sqrt_action)
