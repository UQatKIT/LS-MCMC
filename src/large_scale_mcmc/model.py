from abc import ABC, abstractmethod

import numpy as np


# ==================================================================================================
class MCMCModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate_potential(self):
        pass

    @abstractmethod
    def evaluate_potential_grad(self):
        pass

    @abstractmethod
    def compute_prior_precision_action(self, state: np.ndarray):
        pass

    @abstractmethod
    def compute_preconditioner_action(self, state: np.ndarray):
        pass

    @abstractmethod
    def compute_preconditioner_sqrt_action(self, state: np.ndarray):
        pass