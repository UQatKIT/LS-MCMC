from abc import ABC, abstractmethod

import numpy as np


# ==================================================================================================
class MCMCQoI(ABC):
    @abstractmethod
    def evaluate(self, state: np.ndarray, accepted: bool):
        pass


# --------------------------------------------------------------------------------------------------
class MeanQoI(MCMCQoI):
    @staticmethod
    def evaluate(state: np.ndarray, _: bool):
        return np.mean(state)


# ==================================================================================================
class MCMCStatistic(ABC):
    @abstractmethod
    def evaluate(self, qoi_value: np.ndarray):
        pass


# --------------------------------------------------------------------------------------------------
class IdentityStatistic(MCMCStatistic):
    @staticmethod
    def evaluate(qoi_value: np.ndarray):
        return qoi_value


# --------------------------------------------------------------------------------------------------
class RunningMeanStatistic(MCMCStatistic):
    def __init__(self):
        self._running_value = 0
        self._num_samples = 0

    def evaluate(self, qoi_value: np.ndarray):
        new_value = self._num_samples / (
            self._num_samples + 1
        ) * self._running_value + qoi_value / (self._num_samples + 1)
        self._num_samples += 1
        self._running_value = new_value
        return new_value


# --------------------------------------------------------------------------------------------------
class BatchMeanStatistic(MCMCStatistic):
    def __init__(self, batch_size: int):
        self._running_value = 0
        self._num_samples = 0
        self._batch_size = batch_size
        self._values = []

    def evaluate(self, qoi_value: np.ndarray):
        self._values.append(qoi_value)
        if len(self._values) == self._batch_size:
            new_value = np.mean(self._values)
            self._values.clear()
        return new_value


# ==================================================================================================
class MCMCOutput:
    def __init__(self, str_id: str, str_format: str, qoi: MCMCQoI, statistic: MCMCStatistic):
        self.str_id = str_id
        self.str_format = str_format
        self._qoi = qoi
        self._statistic = statistic
        self._values = []

    def update(self, state: np.ndarray, accepted: bool):
        scalar_output = self._qoi.evaluate(state, accepted)
        scalar_output = self._statistic.evaluate(scalar_output)
        self._values.append(scalar_output)

    @property
    def value(self):
        return self._values[-1]

    @property
    def all_values(self):
        return np.array(self._values)
