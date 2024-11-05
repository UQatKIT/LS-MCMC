from abc import ABC, abstractmethod
from numbers import Number

import numpy as np


# ==================================================================================================
class MCMCQoI(ABC):
    @abstractmethod
    def evaluate(self, state: np.ndarray, accepted: bool) -> Number:
        """_summary_."""
        raise NotImplementedError


# --------------------------------------------------------------------------------------------------
class ComponentQoI(MCMCQoI):
    def __init__(self, component: int) -> None:
        self._component = component

    def evaluate(self, state: np.ndarray, _accepted: bool) -> float:
        return state[self._component]


# --------------------------------------------------------------------------------------------------
class MeanQoI(MCMCQoI):
    @staticmethod
    def evaluate(state: np.ndarray, _: bool) -> float:
        return np.mean(state)


# --------------------------------------------------------------------------------------------------
class AcceptanceQoI(MCMCQoI):
    @staticmethod
    def evaluate(_: np.ndarray, accepted: bool) -> float:
        return float(accepted)


# ==================================================================================================
class MCMCStatistic(ABC):
    @abstractmethod
    def evaluate(self, qoi_value: Number) -> Number:
        """_summary_."""
        raise NotImplementedError


# --------------------------------------------------------------------------------------------------
class IdentityStatistic(MCMCStatistic):
    @staticmethod
    def evaluate(qoi_value: Number) -> Number:
        return qoi_value


# --------------------------------------------------------------------------------------------------
class RunningMeanStatistic(MCMCStatistic):
    def __init__(self) -> None:
        self._running_value = 0
        self._num_samples = 0

    def evaluate(self, qoi_value: Number) -> float:
        new_value = self._num_samples / (
            self._num_samples + 1
        ) * self._running_value + qoi_value / (self._num_samples + 1)
        self._num_samples += 1
        self._running_value = new_value
        return new_value


# --------------------------------------------------------------------------------------------------
class BatchMeanStatistic(MCMCStatistic):
    def __init__(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")
        self._running_value = 0
        self._num_samples = 0
        self._batch_size = batch_size
        self._values = []
        self._batch_mean = 0

    def evaluate(self, qoi_value: Number) -> float:
        self._values.append(qoi_value)
        if len(self._values) == self._batch_size:
            self._batch_mean = np.mean(self._values)
            self._values.clear()
        return self._batch_mean


# ==================================================================================================
class MCMCOutput:
    def __init__(
        self,
        qoi: MCMCQoI,
        statistic: MCMCStatistic,
        str_id: str | None = None,
        str_format: str | None = None,
        log: bool = False,
    ) -> None:
        if log and str_id is None:
            raise ValueError("String ID must be provided if output is to be logged.")
        if log and str_format is None:
            raise ValueError("String format must be provided if output is to be logged.")
        self.str_id = str_id
        self.str_format = str_format
        self._qoi = qoi
        self._statistic = statistic
        self.log = log
        self._values = []

    def update(self, state: np.ndarray, accepted: bool) -> None:
        scalar_output = self._qoi.evaluate(state, accepted)
        scalar_output = self._statistic.evaluate(scalar_output)
        self._values.append(scalar_output)

    @property
    def value(self) -> Number:
        return self._values[-1]

    @property
    def all_values(self) -> np.ndarray:
        return np.array(self._values)
