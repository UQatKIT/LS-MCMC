"""MCMC Output Tracking.

Outputs generally consist of two components: a Quantity of Interest (QoI), which is computed solely
from the samples, and a statistic, which is derived from the QoI.

For example, to track the acceptance rate, the QoI could be "Was the sample accepted?".
A corresponding statistic might be the average acceptance over the last 100 samples,
or the overall average across all samples.

This module implements some basic QoIs and statistics.

Classes:
    MCMCQoI: Abstract base class of Quantity of Interest
    MCMCStatistic: Abstract base class for statistics
    MCMCOutput: Output object that combines a QoI and statistic
    SimplifiedMCMCOutput: MCMCOutput that autoconfigures as much as possible
    ComponentQoi: A specific component of the state as a QoI
    MeanQoI: The mean as a QoI
    AcceptanceQoI: The acceptance as a QoI
    IdentityStatistic: Identity as a statistic
    RunningMeanStatistic: A running mean as a statistic
    BatchMeanStatistic: A batch mean as a statistic
    Acceptance: Preconfigured output that tracks the running mean of the acceptance
"""

from abc import ABC, abstractmethod
from numbers import Number

import numpy as np


# ==================================================================================================
class MCMCQoI(ABC):
    """Abstract base class for Quantities of Interest (QoI) in MCMC sampling.

    Methods:
        evaluate: Calculate QoI from state vector
        name: Name of the QoI
    """

    @abstractmethod
    def evaluate(
        self, state: np.ndarray[tuple[int], np.dtype[np.floating]], accepted: bool
    ) -> Number:
        """Evaluates QoI from state vector.

        Args:
            state: Current state vector from MCMC chain.
            accepted: Whether the current proposal was accepted.

        Returns:
            Evaluated quantity of interest value
        """
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the quantity of interest."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Name of the QoI for logging the output."""
        return self.name()


# --------------------------------------------------------------------------------------------------
class ComponentQoI(MCMCQoI):
    """QoI that extracts a specific component from the state vector.

    Methods:
        evaluate: Calculate QoI from state vector
        name: Name of the QoI
    """

    def __init__(self, component: int) -> None:
        """Initialize ComponentQoI.

        Args:
            component: Index of the component to extract from state vector.
        """
        self._component = component

    def evaluate(
        self, state: np.ndarray[tuple[int], np.dtype[np.floating]], _accepted: bool
    ) -> float:
        """Extracts specified component from state."""
        return state[self._component]

    def name(self) -> str:
        """String representation of the form "component <component number>" used for logging."""
        return f"component {self._component}"


# --------------------------------------------------------------------------------------------------
class MeanQoI(MCMCQoI):
    """QoI that computes the mean of all components in the state vector.

    Methods:
        evaluate: Calculate QoI from state vector
        name: Name of the QoI
    """

    @staticmethod
    def evaluate(state: np.ndarray[tuple[int], np.dtype[np.floating]], _: bool) -> float:
        """Extracts mean of the state."""
        return np.mean(state)

    @staticmethod
    def name() -> str:
        """String "mean" used for logging the output."""
        return "mean"


# --------------------------------------------------------------------------------------------------
class AcceptanceQoI(MCMCQoI):
    """QoI that tracks proposal acceptance status.

    Methods:
        evaluate: Calculate QoI from state vector
        name: Name of the QoI
    """

    @staticmethod
    def evaluate(_: np.ndarray[tuple[int], np.dtype[np.floating]], accepted: bool) -> float:
        """Extracts whether a state was accepted or not and converts to float for calculations."""
        return float(accepted)

    @staticmethod
    def name() -> str:
        """String "acceptance" used for logging the output."""
        return "acceptance"


# ==================================================================================================
class MCMCStatistic(ABC):
    """Abstract base class for statistics computed from QoI values.

    Methods:
        evaluate: Calculate statistic from a Quantity of Interest
        name: Name of the Statistic
    """

    @abstractmethod
    def evaluate(self, qoi_value: Number) -> Number:
        """Evaluates a certain statistic from a Quantity of Interest.

        Args:
            qoi_value: Value from a quantity of interest evaluation

        Returns:
            Computed statistic value.
        """
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the statistic."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Name of the Statistic used for logging."""
        return self.name()


# --------------------------------------------------------------------------------------------------
class IdentityStatistic(MCMCStatistic):
    """Statistic that returns the input value unchanged.

    Methods:
        evaluate: Calculate statistic from a Quantity of Interest
        name: Name of the Statistic
    """

    @staticmethod
    def evaluate(qoi_value: Number) -> Number:
        """Returns the QoI unchanged."""
        return qoi_value

    @staticmethod
    def name() -> str:
        """String "identity" for logging the output."""
        return "identity"


# --------------------------------------------------------------------------------------------------
class RunningMeanStatistic(MCMCStatistic):
    """Statistic that computes a running mean of QoI values.

    Methods:
        evaluate: Calculate statistic from a Quantity of Interest
        name: Name of the Statistic
    """

    def __init__(self) -> None:
        """Initialize running mean."""
        self._running_value = 0
        self._num_samples = 0

    def evaluate(self, qoi_value: Number) -> float:
        """Updates the running mean based on a new sample QoI."""
        new_value = self._num_samples / (
            self._num_samples + 1
        ) * self._running_value + qoi_value / (self._num_samples + 1)
        self._num_samples += 1
        self._running_value = new_value
        return new_value

    def name(self) -> str:
        """String "mean" for logging the output."""
        return "mean"


# --------------------------------------------------------------------------------------------------
class BatchMeanStatistic(MCMCStatistic):
    """Statistic that computes the mean of QoI values in batches.

    Methods:
        evaluate: Calculate statistic from a Quantity of Interest
        name: Name of the Statistic
    """

    def __init__(self, batch_size: int) -> None:
        """Initialize BatchMeanStatistic.

        Args:
            batch_size: Number of samples per batch

        Raises:
            ValueError: If batch_size is less than or equal to zero.
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")
        self._running_value = 0
        self._num_samples = 0
        self._batch_size = batch_size
        self._values = []
        self._batch_mean = 0

    def evaluate(self, qoi_value: Number) -> float:
        """Updates the batch mean based on a new sample QoI."""
        self._values.append(qoi_value)
        if len(self._values) >= self._batch_size:
            self._batch_mean = np.mean(self._values)
            self._values.clear()
        return self._batch_mean

    def name(self) -> str:
        """String "BM<batch size>" for logging the output."""
        return f"BM{self._batch_size}"


# ==================================================================================================
class MCMCOutput:
    """Combines a QoI and statistic for MCMC output tracking and logging.

    Methods:
        update: update output based on a new state
        value: return most recent output value
        values: return all output values
    """

    def __init__(
        self,
        qoi: MCMCQoI,
        statistic: MCMCStatistic,
        str_id: str | None = None,
        str_format: str | None = None,
        log: bool = False,
    ) -> None:
        """Initialize MCMCOutput.

        Args:
            qoi: Quantity of interest to evaluate
            statistic: Statistic to compute from QoI values
            str_id: String identifier for logging. Required if log=True
            str_format: Format string for logging. Required if log=True
            log: Whether to include this output in logging

        Raises:
            ValueError: If log=True but str_id or str_format is None.
        """
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

    def update(self, state: np.ndarray[tuple[int], np.dtype[np.floating]], accepted: bool) -> None:
        """Update output with new MCMC state.

        Args:
            state: Current state vector from MCMC chain
            accepted: Whether the current proposal was accepted
        """
        scalar_output = self._qoi.evaluate(state, accepted)
        scalar_output = self._statistic.evaluate(scalar_output)
        self._values.append(scalar_output)

    @property
    def value(self) -> Number:
        """Returns the most recent output value."""
        return self._values[-1]

    @property
    def all_values(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """Returns all computed output values as numpy array."""
        return np.array(self._values)


class Acceptance(MCMCOutput):
    """Pre-configured output for tracking average acceptance rate.

    Methods:
        update: update output based on a new state
        value: return most recent output value
        values: return all output values
    """

    def __init__(self) -> None:
        """Initialize acceptance output."""
        super().__init__(
            qoi=AcceptanceQoI(),
            statistic=RunningMeanStatistic(),
            str_id=f"{'Acceptance':<12}",
            str_format="<12.4",
            log=True,
        )


class SimplifiedOutput(MCMCOutput):
    """Simplified output constructor with automatic string formatting.

    Methods:
        update: update output based on a new state
        value: return most recent output value
        values: return all output values
    """

    def __init__(
        self,
        qoi: MCMCQoI,
        statistic: MCMCStatistic,
    ) -> None:
        """Initialize SimplifiedOutput with automatic formatting.

        Args:
            qoi: Quantity of interest to evaluate.
            statistic: Statistic to compute from QoI values.
        """
        str_id = f"{statistic} of {qoi}"
        if isinstance(statistic, IdentityStatistic):
            str_id = f"{qoi}"
        str_id = f"{str_id:<12}"
        super().__init__(
            qoi=qoi,
            statistic=statistic,
            str_id=str_id,
            str_format=f"<+{max(12, len(str_id))}.3e",
            log=True,
        )
