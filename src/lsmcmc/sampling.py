import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from . import algorithms, logging, output, storage


# ==================================================================================================
@dataclass
class SamplerRunSettings:
    num_samples: int
    initial_state: np.ndarray
    print_interval: int
    store_interval: int


# ==================================================================================================
class Sampler:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        algorithm: algorithms.MCMCAlgorithm,
        sample_storage: storage.MCMCStorage | None = None,
        outputs: Iterable[output.MCMCOutput] | None = None,
        logger: logging.MCMCLogger | None = None,
    ) -> None:
        self._algorithm = algorithm
        self._samples = sample_storage
        self._outputs = outputs if outputs is not None else []
        self._logger = logger
        self._print_interval = None
        self._store_interval = None
        self._start_time = None

    # ----------------------------------------------------------------------------------------------
    def run(
        self, run_settings: SamplerRunSettings
    ) -> tuple[storage.MCMCStorage, Iterable[output.MCMCOutput]]:
        if run_settings.num_samples <= 0:
            raise ValueError("Number of samples must be greater than zero.")
        if run_settings.print_interval <= 0:
            raise ValueError("Print interval must be greater than zero.")
        if run_settings.store_interval <= 0:
            raise ValueError("Store interval must be greater than zero.")
        if run_settings.print_interval > run_settings.num_samples:
            raise ValueError("Print interval must be less than the number of samples.")
        if run_settings.store_interval > run_settings.num_samples:
            raise ValueError("Store interval must be less than the number of samples.")

        current_state = run_settings.initial_state
        self._num_samples = run_settings.num_samples
        self._print_interval = run_settings.print_interval
        self._store_interval = run_settings.store_interval
        self._start_time = time.time()
        self._run_utilities(0, current_state, accepted=True)

        try:
            for i in range(1, self._num_samples):
                new_state, accepted = self._algorithm.compute_step(current_state)
                self._run_utilities(i, new_state, accepted=accepted)
                current_state = new_state
        except BaseException as exc:
            self._logger.exception(exc)
        finally:
            return self._samples, self._outputs

    # ----------------------------------------------------------------------------------------------
    def _run_utilities(self, it: int, state: np.ndarray, accepted: bool) -> None:
        assert it >= 0, f"Iteration number must be non-negative, but has value{it}"
        store_values = (it % self._store_interval == 0) or (it == self._num_samples + 1)
        log_values = (it % self._print_interval == 0) or (it == self._num_samples + 1)

        for out in self._outputs:
            out.update(state, accepted)
        if self._samples and store_values:
            self._samples.store(state)
        if self._logger and log_values:
            if it == 0:
                self._logger.log_header(self._outputs)
            runtime = time.time() - self._start_time
            self._logger.log_outputs(self._outputs, it, runtime)
