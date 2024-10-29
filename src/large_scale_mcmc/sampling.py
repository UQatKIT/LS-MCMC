import time
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
        sample_storage: storage.MCMCStorage,
        outputs: output.MCMCOutput,
        logger: logging.MCMCLogger,
    ):
        self._algorithm = algorithm
        self._samples = sample_storage
        self._outputs = outputs
        self._logger = logger
        self._print_interval = None
        self._store_interval = None
        self._start_time = None

    # ----------------------------------------------------------------------------------------------
    def run(self, run_settings: SamplerRunSettings):
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
    def _run_utilities(self, iteration: int, state: np.ndarray, accepted: bool):
        for output in self._outputs:
            output.update(state, accepted)

        store_values = (iteration % self._store_interval == 0) or (
            iteration == self._num_samples + 1
        )
        log_values = (iteration % self._print_interval == 0) or (iteration == self._num_samples + 1)

        if self._samples and store_values:
            self._samples.store(state)
        if self._logger and log_values:
            if iteration == 0:
                self._logger.log_header(self._outputs)
            current_time = time.time() - self._start_time
            self._logger.log_outputs(self._outputs, iteration, current_time)
