from dataclasses import dataclass

import numpy as np

from . import algorithms, logging, statistics, storage


# ==================================================================================================
@dataclass
class SamplerRunSettings:
    num_samples: int
    initial_state: np.ndarray
    print_interval: int


# ==================================================================================================
class Sampler:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        algorithm: algorithms.MCMCAlgorithm,
        sample_storage: storage.MCMCStorage,
        stats: statistics.MCMCStatistics,
        logger: logging.MCMCLogger,
    ):
        self._algorithm = algorithm
        self._samples = sample_storage
        self._statistics = stats
        self._logger = logger
        self._print_interval = None

    # ----------------------------------------------------------------------------------------------
    def run(self, run_settings: SamplerRunSettings):
        current_state = run_settings.initial_state
        self._print_interval = run_settings.print_interval
        self._samples.store(current_state)
        self._update_statistics(current_state)
        self._log_statistics(iteration=0)

        try:
            for i in range(run_settings.num_samples):
                new_state, accepted = self._algorithm.compute_step(current_state)
                self._samples.store(new_state)
                self._update_statistics(new_state, accepted)
                self._log_statistics(iteration=i + 1)
        except BaseException as exc:
            self._logger.exception(exc)
        finally:
            return self._samples, self._statistics

    # ----------------------------------------------------------------------------------------------
    def _update_statistics(self, state: np.ndarray, accepted: bool):
        pass

    # ----------------------------------------------------------------------------------------------
    def _log_statistics(self, iteration: int):
        pass
