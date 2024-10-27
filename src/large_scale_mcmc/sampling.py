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
        outputs: statistics.MCMCOutput,
        logger: logging.MCMCLogger,
    ):
        self._algorithm = algorithm
        self._samples = sample_storage
        self._outputs = outputs
        self._logger = logger
        self._print_interval = None

    # ----------------------------------------------------------------------------------------------
    def run(self, run_settings: SamplerRunSettings):
        current_state = run_settings.initial_state
        self._print_interval = run_settings.print_interval
        self._num_samples = run_settings.num_samples
        self._samples.store(current_state)
        [output.update(current_state, accepted=True) for output in self._outputs]
        self._logger.log_header(self._outputs)
        self._logger.log_outputs(self._outputs, iteration=0)

        try:
            for i in range(1, self._num_samples):
                new_state, accepted = self._algorithm.compute_step(current_state)
                self._samples.store(new_state)
                [output.update(current_state, accepted) for output in self._outputs]
                if (i % self._print_interval == 0) or (i == self._num_samples+1):
                    self._logger.log_outputs(self._outputs, iteration=i)
        except BaseException as exc:
            self._logger.exception(exc)
        finally:
            return self._samples, self._outputs
