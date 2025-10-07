"""Main Sampler that handles the algorithm, output and storage of samples.

Classes:
    SamplerRunSettings: Settings for a sampler run
    Sampler: Main sampler that runs the MCMC loop
"""

import pickle
import signal
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

import numpy as np

from . import algorithms, logging, output, storage

CHECKPOINT_PATH = "./sampler_checkpoint.pickle"


# ==================================================================================================
@dataclass
class SamplerRunSettings:
    """Settings for running the MCMC sampler.

    Attributes:
        num_samples (int): Number of samples to generate.
        initial_state (np.ndarray): Initial state for the sampler.
        print_interval (int): Interval at which outputs should be printed or saved to a file.
        store_interval (int): Interval at which sample values should be stored.
    """

    num_samples: int
    initial_state: np.ndarray[tuple[int], np.dtype[np.floating]]
    print_interval: int = 1
    store_interval: int = 1
    checkpoint_path: Path = Path(CHECKPOINT_PATH)


@dataclass
class _SamplerCheckpoint:
    """Checkpoint data for resuming MCMC sampling.

    Attributes:
        iteration: Current iteration number.
        current_state: Current state of the Markov chain.
        rng_state: State of the random number generator.
        run_settings: Original run settings.
        outputs_state: Serialized state of output objects.
    """

    iteration: int
    current_state: np.ndarray[tuple[int], np.dtype[np.floating]]
    rng_state: dict
    run_settings: SamplerRunSettings
    outputs_state: Iterable[output.MCMCOutput]


# ==================================================================================================
class Sampler:
    """MCMC sampler that runs a given algorithm and manages outputs, logging, and storage.

    Methods:
        resume_from_checkpoint: Resume sampling from a saved checkpoint
        run: Run the main MCMC loop
    """

    def __init__(
        self,
        algorithm: algorithms.MCMCAlgorithm,
        sample_storage: storage.MCMCStorage = None,
        outputs: Iterable[output.MCMCOutput] | None = None,
        logger: logging.MCMCLogger | None = None,
        seed: int = 0,
    ) -> None:
        """Initializes the Sampler.

        Args:
            algorithm (algorithms.MCMCAlgorithm): The MCMC algorithm to use.
            sample_storage (storage.MCMCStorage, optional): Storage for samples (e.g. disk).
            outputs (Iterable[output.MCMCOutput], optional): Outputs to compute during sampling
                for logging.
            logger (logging.MCMCLogger, optional): Logger for progress and diagnostics.
            seed (int, optional): Random seed for reproducibility.
        """
        self._algorithm = algorithm
        self._samples = sample_storage
        self._outputs = outputs if outputs is not None else []
        self._logger = logger
        self._print_interval = None
        self._store_interval = None
        self._start_time = None
        self._terminate = False
        self._rng = np.random.default_rng(seed=seed)

    @classmethod
    def resume_from_checkpoint(
        cls,
        algorithm: algorithms.MCMCAlgorithm,
        sample_storage: storage.MCMCStorage,
        logger: logging.MCMCLogger,
        checkpoint_path: Path | None = None,
    ) -> "Sampler":
        """Resume sampling from a saved checkpoint."""
        if not checkpoint_path:
            checkpoint_path = Path(CHECKPOINT_PATH)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Could not find checkpoint: {checkpoint_path}")
        with Path.open(checkpoint_path, "rb") as f:
            checkpoint: _SamplerCheckpoint = pickle.load(f)

        sampler = cls(
            algorithm=algorithm,
            sample_storage=sample_storage,
            outputs=checkpoint.outputs_state,
            logger=logger,
        )

        sampler._rng.bit_generator.state = checkpoint.rng_state
        checkpoint.run_settings.initial_state = checkpoint.current_state
        return sampler.run(run_settings=checkpoint.run_settings, iteration=checkpoint.iteration)

    def _verify_run_settings(self, run_settings: SamplerRunSettings) -> None:
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

    def run(
        self, run_settings: SamplerRunSettings, iteration: int = 0
    ) -> tuple[storage.MCMCStorage, Iterable[output.MCMCOutput]]:
        """Run the MCMC sampler for the specified settings.

        Args:
            run_settings (SamplerRunSettings): Settings for the sampler run.
            iteration (int): The iteration number at which the sampler starts.
                Only relevant for restarting a chain.

        Returns:
            tuple[storage.MCMCStorage, Iterable[output.MCMCOutput]]:
                The sample storage and the outputs after sampling.
        """
        self._verify_run_settings(run_settings)

        self._setup_handlers()
        self._current_state = run_settings.initial_state
        self._num_samples = run_settings.num_samples
        self._print_interval = run_settings.print_interval
        self._store_interval = run_settings.store_interval
        self._start_time = time.time()
        if iteration == 0:
            self._run_utilities(iteration, self._current_state, accepted=True)

        # save sample variables as class members for correct termination handling
        self._iteration = 1 + iteration
        self._run_settings = run_settings

        try:
            for i in range(1 + iteration, self._num_samples):
                self._iteration = i
                new_state, accepted = self._algorithm.compute_step(self._current_state, self._rng)
                self._run_utilities(i, new_state, accepted=accepted)
                self._current_state = new_state
                if self._terminate:
                    break
        except BaseException:
            self._logger.exception("An Error occured during sampling.")
        else:
            self._iteration = None
            self._current_state = None
            self._run_settings = None
            return self._samples, self._outputs

    def _setup_handlers(self) -> None:
        def _handler(_signum: int, _frame: FrameType | None) -> None:
            self._terminate = True
            self._handle_termination()

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)

    def _handle_termination(
        self,
    ) -> None:
        if self._logger:
            self._logger.info("Received stop signal, shutting down gracefully.")

        # save samples
        if self._samples:
            try:
                self._samples.flush()
                if self._logger:
                    self._logger.info("Storage flushing complete.")
            except BaseException:
                if self._logger:
                    self._logger.exception("Failed to flush samples.")

        # save chain sate (i.e. rnga and some metadata)
        try:
            checkpoint = _SamplerCheckpoint(
                iteration=self._iteration,
                current_state=self._current_state.copy(),
                rng_state=self._rng.bit_generator.state,
                run_settings=self._run_settings,
                outputs_state=self._outputs,
            )
            with Path.open(self._run_settings.checkpoint_path, "wb") as f:
                pickle.dump(checkpoint, f)
            if self._logger:
                self._logger.info(f"Checkpoint saved to {self._run_settings.checkpoint_path}")
        except BaseException:
            if self._logger:
                self._logger.exception("Failed to save checkpoint.")

    def _run_utilities(
        self, it: int, state: np.ndarray[tuple[int], np.dtype[np.floating]], accepted: bool
    ) -> None:
        """Update outputs, store samples, and log progress for the current iteration.

        Args:
            it (int): Current iteration number.
            state (np.ndarray): Current state of the chain.
            accepted (bool): Whether the proposed state was accepted.
        """
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
