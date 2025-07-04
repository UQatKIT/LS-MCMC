import numpy as np
from pathlib import Path

from lsmcmc import algorithms, logging, output, sampling, storage, model

class BananaModel(model.MCMCModel):
    def __init__(self) -> None:
        self._reference_point = np.array([0.0, 0.0])
        self._preconditioner_sqrt_matrix = np.identity(2)

    @staticmethod
    def evaluate_potential(state: np.ndarray) -> float:
        potential = (
            10 * np.square(np.square(state[0]) - state[1])
            + np.power(state[1], 4)
            - 0.5 * (np.square(state[0]) + np.square(state[1]))
        )
        return potential

    def compute_preconditioner_sqrt_action(self, state: np.ndarray) -> np.ndarray:
        action = self._preconditioner_sqrt_matrix @ state
        return action

    @property
    def reference_point(self) -> np.ndarray:
        return self._reference_point


def test_example_integration():
    # Set up outputs
    acceptance_rate_output = output.Acceptance()
    c0_output = output.SimplifiedOutput(output.ComponentQoI(0), output.IdentityStatistic())
    running_mean_c0_output = output.SimplifiedOutput(output.ComponentQoI(0), output.RunningMeanStatistic())
    batch_mean_c0_output = output.SimplifiedOutput(output.ComponentQoI(0), output.BatchMeanStatistic(1000))
    outputs = (acceptance_rate_output, c0_output, running_mean_c0_output, batch_mean_c0_output)

    # Logger and sampler settings
    logger_settings = logging.LoggerSettings(
        do_printing=False,  # Disable printing for test
        logfile_path=Path("logfile_test.log"),
        write_mode="w",
    )
    sampler_settings = sampling.SamplerRunSettings(
        num_samples=1000,
        initial_state=np.array([-0.5, 0.2]),
        print_interval=500,
        store_interval=1,
    )

    # Set up sampler
    sample_storage = storage.NumpyStorage()
    posterior_model = BananaModel()
    logger = logging.MCMCLogger(logger_settings)
    algorithm = algorithms.pCNAlgorithm(posterior_model, step_width=0.4)
    sampler = sampling.Sampler(algorithm, sample_storage, outputs, logger)

    samples, outputs_result = sampler.run(sampler_settings)

    # Basic checks
    assert hasattr(samples, "values")
    assert isinstance(samples.values, np.ndarray)
    assert samples.values.shape[1] == sampler_settings.num_samples, "Not enough samples were generated"
    assert samples.values.shape[0] == 2, "Samples have the wrong shape"

    # Check that outputs were computed
    assert len(outputs_result) == 4