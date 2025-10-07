from pathlib import Path

import numpy as np
import pytest

from ls_mcmc import algorithms, logging, model, output, sampling, storage


class BananaModel(model.DifferentiableMCMCModel):
    """Toy 'banana-shaped' 2D target with differentiable potential for MCMC algorithms."""

    def __init__(self) -> None:
        """Initialize reference point and identity preconditioner."""
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

    @staticmethod
    def evaluate_gradient_of_potential(state: np.ndarray) -> np.ndarray:
        x, y = state[0], state[1]
        dv_dx = 10 * 2 * (x**2 - y) * 2 * x - x
        dv_dy = 10 * 2 * (x**2 - y) * (-1) + 4 * y**3 - y
        return np.array([dv_dx, dv_dy])

    def compute_preconditioner_sqrt_action(self, state: np.ndarray) -> np.ndarray:
        action = self._preconditioner_sqrt_matrix @ state
        return action

    @property
    def reference_point(self) -> np.ndarray:
        return self._reference_point


class GaussModel(model.DifferentiableMCMCModel):
    r"""2D Gaussian with given mean and covariance, exposing $\Phi(x)$ and $\nabla\Phi(x)$."""

    def __init__(
        self,
        mean: np.ndarray[tuple[int], np.dtype[np.floating]],
        cov: np.ndarray[tuple[int, int], np.dtype[np.floating]],
    ) -> None:
        self._reference_point = np.array([0.0, 0.0])
        self._preconditioner_sqrt_matrix = np.identity(2)
        mean = np.array(mean)
        cov = np.array(cov)
        self._mean = mean
        self._cov = cov
        assert np.array_equal(cov, cov.T), "Covariance Matrix not symmetric or not square"
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as err:
            raise ValueError("Covariance Matrix is not SPD.") from err
        assert cov.shape[0] == mean.shape[0], "Covariance does not have the same shape as mean."
        self._prec = np.linalg.inv(cov)

    def evaluate_potential(self, state: np.ndarray) -> float:
        potential = (
            1 / 2 * np.dot(state - self._mean, self._prec @ (state - self._mean))
            - 1 / 2 * state @ state
        )
        return potential

    def evaluate_gradient_of_potential(self, state: np.ndarray) -> np.ndarray:
        return self._prec @ (state - self._mean) - state

    def compute_preconditioner_sqrt_action(self, state: np.ndarray) -> np.ndarray:
        action = self._preconditioner_sqrt_matrix @ state
        return action

    @property
    def reference_point(self) -> np.ndarray:
        return self._reference_point


@pytest.fixture
def sample_storage() -> storage.MCMCStorage:
    return storage.NumpyStorage()


@pytest.fixture
def outputs() -> tuple[output.MCMCOutput]:
    acceptance_rate_output = output.Acceptance()
    c0_output = output.SimplifiedOutput(output.ComponentQoI(0), output.IdentityStatistic())
    running_mean_c0_output = output.SimplifiedOutput(
        output.ComponentQoI(0), output.RunningMeanStatistic()
    )
    batch_mean_c0_output = output.SimplifiedOutput(
        output.ComponentQoI(0), output.BatchMeanStatistic(1000)
    )
    return (acceptance_rate_output, c0_output, running_mean_c0_output, batch_mean_c0_output)


@pytest.fixture
def sampler_settings() -> sampling.SamplerRunSettings:
    return sampling.SamplerRunSettings(
        num_samples=10000,
        initial_state=np.array([-0.5, 0.2]),
        print_interval=500,
        store_interval=1,
    )


@pytest.fixture
def logger(tmp_path: Path) -> logging.MCMCLogger:
    logfile_path = tmp_path / "logfile_test.log"
    logger_settings = logging.LoggerSettings(
        do_printing=False,  # Disable printing for test
        logfile_path=logfile_path,
    )
    return logging.MCMCLogger(logger_settings)


def test_banana_model_pCN(
    sample_storage: storage.MCMCStorage,
    outputs: tuple[output.MCMCOutput],
    sampler_settings: sampling.SamplerRunSettings,
    logger: logging.MCMCLogger,
) -> None:
    """Integration test: pCN sampling on BananaModel and output collection."""
    posterior_model = BananaModel()
    algorithm = algorithms.pCNAlgorithm(posterior_model, step_width=0.4)
    sampler = sampling.Sampler(algorithm, sample_storage, outputs, logger)

    samples, outputs_result = sampler.run(sampler_settings)

    # Basic checks
    assert hasattr(samples, "values")
    assert isinstance(samples.values, np.ndarray)
    assert samples.values.shape[0] == sampler_settings.num_samples, (
        "Not enough samples were generated"
    )
    assert samples.values.shape[1] == 2, "Samples have the wrong shape"

    # Check that outputs were computed
    assert len(outputs_result) == 4


def test_banana_model_MALA(
    sample_storage: storage.MCMCStorage,
    outputs: tuple[output.MCMCOutput],
    sampler_settings: sampling.SamplerRunSettings,
    logger: logging.MCMCLogger,
) -> None:
    """Integration test: MALA sampling on BananaModel and output collection."""
    posterior_model = BananaModel()
    algorithm = algorithms.MALAAlgorithm(posterior_model, step_width=0.4)
    sampler = sampling.Sampler(algorithm, sample_storage, outputs, logger)

    samples, outputs_result = sampler.run(sampler_settings)

    # Basic checks
    assert hasattr(samples, "values")
    assert isinstance(samples.values, np.ndarray)
    assert samples.values.shape[0] == sampler_settings.num_samples, (
        "Not enough samples were generated"
    )
    assert samples.values.shape[1] == 2, "Samples have the wrong shape"

    # Check that outputs were computed
    assert len(outputs_result) == 4


def test_gauss_model_pCN(
    sample_storage: storage.MCMCStorage,
    outputs: tuple[output.MCMCOutput],
    sampler_settings: sampling.SamplerRunSettings,
    logger: logging.MCMCLogger,
) -> None:
    """Integration test: pCN on GaussModel; verifies sample mean and covariance."""
    mean = np.array([1, 2])
    cov = np.array([[1, 0], [0, 1]])
    posterior_model = GaussModel(mean=mean, cov=cov)
    algorithm = algorithms.pCNAlgorithm(posterior_model, step_width=0.1)
    sampler = sampling.Sampler(algorithm, sample_storage, outputs, logger, seed=0)

    samples, outputs_result = sampler.run(sampler_settings)

    # Basic checks
    assert hasattr(samples, "values")
    assert isinstance(samples.values, np.ndarray)
    assert samples.values.shape[0] == sampler_settings.num_samples, (
        "Not enough samples were generated"
    )
    assert samples.values.shape[1] == 2, "Samples have the wrong shape"

    # Check that outputs were computed
    assert len(outputs_result) == 4

    # True sampling check
    sample_mean = np.mean(samples.values, axis=0)
    mean_msg = (
        "Mean too far from true mean, maybe stochastic error. "
        f"True mean: {mean}, sample mean: {sample_mean}"
    )
    assert np.linalg.norm(sample_mean - mean) < 0.2, mean_msg
    sample_cov = np.cov(samples.values, rowvar=False)
    cov_msg = (
        "Mean too far from true mean, maybe stochastic error. "
        f"True cov: {cov}, sample mean: {sample_cov}"
    )
    assert np.linalg.norm(sample_cov - cov) < 0.3, cov_msg


def test_gauss_model_MALA(
    sample_storage: storage.MCMCStorage,
    outputs: tuple[output.MCMCOutput],
    sampler_settings: sampling.SamplerRunSettings,
    logger: logging.MCMCLogger,
) -> None:
    """Integration test: MALA on GaussModel; verifies sample mean and covariance."""
    # Set up sampler
    mean = np.array([1, 2])
    cov = np.array([[1, 0], [0, 1]])
    sample_storage = storage.NumpyStorage()
    posterior_model = GaussModel(mean=mean, cov=cov)
    algorithm = algorithms.MALAAlgorithm(posterior_model, step_width=0.1)
    sampler = sampling.Sampler(algorithm, sample_storage, outputs, logger, seed=0)

    samples, outputs_result = sampler.run(sampler_settings)

    # Basic checks
    assert hasattr(samples, "values")
    assert isinstance(samples.values, np.ndarray)
    assert samples.values.shape[0] == sampler_settings.num_samples, (
        "Not enough samples were generated"
    )
    assert samples.values.shape[1] == 2, "Samples have the wrong shape"

    # Check that outputs were computed
    assert len(outputs_result) == 4

    # true sampling check
    assert np.linalg.norm(np.mean(samples.values, axis=0) - mean) < 0.2, (
        f"mean too far from true mean, maybe stochastic error. \
            True mean: {mean}, sample cov: {np.mean(samples.values, axis=0)}"
    )
    assert np.linalg.norm(np.cov(samples.values, rowvar=False) - cov) < 0.3, (
        f"cov too far from true cov, maybe stochastic error. \
            True cov: {cov}, sample cov: {np.cov(samples.values, rowvar=False)}"
    )
