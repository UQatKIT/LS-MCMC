import matplotlib.pyplot as plt
import numpy as np

from ls_mcmc import model


# ==================================================================================================
class BananaModel(model.DifferentiableMCMCModel):
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

    @staticmethod
    def evaluate_gradient_of_potential(state: np.ndarray) -> np.ndarray:
        x, y = state[0], state[1]
        # d/dx
        dV_dx = 10 * 2 * (x**2 - y) * 2 * x - x
        # d/dy
        dV_dy = 10 * 2 * (x**2 - y) * (-1) + 4 * y**3 - y
        return np.array([dV_dx, dV_dy])

    def compute_preconditioner_sqrt_action(self, state: np.ndarray) -> np.ndarray:
        action = self._preconditioner_sqrt_matrix @ state
        return action

    @property
    def reference_point(self) -> np.ndarray:
        return self._reference_point


# ==================================================================================================
def evaluate_density(
    x_value: np.ndarray[tuple[int], np.dtype[float]],
    y_value: np.ndarray[tuple[int], np.dtype[float]],
) -> float:
    logp = 10 * np.square(np.square(x_value) - y_value) + np.power(y_value, 4)
    probability = np.exp(-logp)
    return probability


def plot_density(
    x_value: np.ndarray[tuple[int], np.dtype[float]],
    y_value: np.ndarray[tuple[int], np.dtype[float]],
    density: np.ndarray[tuple[int], np.dtype[float]],
) -> None:
    _, ax = plt.subplots(figsize=(4, 4), layout="constrained")
    ax.contourf(x_value, y_value, density, cmap="Blues", levels=50)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    plt.show()
