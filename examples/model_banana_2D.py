import matplotlib.pyplot as plt
import numpy as np

from lsmcmc import model


# ==================================================================================================
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


# ==================================================================================================
def evaluate_density(x_value, y_value):
    logp = 10 * np.square(np.square(x_value) - y_value) + np.power(y_value, 4)
    probability = np.exp(-logp)
    return probability


def plot_density(x_value, y_value, density):
    _, ax = plt.subplots(figsize=(4, 4), layout="constrained")
    ax.contourf(x_value, y_value, density, cmap="Blues", levels=50)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    plt.show()
