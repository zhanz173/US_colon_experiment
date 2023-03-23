import numpy as np
from typing import List, Callable, Optional, Union

class Gaussian:
    def __init__(self, dim: int, eta: Optional[np.ndarray]=None, lam: Optional[np.ndarray]=None, type: np.dtype = float):
        self.dim = dim

        if eta is not None and len(eta) == dim:
            self.eta = eta.type(type)
        else:
            self.eta = np.zeros(dim, dtype=type)

        if lam is not None and lam.shape == (dim, dim):
            self.lam = lam.type(type)
        else:
            self.lam = np.zeros([dim, dim], dtype=type)

    def mean(self) -> np.ndarray:
        return np.matmul(np.linalg.inv(self.lam), self.eta)

    def cov(self) -> np.ndarray:
        return np.linalg.inv(self.lam)

    def mean_and_cov(self) -> List[np.ndarray]:
        cov = self.cov()
        mean = np.matmul(cov, self.eta)
        return [mean, cov]

    def set_with_cov_form(self, mean: np.ndarray, cov: np.ndarray) -> None:
        self.lam = np.linalg.inv(cov)
        self.eta = self.lam @ mean

"""
    Defines squared loss functions that correspond to Gaussians. 
    Robust losses are implemented by scaling the Gaussian covariance.
"""

class SquaredLoss(): 
    def __init__(self, dofs: int, diag_cov: Union[float, np.ndarray]) -> None:
        """
            dofs: dofs of the measurement
            cov: diagonal elements of covariance matrix
        """
        assert len(diag_cov) == dofs
        mat = np.zeros((dofs, dofs), dtype=diag_cov.dtype)
        mat[:,:] = diag_cov
        self.cov = mat
        self.effective_cov = mat.copy()

    def get_effective_cov(self, residual: np.ndarray) -> None:
        """ Returns the covariance of the Gaussian (squared loss) that matches the loss at the error value. """
        self.effective_cov = self.cov.copy()

    def robust(self) -> bool:
        return not np.array_equal(self.cov, self.effective_cov)


class HuberLoss(SquaredLoss):
    def __init__(self, dofs: int, diag_cov: Union[float, np.ndarray], stds_transition: float) -> None:
        """ 
            stds_transition: num standard deviations from minimum at which quadratic loss transitions to linear 
        """
        SquaredLoss.__init__(self, dofs, diag_cov)
        self.stds_transition = stds_transition

    def get_effective_cov(self, residual: np.ndarray) -> None:
        mahalanobis_dist = np.sqrt(residual @ np.linalg.inv(self.cov) @ residual)
        if mahalanobis_dist > self.stds_transition:
            self.effective_cov = self.cov * mahalanobis_dist**2 / (2 * self.stds_transition * mahalanobis_dist - self.stds_transition**2)
        else:
            self.effective_cov = self.cov.copy()


class TukeyLoss(SquaredLoss):
    def __init__(self, dofs: int, diag_cov: Union[float, np.ndarray], stds_transition: float) -> None:
        """ 
            stds_transition: num standard deviations from minimum at which quadratic loss transitions to constant 
        """
        SquaredLoss.__init__(self, dofs, diag_cov)
        self.stds_transition = stds_transition

    def get_effective_cov(self, residual: np.ndarray) -> None:
        mahalanobis_dist = np.sqrt(residual @ np.linalg.inv(self.cov) @ residual)
        if mahalanobis_dist > self.stds_transition:
            self.effective_cov = self.cov * mahalanobis_dist**2 / self.stds_transition**2
        else:
            self.effective_cov = self.cov.copy()

class MeasModel: 
    def __init__(self, meas_fn: Callable, jac_fn: Callable, loss: SquaredLoss, *args) -> None:
        self._meas_fn = meas_fn
        self._jac_fn = jac_fn
        self.loss = loss
        self.args = args
        self.linear = True

    def jac_fn(self, x: np.ndarray) -> np.ndarray:
        return self._jac_fn(x, *self.args)

    def meas_fn(self, x: np.ndarray) -> np.ndarray:
        return self._meas_fn(x, *self.args)
