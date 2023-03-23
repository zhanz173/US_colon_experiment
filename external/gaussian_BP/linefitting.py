import numpy as np
from .factor_graph import FactorGraph, GBPSettings
from .gaussian import HuberLoss, MeasModel, SquaredLoss
import matplotlib.pyplot as plt

gbp_settings = GBPSettings(
    damping = 0.1,
    beta = 0.01,
    num_undamped_iters = 1,
    min_linear_iters = 1,
    dropout = 0.0,
)

def height_meas_fn(x: np.ndarray, gamma: np.ndarray):
    gamma = gamma.squeeze()
    J = np.array([1-gamma, gamma])
    return J @ x.reshape(-1,1)

def height_jac_fn(x: np.array, gamma: np.array):
    gamma = gamma.squeeze()
    return np.array([[1-gamma, gamma]])

def smooth_meas_fn(x: np.array):
    return np.array([x[1] - x[0]])

def smooth_jac_fn(x: np.array):
    return np.array([[-1., 1.]])

class HeightMeasurementModel(MeasModel):
    def __init__(self, loss: HuberLoss, gamma: np.array) -> None:
        MeasModel.__init__(self, height_meas_fn, height_jac_fn, loss, gamma)
        self.linear = True


class SmoothingModel(MeasModel):
    def __init__(self, loss: HuberLoss) -> None:
        MeasModel.__init__(self, smooth_meas_fn, smooth_jac_fn, loss)
        self.linear = True


class LineFittingModel():
    def __init__(self, n_measurements: int,prior_cov:np.ndarray, smooth_cov:np.ndarray, data_cov:np.ndarray):
        self.fg = FactorGraph(gbp_settings)
        self.n_varnodes = n_measurements+1
        self.prior_cov = prior_cov
        self.smooth_cov = smooth_cov
        self.data_cov = data_cov

        self._create_FactorGraph()
        self.fg.print(brief=True)

    def update_measurement(self,meas:np.ndarray):
        self.mean = np.mean(meas)
        self.fg.update_measurement_factor(meas-self.mean)

    def smooth(self,measuremnets,n_iters=10)-> np.ndarray:
        self.update_measurement(measuremnets)
        self.fg.gbp_solve(n_iters)
        return self.fg.belief_means() + self.mean

    def _create_FactorGraph(self) -> None :
        for i in range(self.n_varnodes):
            self.fg.add_var_node(1, np.array([0.]), self.prior_cov)

        for i in range(self.n_varnodes-1):
            self.fg.add_factor([i, i+1], np.array([0.]), 
            SmoothingModel(HuberLoss(1, self.smooth_cov, 1)),
            {'type': 'smooth'}
            )

        for i in range(self.n_varnodes-1):
            self.fg.add_factor(
            [i, i+1], np.array([150]), 
            HeightMeasurementModel(
                HuberLoss(1, self.data_cov,1), 
                np.array([0.5]),
                ),
            {'type': 'measurement'}  
            )

if __name__ == "__main__":
    prior_cov = np.array([10.0])
    data_cov = np.array([2]) 
    smooth_cov = np.array([1])
    data_std = np.sqrt(data_cov)
    n_measurements=9

    meas_x = np.linspace(0,9,n_measurements)
    meas_y = np.array([156,155,155,158,134,135,134,157,133],dtype=np.float32)
    uut = LineFittingModel(n_measurements,prior_cov,smooth_cov,data_cov)
    plt.plot(meas_x, meas_y)
    plt.show()

    means = uut.smooth(meas_y)
    plt.plot( means)
    plt.show()