#%%
from factor_graph import FactorGraph, GBPSettings
from gaussian import HuberLoss, MeasModel, SquaredLoss
import numpy as np
import os
import matplotlib.pyplot as plt

#%%
def height_meas_fn(x: np.ndarray, gamma: np.ndarray):
    gamma = gamma.squeeze()
    J = np.array([1-gamma, gamma])
    return J @ x.reshape(-1,1)

def height_jac_fn(x: np.array, gamma: np.array):
    gamma = gamma.squeeze()
    return np.array([[1-gamma, gamma]])

class HeightMeasurementModel(MeasModel):
    def __init__(self, loss: HuberLoss, gamma: np.array) -> None:
        MeasModel.__init__(self, height_meas_fn, height_jac_fn, loss, gamma)
        self.linear = True

def smooth_meas_fn(x: np.array):
    return np.array([x[1] - x[0]])

def smooth_jac_fn(x: np.array):
    return np.array([[-1., 1.]])

class SmoothingModel(MeasModel):
    def __init__(self, loss: HuberLoss) -> None:
        MeasModel.__init__(self, smooth_meas_fn, smooth_jac_fn, loss)
        self.linear = True



#%%
n_varnodes = 10
x_range = 10
n_measurements = 9

gbp_settings = GBPSettings(
    damping = 0.1,
    beta = 0.01,
    num_undamped_iters = 1,
    min_linear_iters = 1,
    dropout = 0.0,
)
prior_cov = np.array([10])
data_cov = np.array([0.05]) 
smooth_cov = np.array([0.1])
data_std = np.sqrt(data_cov)

#%%
meas_x = np.linspace(0,10,n_measurements)
meas_y = np.sin(meas_x)
plt.scatter(meas_x, meas_y, color="red", label="Measurements", marker=".")
plt.legend()
plt.savefig('gbp-1d-data.pdf')
plt.show()

#%%
fg = FactorGraph(gbp_settings)

xs = np.linspace(0, x_range, n_varnodes).reshape(-1,1)
for i in range(n_varnodes):
    fg.add_var_node(1, np.array([0.]), prior_cov)

for i in range(n_varnodes-1):
    fg.add_factor(
    [i, i+1], 
    np.array([0.]), 
    SmoothingModel(HuberLoss(1, smooth_cov, 1))
    )

for i in range(n_measurements):
    ix2 = np.argmax(xs > meas_x[i])
    ix1 = ix2 - 1
    gamma = (meas_x[i] - xs[ix1]) / (xs[ix2] - xs[ix1])
    fg.add_factor(
    [i, i+1], 
    np.array([0.0]), 
    HeightMeasurementModel(
        HuberLoss(1, data_cov,1), 
        np.array([0.5])  
        ),
        {"type":"measurement"}
    )
fg.print(brief=True)

#%%
covs = np.sqrt(np.concatenate(fg.belief_covs()).flatten())
plt.errorbar(xs, fg.belief_means(), yerr=covs, fmt='o', color="C0", label='Beliefs')
plt.scatter(meas_x, meas_y, color="red", label="Measurements", marker=".")
plt.legend()
plt.show()

#%%
fg.update_measurement_factor(meas_y)
fg.gbp_solve(n_iters=10)
covs = np.sqrt(np.concatenate(fg.belief_covs()).flatten())
plt.errorbar(xs, fg.belief_means(), yerr=covs, fmt='o', color="C0", label='Beliefs')
plt.scatter(meas_x, meas_y, color="red", label="Measurements", marker=".")
plt.legend()
plt.savefig('gbp-1d-posteriors.pdf')
plt.show()