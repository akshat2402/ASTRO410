import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
import math
import scipy.optimize as opt

dat = pd.read_csv("hw2_fitting.dat", sep=" ", header=None)

def model(params, x):
    alpha_L, nu0= params
    y = (1./np.pi)*(alpha_L/((x - nu0)**2. + alpha_L**2.))
    return y

# Define the residual function to be minimized
def residuals(params, x, y):
    return y - model(params, x)

# Generate some sample data to fit
xdata = dat[0]
ydata = dat[1]

# Set up the initial guess for the parameters
params0 = [7.55, 46.27]

# Set up the Levenberg-Marquardt algorithm parameters
lamda = 0.001
max_iter = 100
tolerance = 1e-6

# Run the optimization using the Levenberg-Marquardt algorithm
params = params0
for i in range(max_iter):
    jacobian = np.array([
        np.exp(-params[1] * xdata),
        -params[0] * xdata * np.exp(-params[1] * xdata),
        np.ones(len(xdata))
    ]).T
    residuals_vector = residuals(params, xdata, ydata)
    grad = jacobian.T @ residuals_vector
    hessian = jacobian.T @ jacobian + lamda * np.diag(jacobian.T @ jacobian)
    delta = np.linalg.solve(hessian, grad)
    new_params = params + delta
    new_residuals = residuals(new_params, xdata, ydata)
    new_cost = np.sum(new_residuals ** 2)
    old_cost = np.sum(residuals_vector ** 2)
    if new_cost < old_cost:
        lamda /= 10
        params = new_params
        if np.abs(new_cost - old_cost) < tolerance:
            break
    else:
        lamda *= 10

# Print the results
print(params)

# Plot the data and the fitted function
plt.plot(xdata, ydata, 'bo')
plt.plot(xdata, model(params, xdata), 'r-')
plt.show()