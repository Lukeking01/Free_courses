

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#relative filepath for datafile
path = "BERN02/labs/pollution_cleaneddata.csv"

# Read the data
file = pd.read_csv(path)

sorted_file = file.sort_values("POOR")

#Filter out the mortality rate
mortality = np.array(sorted_file["MORT"])
poor = np.array(sorted_file["POOR"])


def regression(x_0,k):

    def find_k_nearest_neighbours(mort, poor, x_0, k):
        distances = np.abs(poor - x_0)

        # Indices of the k smallest distances
        nearest_indices = np.argsort(distances)[:k]

        # Get the corresponding values
        nearest_poor = poor[nearest_indices]
        nearest_mort = mort[nearest_indices]
        nearest_distances = distances[nearest_indices]

        return nearest_mort, nearest_poor, nearest_distances

    nearest_mort, nearest_poor, nearest_distances = find_k_nearest_neighbours(mortality, poor,x_0,k)

    #let the weights be the inverse distances to the point
    weights = 1/(nearest_distances)

    def to_be_minimized(beta, weights, mort, poor):
        beta_0, beta_1 = beta

        residuals = mort - beta_0 - beta_1 * poor

        return np.sum(weights * residuals**2)


    def f(beta_0, beta_1, x):
        return beta_0 + beta_1 * x
            
    result = minimize(
        to_be_minimized,
        x0=[0, 0],  # initial guess for beta_0 and beta_1
        args=(weights, nearest_mort, nearest_poor)
    )

    beta_0, beta_1 = result.x

    return f(beta_0, beta_1, x_0)
plt.scatter(poor,mortality,s=6,c="Gray")

for k in [10,18,25]:
    x_s = np.linspace(10,25,30)
    morts = []
    for x in x_s:
        morts.append(regression(x,k))

    plt.plot(x_s,morts,label=f"k={k}")
plt.legend()
plt.savefig("regression.jpeg")
plt.show()

