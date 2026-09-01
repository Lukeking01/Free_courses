

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

def Regression(y,x,k,x_0):
    def local_regression(mortality,poor,k,x_0):

        def find_k_nearest_neighbours(mort, poor, x_0, k):
            '''
            Finds the k nearest neighbours and returns them in a list.
            Also returns the corresponding poor values, and the distances to them.
            '''
            distances = np.abs(poor - x_0)

            # Indices of the k smallest distances
            nearest_indices = np.argsort(distances)[:k]

            # Get the corresponding values
            nearest_poor = poor[nearest_indices]
            nearest_mort = mort[nearest_indices]
            nearest_distances = distances[nearest_indices]

            return nearest_mort, nearest_poor, nearest_distances

        def RSS(beta, weights, mort, poor):
            '''
            Calculates the Residual Sum of Squares for given quantities.
            '''
            beta_0, beta_1 = beta

            residuals = mort - beta_0 - beta_1 * poor

            return np.sum(weights * residuals**2)

        def f(beta, x):
            '''
            Calculates the predicted value.
            '''
            beta_0, beta_1 = beta
            return beta_0 + beta_1 * x
        
        # Find nearest neighbours
        nearest_mort, nearest_poor, nearest_distances = find_k_nearest_neighbours(mortality, poor,x_0,k)
        
        # Let the weights be the inverse distances to the point
        weights = 1/(nearest_distances+1e-7)
        
        # Minimize the RSS
        result = minimize(
            RSS,
            x0=[0, 0],  # initial guess for beta_0 and beta_1
            args=(weights, nearest_mort, nearest_poor)
        )
        beta = result.x

        return f(beta, x_0)
    
    def MSE(mort, poor, k):
        predictions = []

        for p in poor:
            predictions.append(local_regression(mort, poor, k, p))

        predictions = np.array(predictions)

        errors = mort - predictions
        return np.mean(errors**2)
    
    p_s = x_0
    pred = []
    for p in p_s:
        pred.append(local_regression(y,x,k,p))
    
    se = MSE(y,x,k)
    plt.scatter(x,y,c="Gray")
    plt.plot(p_s,pred,label=f"k={k}")
    plt.legend()
    plt.savefig("regression.jpeg")
    plt.show()
    
    return pred, se

k= 25 #10 18 25
x_0 = np.linspace(10,25,30)
pred,se = Regression(mortality,poor,k,x_0)
print("Prediction is")
print(pred)
print("Standard error is")
print(se)
print("compare with")
print(np.mean((mortality - np.mean(mortality))**2))


