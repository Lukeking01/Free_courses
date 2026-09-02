
'''
Exercise 2: Local Regression by Lukas Nord.

File should be run in a directory containing the labs folder, which contains the pollution_cleaneddata.csv file, as well as the Exercises folder, which contains this file. The output will be saved in the Exercises folder. This is because of relative filepaths. If you want to change this, change the path variable and the save_path variable.

The file returns a plot of the local regression for k=5, k=12, and k=20, as well as the predictions for x_0 = 10, 18, and 25. The predictions are also printed to the console and should be:

k = 5
POOR = [10 18 25]:
 Prediction = [883.9865831623683, 951.4378129739152, 1037.2409739518134],
 Error = [42.70326526379657, 91.31501098027093, 51.181225961161104]
k = 12
POOR = [10 18 25]:
 Prediction = [890.3926998614686, 958.7894294090489, 1016.3352236277847],
 Error = [59.530852932089836, 44.83942938688078, 66.80277508740089]
k = 20
POOR = [10 18 25]:
 Prediction = [897.8195076138318, 957.776296048053, 1011.7944153545641],
 Error = [79.66505909774936, 58.35093183148004, 66.37510953919103]
'''

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# relative filepath for datafile
path = "labs\\pollution_cleaneddata.csv"
# relative filepath for saving the figure
save_path = "Exercises\\regression.jpeg"

# Read the data
file = pd.read_csv(path)

sorted_file = file.sort_values("POOR")

#Filter out the mortality rate
mortality = np.array(sorted_file["MORT"])
poor = np.array(sorted_file["POOR"])

# Colors for plotting 
colors = ["blue", "green", "red"]

def Regression(y,x,k,x_0):
    '''
    Performs local regression for a given k and a list of x_0 values.
    Returns the predictions and the standard errors.
    '''
    def local_regression(mortality,poor,k,x_0):
        '''
        Performs local regression for a single given k and x_0.
        Returns the prediction and the standard error.
        '''

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
        
        # Let the weights be a gaussian distribution with mean 0 and standard deviation equal to the standard deviation of the distances
        weights = np.exp(-nearest_distances**2 / (2 * np.std(nearest_distances)**2))
        
        # Minimize the RSS
        result = minimize(
            RSS,
            x0=[0, 0],  # initial guess for beta_0 and beta_1
            args=(weights, nearest_mort, nearest_poor)
        )
        beta = result.x
        fitted = f(beta,nearest_poor)
        errors = nearest_mort - fitted
        variance = np.mean(errors**2)
        se = np.sqrt(variance)
        return f(beta, x_0), se
    
   
    pred = []
    errors = []
    for _p in x_0:
        p,e = local_regression(y,x,k,_p)
        pred.append(p.item())
        errors.append(e.item())
        
    
    return pred, errors


x_0 = np.array([10,18,25]) 
many = np.linspace(9,27,90) # Used for plotting the regression line
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, k in enumerate([5,12,20]):
    # Finds the predictions and errors
    pred,errors = Regression(mortality,poor,k,x_0)
    many_pred, many_errors = Regression(mortality,poor,k,many)

    # Plots the predictions and the regression line
    axes[i].scatter(x_0,pred,label=f"Prediction for k={k} at x_0={x_0}", color=colors[i], marker="x", s=100)

    axes[i].scatter(poor,mortality,c="Gray")

    axes[i].plot(many,many_pred,label=f"k={k}", color=colors[i])

    axes[i].fill_between(many, np.array(many_pred) - np.array(many_errors), np.array(many_pred) + np.array(many_errors), color=colors[i], alpha=0.2)

    print(f"k = {k}")
    print(f"POOR = {x_0}:\n Prediction = {pred},\n Error = {errors}")

    axes[i].grid(True)
    axes[i].legend()

# Relative path for saving the figure
plt.savefig(save_path)
plt.show()
