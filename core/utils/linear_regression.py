import numpy as np

def linear_regression(x, y):
    coefficients = np.polyfit(x, y, 1)
    return coefficients