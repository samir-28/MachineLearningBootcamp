import numpy as np
from sklearn.linear_model import LinearRegression

# Given data points
x = np.array([2, 4, 6, 8, 10]).reshape(-1, 1)  # Reshaping for sklearn input
y = np.array([6, 5, 4, 3, 2])

# Creating the linear regression model
model = LinearRegression()

# Fitting the model with the data
model.fit(x, y)

# Getting the coefficients
b0 = model.intercept_  # Intercept (b0)
b1 = model.coef_[0]    # Slope (b1)

# Displaying the results with two digits after the decimal
print(f'Calculated coefficients: b0 = {b0:.2f}, b1 = {b1:.2f}')
print(f'Equation of the line: Y = {b0:.2f} + {b1:.2f}X')
