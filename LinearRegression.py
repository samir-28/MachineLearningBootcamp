
import numpy as np

# Given data points
x = np.array([2, 4, 6, 8, 10])
y = np.array([6, 5, 4, 3, 2])

# Function to calculate linear regression prediction
def regression(x_val, b0, b1):
    return b0 + b1 * x_val

# Number of data points
n = len(x)

# Calculating the sums
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_xsq = np.sum(x ** 2)

# Calculating coefficients b1 and b0
b1 = (n * sum_xy - sum_x * sum_y) / (n * sum_xsq - sum_x ** 2)
b0 = np.mean(y) - b1 * np.mean(x)

print(f'Calculated coefficients: b0 = {b0}, b1 = {b1}')
print(f'Equation of the line: Y = {b0} + {b1}X')

# Loss function: Mean Squared Error (MSE)  
def calculate_loss(x, y, b0, b1):
    y_pred = regression(x, b0, b1) 
    print(y_pred)# Predictions based on the model
    mse = np.mean((y_pred - y) ** 2)  # Mean Squared Error (MSE)
    return mse

# Gradient descent => finnding minimum loss
# Rule to update: W - (partial derivative) dl /  (partial derivative) dw

# User input for predicion                          
a = int(input("Enter the value of X: "))
predicted_y = regression(a, b0, b1)
print(f'When X = {a}, Y = {predicted_y}')

# Calculate and print the loss (MSE)
loss = calculate_loss(x, y, b0, b1)
print(f'Mean Squared Error (MSE) for the model: {loss}')
