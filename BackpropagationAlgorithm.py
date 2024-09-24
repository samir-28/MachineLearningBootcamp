import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Define the XOR input and output
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 0])  # XOR outputs

# Create and train the MLP classifier
model = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, activation='logistic', random_state=42)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Display the learned weights and bias
# The MLPClassifier does not directly expose weights, but we can access them through the model's attributes
weights = model.coefs_
biases = model.intercepts_

print("Learned weights:")
for i, weight_matrix in enumerate(weights):
    print(f"Layer {i} weights:\n{np.round(weight_matrix, 2)}")

print("\nLearned biases:")
for i, bias_vector in enumerate(biases):
    print(f"Layer {i} biases:\n{np.round(bias_vector, 2)}")

# Print predictions and accuracy
print("\nPredictions:")
for input_vec, pred in zip(X, predictions):
    print(f"Input: {input_vec}, Predicted Output: {pred}")

accuracy = accuracy_score(y, predictions)
print(f"\nAccuracy: {accuracy:.2f}")
