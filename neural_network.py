"""
Neural Network from Scratch
Author: Greko Bazana Enrique

Fully connected neural network implemented using NumPy only.
All forward and backward propagation is computed manually using matrix calculus.
"""

# sklearn is used ONLY for loading and splitting the Iris dataset
# The neural network itself is implemented entirely from scratch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the Iris dataset
data = load_iris()
X = data.data # shape: (150, 4)
y = data.target.reshape(-1, 1) # reshape for encoder

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y) # shape: (150, 3)

# Normalize features (optional)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
X, y_encoded, test_size=0.2, random_state=42
)

# Set network dimensions
input_dim = 4 # number of features
hidden_dim = 8 # size of hidden layer
output_dim = 3 # number of classes


# Initialize weights and biases
np.random.seed(0)
# Optional: Xavier initialization for small networks
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1. / input_dim)
b1 = np.random.randn(1, hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1. / hidden_dim)
b2 = np.random.randn(1, output_dim)

# Activation functions
def relu(a):
    return np.maximum(0, a)

def relu_derivative(a):
    return np.where(a > 0, 1, 0)

# TODO:
def softmax(a): #gpt
    exp_a = np.exp(a - np.max(a, axis=1, keepdims=True))
    return exp_a / np.sum(exp_a, axis=1, keepdims=True)

# Loss function: cross-entropy
def cross_entropy(y_pred, y_true):
    eps = 1e-10
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

# Gradient of loss w.r.t. logits (output preactivation) TODO:
def dL_da2(y_pred, y_true):
    return y_pred - y_true

# Training loop
learning_rate = 0.01 # learning rate
epochs = 10000
loss_history = []
for epoch in range(epochs + 1):

    # Forward pass TODO:
    a1 = X_train @ W1 + b1  # preactivation for hidden layer
    z1 = relu(a1)  # activation for hidden layer
    a2 = z1 @ W2 + b2  # preactivation for output layer
    y_pred = softmax(a2)  # activation output layer (softmax)

    # Compute loss
    loss = cross_entropy(y_pred, y_train)

    # Backpropagation
    da2 = dL_da2(y_pred, y_train) # dL/da2
    dW2 = z1.T @ da2 # dL/dW2
    db2 = np.sum(da2, axis=0, keepdims=True)
    dz1 = da2 @ W2.T # dL/dz1
    da1 = dz1 * relu_derivative(a1) # dL/da1
    dW1 = X_train.T @ da1
    db1 = np.sum(da1, axis=0, keepdims=True)

    # Update parameters
    W1 -= learning_rate * dW1   #Update W1
    b1 -= learning_rate * db1   #Update b1
    W2 -= learning_rate * dW2   #Update W2
    b2 -= learning_rate * db2   #update b2

    loss_history.append(loss)

    '''if loss < 0.01:
        print(f"Early stopping at epoch {epoch}, Loss: {loss:.4f}")
        break'''

    # Print loss every 10 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


# Plot the loss over epochs
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.savefig("results/loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()


# test accuracy evaluation
# Final predictions on test set
a1_test = X_test @ W1 + b1
z1_test = relu(a1_test)
a2_test = z1_test @ W2 + b2
z2_test = softmax(a2_test)

y_pred = softmax(z2_test)

# Convert probabilities to one-hot encoded predictions
y_pred_classes = np.argmax(y_pred, axis=1)
y_pred_one_hot = np.eye(y_pred.shape[1])[y_pred_classes]

#Print predicted vs actual for comparison
print("Final Predicted Output (first 5 samples):")
print(y_pred_one_hot[:5].astype(int))  # Cast to int for cleaner output
print("\nTarget Output (first 5 samples):")
print(y_test[:5].astype(int))  # Already one-hot, so this is readable

'''# Print predicted vs actual classes
print("Predicted class indices (first 5 samples):", y_pred_classes[:5])
print("True class indices (first 5 samples):     ", y_true_classes[:5])

# Compute accuracy
accuracy = np.mean(y_pred_classes == y_true_classes)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
'''