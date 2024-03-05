import numpy as np
import tensorflow as tf

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Flatten the images
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Normalize pixel values
x_train_normalized = x_train_flat / 255.0
x_test_normalized = x_test_flat / 255.0

# One-hot encode the labels
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Softmax activation function
def softmax(x):
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Forward propagation
def forward(X, W1, b1, W2, b2):
    # Input to hidden layer
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    # Hidden to output layer
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)

    return a1, a2

# Backward propagation
def backward(X, y, a1, a2, W2):
    m = X.shape[0]

    # Compute gradients for the output layer
    delta2 = a2 - y
    dW2 = np.dot(a1.T, delta2) / m
    db2 = np.sum(delta2, axis=0) / m

    # Compute gradients for the hidden layer
    delta1 = np.dot(delta2, W2.T) * (a1 > 0)
    dW1 = np.dot(X.T, delta1) / m
    db1 = np.sum(delta1, axis=0) / m

    return dW1, db1, dW2, db2

# Initialize weights and biases
input_size = 32 * 32 * 3  # Input image size
hidden_size = 128  # Hidden layer size
output_size = 10  # Number of classes

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros(output_size)

# Training parameters
learning_rate = 0.01
num_epochs = 100
batch_size = 128

# Training loop
for epoch in range(num_epochs):
    # Mini-batch training
    for i in range(0, x_train_normalized.shape[0], batch_size):
        # Get mini-batch
        x_batch = x_train_normalized[i:i+batch_size]
        y_batch = y_train_onehot[i:i+batch_size]
        
        # Forward propagation
        a1, a2 = forward(x_batch, W1, b1, W2, b2)
        # Convert probabilities to binary
        threshold = 0.5
        binary_output = (a2 >= threshold).astype(int)
        # print(a2)
        # print(y_batch.astype(int))
        # print(binary_output)

        # Compute loss
        loss = -np.mean(np.log(a2[range(len(y_batch)), np.argmax(y_batch, axis=1)]))
        accuracy = np.mean(binary_output.flatten() == y_batch.flatten())
        
        # Backward propagation
        dW1, db1, dW2, db2 = backward(x_batch, y_batch, a1, a2, W2)
        
        # Update weights and biases
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2


    # Print loss every few epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy*100}")

# Prediction
def predict(X):
    _, a2 = forward(X, W1, b1, W2, b2)
    return np.argmax(a2, axis=1)

# Test accuracy
y_pred = predict(x_test_normalized)
accuracy = np.mean(y_pred == y_test.flatten())
print("Test Accuracy:", accuracy*100)
