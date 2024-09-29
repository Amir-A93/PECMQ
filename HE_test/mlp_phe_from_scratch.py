import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from phe import paillier

# Global variable for the number of encrypted weights
N = 10

# Linear approximation for ReLU
def approx_relu(x, alpha=1.0, beta=0.0):
    return np.where(x < 0, beta, alpha * x + beta)

def approx_relu_derivative(x, alpha=1.0):
    return np.where(x < 0, 0, alpha)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# Encrypt the first layer weights
def encrypt_first_layer(W1, public_key):
    encrypted_weights = np.empty_like(W1, dtype=object)
    count = 0
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            if count < N:
                encrypted_weights[i, j] = public_key.encrypt(W1[i, j])
                count += 1
            else:
                encrypted_weights[i, j] = W1[i, j]
    count = 0
    return encrypted_weights

def encrypted_dot_product(X, encrypted_W1, public_key):
    """Compute dot product using encrypted weights."""
    num_samples = X.shape[0]
    num_hidden = encrypted_W1.shape[1]
    Z1 = np.zeros((num_samples, num_hidden), dtype=object)
    
    for i in range(num_samples):
        for j in range(num_hidden):
            print("processing element " + str((num_samples * i)+j) + " out of " + str(num_hidden * num_samples) )
            # Compute the dot product for each element of the output
            sum_enc = public_key.encrypt(0)  # Initialize encrypted sum
            for k in range(X.shape[1]):
                sum_enc = sum_enc + X[i, k] * encrypted_W1[k, j]
            Z1[i, j] = sum_enc
    return Z1

def forward_encrypted(X, encrypted_W1, b1, W2, b2, public_key, private_key):
    """Forward pass with encrypted weights."""
    # Perform the dot product using encrypted weights
    Z1 = encrypted_dot_product(X, encrypted_W1, public_key)
    
    # Add biases (we assume biases are not encrypted)
    Z1 += b1
    
    # Apply activation function (approximated in plaintext)
    A1 = approx_relu(Z1)
    
    # Perform dot product for second layer (assuming W2 is not encrypted)
    Z2 = np.dot(A1, W2) + b2
    
    return A1, Z2


# Decrypt and apply softmax
def softmax_encrypted(X, encrypted_output, private_key):
    decrypted_output = np.array([private_key.decrypt(val) for val in encrypted_output.flatten()]).reshape(encrypted_output.shape)
    return softmax(decrypted_output)

# Load MNIST dataset
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data
    y = mnist.target.astype(int)
    return X, y

# Preprocess data
def preprocess_data(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.values.reshape(-1, 1))
    
    return X, y

# Initialize weights
def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Forward pass
def forward_pass(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = approx_relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return A1, A2

# Backward pass and update weights
def backward_pass(X, y, A1, A2, W1, W2, b1, b2, learning_rate):
    m = X.shape[0]
    
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * approx_relu_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    return W1, b1, W2, b2

# Train the model
def train(X_train, y_train, X_val, y_val, input_size, hidden_size, output_size, epochs, learning_rate):
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        A1, A2 = forward_pass(X_train, W1, b1, W2, b2)
        W1, b1, W2, b2 = backward_pass(X_train, y_train, A1, A2, W1, W2, b1, b2, learning_rate)
        
        if epoch % 10 == 0:
            val_A1, val_A2 = forward_pass(X_val, W1, b1, W2, b2)
            val_accuracy = accuracy_score(np.argmax(y_val, axis=1), np.argmax(val_A2, axis=1))
            print(f'Epoch {epoch}, Validation Accuracy: {val_accuracy:.4f}')
    
    return W1, b1, W2, b2

# Main function
def main():
    # Load and preprocess data
    X, y = load_mnist()
    X, y = preprocess_data(X, y)
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = y_train.shape[1]
    epochs = 50
    learning_rate = 0.01
    
    # Train the model
    W1, b1, W2, b2 = train(X_train, y_train, X_val, y_val, input_size, hidden_size, output_size, epochs, learning_rate)
    
    # Generate Paillier keys
    public_key, private_key = paillier.generate_paillier_keypair()
    
    print("encryption started")
    # Encrypt the first layer weights
    encrypted_W1 = encrypt_first_layer(W1, public_key)
    
    print("forward encryption started")
    # Evaluate the model
    val_A1, val_A2 = forward_encrypted(X_val, encrypted_W1, b1, W2, b2, public_key, private_key)
    print("softmax encryption started")
    val_A2 = softmax_encrypted(X_val, val_A2, private_key)
    val_accuracy = accuracy_score(np.argmax(y_val, axis=1), np.argmax(val_A2, axis=1))
    print(f'Final Validation Accuracy: {val_accuracy:.4f}')

if __name__ == '__main__':
    main()
