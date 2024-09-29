import numpy as np
from phe import paillier

# Step 1: Generate Paillier public and private keys
public_key, private_key = paillier.generate_paillier_keypair()

# Step 2: Define the matrices
x = np.random.rand(10, 10)  # Replace with your 10x10 matrix of floating point values
y = np.random.rand(10, 10)  # Replace with your 10x10 matrix of unencrypted values

# Perform the matrix multiplication with unencrypted x and y
direct_result = np.dot(x, y)

# Step 3: Encrypt the matrix x
encrypted_x = np.array([[public_key.encrypt(value) for value in row] for row in x])

# Step 4: Multiply encrypted matrix x with unencrypted matrix y
encrypted_result = np.array([[encrypted_x[i][j] * y[j, i] for j in range(10)] for i in range(10)])

# Step 5: Decrypt the result
decrypted_result = np.array([[private_key.decrypt(encrypted_result[i, j]) for j in range(10)] for i in range(10)])

# Print the results
print("Original Matrix x:")
print(x)
print("\nMatrix y:")
print(y)
print("\nEncrypted result (decrypted):")
print(decrypted_result)
print("\nUn-encrypted result:")
print(direct_result)
