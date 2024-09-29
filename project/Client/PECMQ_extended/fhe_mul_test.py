import phe as paillier
import numpy as np

# Generate random binary vectors
vector_size = 2  # You can adjust the size as needed
vector1 = [1,0]
vector2 = [1,1]

print(f"Vector 1: {vector1}")
print(f"Vector 2: {vector2}")

# Generate Paillier keypair
public_key, private_key = paillier.generate_paillier_keypair()

# Encrypt the first vector
encrypted_vector1 = [public_key.encrypt(x) for x in vector1]
encrypted_vector2 = [public_key.encrypt(x) for x in vector1]
# Print the encrypted vector
# print("Encrypted Vector 1:")
# for enc in encrypted_vector1:
#     print(enc.ciphertext())

# Multiply the encrypted vector with the second binary vector
encrypted_result = [enc * x for enc, x in zip(encrypted_vector1, vector2)]
encrypted_result2 = [enc * x for enc, x in zip(encrypted_vector2, vector1)]
# Decrypt and print the result
decrypted_result = [private_key.decrypt(enc) for enc in encrypted_result]
decrypted_result2 = [private_key.decrypt(enc) for enc in encrypted_result2]
print(f"Decrypted result: {decrypted_result}")
print(decrypted_result == decrypted_result2)
print(encrypted_result == encrypted_result2)
