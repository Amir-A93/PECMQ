import random
import galois
import numpy as np
from reedsolo import RSCodec
import sys

def generate_random_binary_array(length):
    """Generate a random binary array of the specified length."""
    return [random.randint(0, 1) for _ in range(length)]

def binary_to_bytes(binary_array):
    """Convert a binary array to a list of bytes."""
    byte_array = bytearray()
    for i in range(0, len(binary_array), 8):
        byte = binary_array[i:i+8]
        byte_value = int(''.join(map(str, byte)), 2)
        # byte_value = int.to_bytes(byte_value,8,sys.byteorder)
        byte_array.append(byte_value)
    return bytes(byte_array)

def bytes_to_binary(byte_array):
    """Convert a list of bytes back to a binary array."""
    binary_array = []
    for byte in byte_array:
        binary_array.extend([int(bit) for bit in format(byte, '08b')])
    return binary_array


def encode_with_reed_solomon(binary_array, n_parity_symbols):
    """Encode the binary array using Reed-Solomon ECC."""
    print("RS code creation begins")
    rsc = RSCodec(n_parity_symbols,)
    byte_data = binary_to_bytes(binary_array)
    print("RS encoding begins")
    encoded_data = rsc.encode(byte_data)
    encoded_binary_array = bytes_to_binary(encoded_data)
    return encoded_binary_array

def decode_with_reed_solomon(encoded_binary_array, n_parity_symbols):
    """Decode a Reed-Solomon encoded binary codeword."""
    rsc = RSCodec(n_parity_symbols)
    byte_data = binary_to_bytes(encoded_binary_array)
    decoded_data = rsc.decode(byte_data)
    decoded_binary_array = bytes_to_binary(list(decoded_data[0]))
    return decoded_binary_array

def encode_with_bch(binary_array):
    n =  31 # Codeword length
    k = 21  # Message length
    # t = 3   # Error correction capability
    """Encode the binary array using BCH ECC with Galois library."""
    gf = galois.GF(2)  # Binary field
    print("BCH code creation begins")
    bch = galois.BCH(n=n,k=k,field=gf)
    
    msg = np.array(binary_array)
    mod = len(msg)%k
    if(mod != 0):
        msg = np.append(msg,np.zeros(k - mod))

    msg =  msg.reshape((int(len(msg)/k),k))
    msg = np.array(msg,dtype=np.uint8)
   
    # Convert binary array to an array of integers
    # byte_data = binary_to_bytes(binary_array)
    # input_data = np.array(binary_array,dtype=np.uint8)

    # Encode using BCH
    print("BCH encoding begins")
    encoded_data = bch.encode(msg)
    #.reshape(n*msg.shape[0])
    # Convert encoded data to binary array
    # encoded_binary_array = np.unpackbits(encoded_data, bitorder='little').tolist()
    return encoded_data


def decode_bch(encoded_message):
    # Decode the encoded message

    n = 31  # Codeword length
    k = 21  # Message length
    # t = 3   # Error correction capability
    """Encode the binary array using BCH ECC with Galois library."""
    # gf = galois.GF(2)  # Binary field
    print("BCH code creation begins")
    bch = galois.BCH(n=n,k=k)

    decoded_message = bch.decode(encoded_message)
    return decoded_message


def encode_with_reed_muller(binary_array, r=1, m=3):
    """Encode the binary array using Reed-Muller ECC."""
    n = 2 ** m  # Length of the codeword
    k = sum([comb(m, i) for i in range(r + 1)])  # Number of information bits

    # Check if the input length is valid
    # if len(binary_array) != k:
    #     raise ValueError(f"Length of input binary array must be {k} for given r={r} and m={m}.")
    msg = np.array(binary_array)
    mod = len(msg)%k
    if(mod != 0):
        msg = np.append(msg,np.zeros(k - mod))

    msg =  msg.reshape((int(len(msg)/k),k))
    msg = np.array(msg,dtype=np.uint8)

    # Generate the generator matrix for Reed-Muller code
    generator_matrix = []
    for i in range(k):
        row = [(1 if bin(i & j).count('1') % 2 == 0 else 0) for j in range(n)]
        generator_matrix.append(row)
    
    # Encode using matrix multiplication (binary mod 2)
    encoded_binary_array = np.dot(msg, generator_matrix) % 2
    return list(encoded_binary_array.reshape(encoded_binary_array.shape[0]*encoded_binary_array.shape[1]))

# Helper function to compute combinations
def comb(n, k):
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    return comb(n - 1, k - 1) + comb(n - 1, k)

# Example usage
if __name__ == "__main__":
    # Length of the random binary array (must be a multiple of 8 for proper byte grouping)
    binary_array_length = 32

    # Generate a random binary array
    binary_array = generate_random_binary_array(binary_array_length)
    print("Original Binary Array:", binary_array)

    # Encode the binary array using Reed-Solomon ECC
    # encoded_rs_binary_array = encode_with_reed_solomon(binary_array, 1)
    # print("Encoded Binary Array (Reed-Solomon):", encoded_rs_binary_array)
    # decoded_binary_array = decode_with_reed_solomon(encoded_rs_binary_array, 1)
    # print("Decoded Binary Array (Reed-Solomon):", decoded_binary_array)
    # print(len(encoded_rs_binary_array))

    # Encode the binary array using BCH ECC
    encoded_bch_binary_array = encode_with_bch(binary_array)
    print("Encoded Binary Array (BCH):", encoded_bch_binary_array)
    decoded_message = decode_bch(encoded_message=encoded_bch_binary_array)
    print("Decoded Binary Array (BCH):",decoded_message)

    # # Encode the binary array using Reed-Muller ECC
    # r, m = 1, 3  # Reed-Muller parameters
    # k = sum([comb(m, i) for i in range(r + 1)])
    # # binary_array_rm = generate_random_binary_array(k)  # Generate binary array with proper length for Reed-Muller
    # encoded_rm_binary_array = encode_with_reed_muller(binary_array, r, m)
    # print("Encoded Binary Array (Reed-Muller):", encoded_rm_binary_array)
