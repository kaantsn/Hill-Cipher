import numpy as np

def matrix_mod_inverse(matrix, mod):
    # Calculate the inverse of the matrix using modular arithmetic
    matrix_inverse = np.linalg.inv(matrix)
    det = int(round(np.linalg.det(matrix)))
    det_inverse = pow(det, -1, mod)
    adjugate = matrix_inverse * det * det_inverse
    matrix_inverse_mod = np.mod(adjugate, mod)
    return matrix_inverse_mod.astype(int)

def matrix_multiply(matrix1, matrix2, mod):
    # Multiply two matrices and apply modular arithmetic
    result = np.dot(matrix1, matrix2)
    result_mod = np.mod(result, mod)
    return result_mod.astype(int)

def text_to_matrix(text, size):
    # Convert text to a matrix
    matrix = [ord(char) - ord('A') for char in text]
    while len(matrix) % size != 0:
        matrix.append(0)
    return np.array(matrix).reshape(-1, size)

def matrix_to_text(matrix):
    # Convert a matrix back to text
    return ''.join([chr(num + ord('A')) for num in matrix.flatten()])

def hill_cipher(plaintext, key_matrix, mode='encrypt'):
    size = len(key_matrix)
    
    if mode == 'encrypt':
        # Encryption: Multiply plaintext matrix with key matrix
        input_matrix = text_to_matrix(plaintext, size)
        output_matrix = matrix_multiply(input_matrix, key_matrix, 26)
        result_text = matrix_to_text(output_matrix)
    elif mode == 'decrypt':
        # Decryption: Multiply ciphertext matrix with inverse of key matrix
        key_matrix_inverse = matrix_mod_inverse(key_matrix, 26)
        input_matrix = text_to_matrix(plaintext, size)
        output_matrix = matrix_multiply(input_matrix, key_matrix_inverse, 26)
        result_text = matrix_to_text(output_matrix)
    else:
        result_text = "Invalid mode. Use 'encrypt' or 'decrypt'."

    return result_text

# Example usage
key_matrix = np.array([[6, 24, 1], [13, 16, 10], [20, 17, 15]])
plaintext = "HELLO"

# Encryption
ciphertext = hill_cipher(plaintext, key_matrix, mode='encrypt')
print(f"Plaintext: {plaintext}")
print(f"Ciphertext: {ciphertext}")

# Decryption
decrypted_text = hill_cipher(ciphertext, key_matrix, mode='decrypt')
print(f"Decrypted Text: {decrypted_text}")
