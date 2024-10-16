import numpy as np

# Define the dimensions of the matrix
m = 5
n = 4
d = 3

# Create random matrices
A = np.random.rand(m, d)
B = np.random.rand(n, d)

# Optimized function to compute squared Euclidean distances
def compute_squared_distances_optimized(A, B):
    # Using broadcasting to compute the squared Euclidean distances
    A_exp = A[:, np.newaxis, :]  # Expand A to shape (m, 1, d)
    B_exp = B[np.newaxis, :, :]  # Expand B to shape (1, n, d)
    
    # Compute the pairwise squared differences
    D = np.sum((A_exp - B_exp)**2, axis=2) ** 0.5 # Sum over the last axis (dimension)

    # The following code will also perform a similar operation 
    # Compute the Euclidean norm (L2) across the last axis and square the result
    # D = np.linalg.norm(A_exp - B_exp, axis=2) 

    return D

# Compute the distances
D_optimized = compute_squared_distances_optimized(A, B)

print(D_optimized)
