import numpy as np
import matplotlib.pyplot as plt

#Define a matrix
A = np.array([[1,3,2],[2,1,3],[3,2,1]])
print(A)

#Transpose a matrix
A_T = A.T
print(A_T)

#Determinant (only for square matrices)
det_A = np.linalg.det(A)
print(det_A)

#Inverse (if determinant is not zero)
try:
    A_inv = np.linalg.det(A)
    print(A_inv)
except np.linalg.LinAlgError:
    print("\nMatrix A is not invertible(det = 0).")

#Define the coefficient matrix (C) and constants (B)
B = np.array([[8,9],[1,7]])
C = np.array([6,4])

#Solve for X
X = np.linalg.solve(B,C)
print(X)

#Visualize a matrix using matplotlib
plt.matshow(B, cmap="coolwarm")
plt.title("Matrix Visualization")
plt.colorbar()
plt.show()
