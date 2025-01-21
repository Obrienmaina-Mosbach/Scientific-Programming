# importing libraries
import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


def number_to_string(argument):
    match argument:
        case 1:
            return "Addition"
        case 2:
            return "Subtraction"
        case 3:
            return "Multiplication"
        case 4:
            return "Transpose"
        case _:
            return "Invalid Selection"

# Function to display a matrix
def display_matrix(matrix):
    for row in matrix:
        print(" ".join(map(str, row)))
    print()

# Get matrix input from the user
def input_matrix(rows, columns):
    matrix = []
    print(f"Enter the entries row-wise ({rows} x {columns}):")
    for _ in range(rows):
        matrix.append(list(map(int, input().split())))
    return matrix

# Matrix addition
def add_matrices(matrixA, matrixB):
    return [[matrixA[i][j] + matrixB[i][j] for j in range(len(matrixA[0]))] for i in range(len(matrixA))]

# Matrix subtraction
def subtract_matrices(matrixA, matrixB):
    return [[matrixA[i][j] - matrixB[i][j] for j in range(len(matrixA[0]))] for i in range(len(matrixA))]

# Matrix multiplication
def multiply_matrices(matrixA, matrixB):
    result = [[sum(a * b for a, b in zip(matrixA_row, matrixB_col)) for matrixB_col in zip(*matrixB)] for matrixA_row in matrixA]
    return result

# Matrix transpose
def transpose_matrix(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

# Main logic
RowA = int(input("Enter the number of rows for the first matrix: "))
ColumnA = int(input("Enter the number of columns for the first matrix: "))
matrixA = input_matrix(RowA, ColumnA)
print("Matrix A:")
display_matrix(matrixA)

RowB = int(input("Enter the number of rows for the second matrix: "))
ColumnB = int(input("Enter the number of columns for the second matrix: "))
matrixB = input_matrix(RowB, ColumnB)
print("Matrix B:")
display_matrix(matrixB)

print("1. Addition\n2. Subtraction\n3. Multiplication\n4. Transpose\n")
onPress = int(input("Input a number to perform an operation: "))
operation = number_to_string(onPress)
print(f"Selected Operation: {operation}")

if onPress == 1:
    if RowA == RowB and ColumnA == ColumnB:
        result = add_matrices(matrixA, matrixB)
        print("Result (Addition):")
        display_matrix(result)
    else:
        print("Addition not possible: Matrices must have the same dimensions.")

elif onPress == 2:
    if RowA == RowB and ColumnA == ColumnB:
        result = subtract_matrices(matrixA, matrixB)
        print("Result (Subtraction):")
        display_matrix(result)
    else:
        print("Subtraction not possible: Matrices must have the same dimensions.")

elif onPress == 3:
    if ColumnA == RowB:
        result = multiply_matrices(matrixA, matrixB)
        print("Result (Multiplication):")
        display_matrix(result)
    else:
        print("Multiplication not possible: Columns of Matrix A must equal rows of Matrix B.")

elif onPress == 4:
    print("Transpose of Matrix A:")
    display_matrix(transpose_matrix(matrixA))
    print("Transpose of Matrix B:")
    display_matrix(transpose_matrix(matrixB))

else:
    print("Invalid Selection")
