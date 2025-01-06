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
            return "Subtruction"
        case 3:
            return "Multiplication"
        case 4:
            return "Transpose"
        case default:
            return "Invalid Selection"
        

RowA = int(input("Enter the number of rows for the first matrix:"))
ColumnA = int(input("Enter the number of columns for the first matrix:"))

# Initialize matrix
matrixA = []
print("Enter the entries row wise:")

# For user input
# A for loop for row entries
for row in range(RowA):    
    a = []
    # A for loop for column entries
    for column in range(ColumnA):   
        a.append(int(input()))
    matrixA.append(a)

for row in range(RowA):
    for column in range(ColumnA):
        print(matrixA[row][column], end=" ")
    print()

print('Dimention: ' + str(RowA) + ' X ' + str(ColumnA))

RowB = int(input("Enter the number of rows for the second matrix:"))
ColumnB = int(input("Enter the number of columns for the second matrix:"))

# Initialize matrix
matrixB = []
print("Enter the entries row wise:")

# For user input
# A for loop for row entries
for row in range(RowB):    
    a = []
    # A for loop for column entries
    for column in range(ColumnB):   
        a.append(int(input()))
    matrixB.append(a)


# For printing the matrix
for row in range(RowB):
    for column in range(ColumnB):
        print(matrixB[row][column], end=" ")
    print()

print('Dimention: ' + str(RowB) + ' X ' + str(ColumnB))

print('1. Addition\n 2. Subtruction \n 3. Multiplication \n 4. Transpose \n')
onPress = int(input('Input a number to perform an operation: '))

head = number_to_string(onPress)
print(head)

