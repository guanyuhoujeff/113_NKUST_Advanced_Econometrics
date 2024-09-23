import numpy as np
from scipy.linalg import lu

# Define matrix A
A = np.array(
    [[5, 9], 
     [9, 7]]
)

# 1(a) Inverse of A
A_inv = np.linalg.inv(A)
print('Inverse of A:')
print(A_inv)

# 1(b) Eigen values of A
eigen_values, _ = np.linalg.eig(A)
print(f'Eigen values of A: {eigen_values}')


print('\n\n-------------------------------------\n\n')

# 2. Solve Ax = b where b = [2, 3]
b = np.array([2, 3])
x = np.linalg.solve(A, b)
print('Solution of Ax = b: x =')
print(x)

print('\n\n-------------------------------------\n\n')


# 3. Find L and U of A using LU decomposition
_, L, U = lu(A)
print('L matrix:')
print(L)
print()
print('U matrix:\n{U}')