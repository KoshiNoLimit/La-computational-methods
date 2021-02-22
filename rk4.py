import numpy as np 


def to_LU(A):
    n = A.shape[0]
    L, U = np.eye(n), np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i <= j:
                U[i, j] = A[i, j] - sum([L[i, k] * U[k, j] for k in range(i)])
            else:
                L[i, j] = (A[i, j] - sum([L[i, k] * U[k, j] for k in range(j)])) / U[j, j]

    return L, U


def slove_LU(L, U, b):
    n = b.shape[0]

    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum([L[i, k]*y[k] for k in range(i)])
    
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] =  (y[i] - sum([U[i, k]*x[k] for k in range(i+1, n)])) / U[i, i]
    
    return x


def get_det(L, U):
    return U.diagonal().prod() * L.diagonal().prod()


def inv_LU(L, U):
    n = L.shape[0]
    X = np.zeros((n, n))
    B = np.eye(n)

    for i in range(n):
        X[i] = slove_LU(L, U, B[i])
    
    return X.T


size = 5

A = np.random.randint(low=1, high=10, size=(size, size))
L, U = to_LU(A)
print("1: ", np.linalg.norm(A - np.matmul(L, U), ord=2))

A = np.random.randint(low=1, high=10, size=(size, size))
A += A.T
L, U = to_LU(A)
print('2: ', np.linalg.norm(L - L.T, ord=2))

A = np.random.randint(low=1, high=10, size=(size, size))
L, U = to_LU(A)
b = np.random.randint(low=1, high=10, size=size)
print('3: ', np.linalg.norm(np.linalg.solve(A, b) - slove_LU(L, U, b), ord=2))

A = np.random.randint(low=1, high=10, size=(size, size))
L, U = to_LU(A)
print('4: ', abs(get_det(L, U) - np.linalg.det(A)))

A = np.random.randint(low=1, high=10, size=(size, size))
B = np.random.randint(low=1, high=10, size=(size, size))
L, U = to_LU(A)
print('5: ',  np.linalg.norm(inv_LU(L, U) - np.linalg.inv(A), ord=2))

