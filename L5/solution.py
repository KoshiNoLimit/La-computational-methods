import numpy as np


def strassen(A0, B0):
    """Метод Штрассена умножения матриц"""
    A, B = np.copy(A0), np.copy(B0)
    size = len(A)
    two_degree = 2
    while two_degree < size:
        two_degree *= 2

    diff = size - two_degree
    if diff:
        for i, in range(size):
            np.concatenate(A[i], np.zeros(diff))
            np.concatenate(B[i], np.zeros(diff))

        for i in range(diff):
            np.append(A, np.zeros(two_degree))
            np.append(B, np.zeros(two_degree))

    print(A, B)

    return False
