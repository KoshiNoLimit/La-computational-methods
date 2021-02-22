import numpy as np


def jacobi(A0, b0, x0, eps):
    """Метод Якоби решения СЛАУ"""
    A, b, x = np.copy(A0), np.copy(b0), np.copy(x0)

    D = np.diagflat(np.diag(A))
    D_inv = np.linalg.inv(D)
    B = np.matmul(D_inv, (D - A))
    c = np.matmul(D_inv, b)

    if np.linalg.norm(B) >= 1:
        raise Exception('Для данной матрицы метод не сходится')

    x_copy = np.copy(x)
    while True:
        x = np.matmul(B, x) + c
        if np.linalg.norm(x-x_copy) < eps:
            return x
        else:
            x_copy = np.copy(x)
