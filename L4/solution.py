import numpy as np


def zeidel(A0, b0, x0, eps):
    """Метод Зейделя решения СЛАУ"""
    A, b, x = np.copy(A0), np.copy(b0), np.copy(x0)

    D = np.diagflat(np.diag(A))
    D_inv = np.linalg.inv(D)
    B = np.matmul(D_inv, (D - A))
    c = np.matmul(D_inv, b)

    if np.linalg.norm(B) >= 1:
        raise Exception('Для данной матрицы метод не сходится')

    x_copy = np.copy(x)
    while True:
        for i in range(len(x)):
            x[i] = np.inner(B[i], x) + c[i]
        if np.linalg.norm(x - x_copy) < eps:
            return x
        else:
            x_copy = np.copy(x)


def relaxation(A0, b0, x0, w, iterations):
    """Метод Релаксации решения СЛАУ"""
    A, b, x = np.copy(A0), np.copy(b0), np.copy(x0)

    D = np.diagflat(np.diag(A))
    D_inv = np.linalg.inv(D)
    B = np.matmul(D_inv, (D - A))
    c = np.matmul(D_inv, b)

    if np.linalg.norm(B) >= 1:
        raise Exception('Для данной матрицы метод не сходится')

    for _ in range(iterations):
        for i in range(len(x)):
            x[i] = np.inner(B[i], x) + c[i]
            k = x[i-1] if i else 0
            x[i] = w*x[i] + (1-w)*k

    return x


