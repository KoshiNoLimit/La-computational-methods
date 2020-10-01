import numpy as np


def gauss_forward(A0, f0):
    """Прямой ход метода гаусса"""
    A = np.copy(A0)
    f = np.copy(f0)

    for i in range(len(A) - 1):
        main_str = i
        for j in range(i + 1, len(A)):
            if A[j, i] > A[main_str, i]:
                main_str = j

        if A[main_str, i] == 0:
            raise KeyError('can not to solve with zero')

        A[[i, main_str]] = A[[main_str, i]]
        f[[i, main_str]] = f[[main_str, i]]

        for j in range(i + 1, len(A)):
            coef = (A[j][i] / A[i][i])
            f[j] -= f[i] * coef
            A[j] -= A[i] * coef

    return A, f


def gauss_reverse(A0, f0):
    """Обратный ход метода Гаусса"""
    A = np.copy(A0)
    f = np.copy(f0)

    for i in range(0, len(A)):
        f[i] /= A[i][i]
        A[i] /= A[i][i]

    for i in range(len(A) - 1, 0, -1):
        for j in range(i, len(A)):
            f[i - 1] -= (A[i - 1][j]) * f[j]

    return f


def gauss_boost(A0, f0):
    """Метод Гаусса"""
    A1, f1 = gauss_forward(A0, f0)
    return gauss_reverse(A1, f1)
