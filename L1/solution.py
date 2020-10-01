import numpy as np


def mul_v_v(a, b):
    """Скалярное умножение векторов"""
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res


def mul_m_m(A, B):
    """Умножение матриц"""
    res = []
    for row in A:
        v = []
        for i in range(len(B[0])):
            v.append(mul_v_v(row, [b_row[i] for b_row in B]))
        res.append(v)
    return res


def gauss_forward(A0, f0):
    """Прямой ход метода Гаусса"""
    A = np.copy(A0)
    f = np.copy(f0)

    for i in range(0, len(A) - 1):
        for j in range(i + 1, len(A)):
            if A[i, i] == 0:
                raise KeyError('can not to solve with zero')
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


def gauss(A0, f0):
    """Метод Гаусса"""
    A1, f1 = gauss_forward(A0, f0)
    return gauss_reverse(A1, f1)
