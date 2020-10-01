import numpy as np

from tests import TestLab1


def random_matrix(size1, size2):
    return np.random.randint(low=1, high=10, size=(size1, size2))


test = TestLab1()


# Скалярное умножение векторов
test.test_vector_mul(
    random_matrix(4, 10)
)

# Умножение матриц
set_of_matrix = [
    (random_matrix(3, 3), random_matrix(3, 1)),
    (random_matrix(3, 3), random_matrix(3, 3)),
    (random_matrix(2, 3), random_matrix(3, 3)),
    (random_matrix(2, 3), random_matrix(3, 2)),
]
for A, B in set_of_matrix:
    test.test_matrix_mul(A, B)

# Метод Гаусса
A = np.array([
    [5.0, 3.0, 2.0],
    [2.0, 2.0, 2.0],
    [1.0, 1.0, 6.0],
])
f = np.array([1.0, 1.0, 1.0])
test.test_gauss(A, f)

A = np.array([
    [3.0, -2.0],
    [5.0, 1.0],
])
f = np.array([-6.0, 3.0])
test.test_gauss(A, f)

A = np.array([
    [1.0, 1.0, -7.0, -1.0],
    [4.0, 1.0, 2.0, -1.0],
    [3.0, 0.0, -4.0, -1.0],
    [1.0, 1.0, 0.0, 3.0],
])
f = np.array([6.0, 0.0, 6.0, 3.0])
test.test_gauss(A, f)
