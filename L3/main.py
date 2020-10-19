import numpy as np
from L3.solution import jacobi

SIZE = 10

A = np.random.randint(low=1, high=10, size=(SIZE, SIZE))
for i in range(SIZE):
    A[i][i] += sum(A[i])


b = np.random.randint(low=1, high=10, size=SIZE)
x_init = np.zeros(SIZE)

for i in range(10, 101, 10):
    x = jacobi(A, b, x_init, iterations=i)
    b_res = np.matmul(A, x)
    print('Оценка погрешности при {} итерациях: '.format(i), np.linalg.norm(b-b_res, ord=np.inf))
