import numpy as np
from L3.solution import jacobi
from L4.solution import zeidel, relaxation

import time

SIZE = 200
COUNT = 5

A = np.random.randint(low=1, high=10, size=(SIZE, SIZE))/10
for i in range(SIZE):
    A[i][i] += 200 * sum(A[i])

b = np.random.randint(low=1, high=10, size=SIZE)/10
x_init = np.zeros(SIZE)


start_time = time.time()
x = jacobi(A, b, x_init, iterations=COUNT)
end_time = time.time()
b_res = np.matmul(A, x)
print(
    'Оценка погрешности Якоби при {} итерациях: '.format(COUNT),
    np.linalg.norm(b-b_res, ord=np.inf),
    'Время: ',  end_time - start_time
)

start_time = time.time()
x = zeidel(A, b, x_init, iterations=COUNT)
end_time = time.time()
b_res = np.matmul(A, x)
print(
    'Оценка погрешности Зейделя при {} итерациях: '.format(COUNT),
    np.linalg.norm(b-b_res, ord=np.inf), 'Время: ',
    end_time - start_time
)

start_time = time.time()
x = relaxation(A, b, x_init, 1, iterations=COUNT)
end_time = time.time()
b_res = np.matmul(A, x)
print(
    'Оценка погрешности Релаксации при {} итерациях: '.format(COUNT),
    np.linalg.norm(b-b_res, ord=np.inf), 'Время: ',
    end_time - start_time
)

