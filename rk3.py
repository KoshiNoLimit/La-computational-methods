import numpy as np
from matplotlib import pyplot
from one_parameter.solution import one_param
from L1.solution import gauss
from L3.solution import jacobi
from L4.solution import zeidel
SIZE = 100
ITERATIONS = 10

import time


def holeckiy(A0, f0):
    A, f = np.copy(A0), np.copy(f0)
    l = np.zeros((len(A), len(A)))
    l[0, 0] = A[0, 0] ** 0.5

    for i in range(1, len(A)):
        l[i, 0] = A[i, 0]/l[0, 0]

    for i in range(1, len(A)):
        l[i, i] = (A[i, i] - sum([l[i,p] ** 2 for p in range(i)]))**0.5

        for j in range(i+1, len(A)):
            l[j,i] = (A[j,i] - sum([l[i,p] * l[j,p] for p in range(i)]))/l[i,i]

    y = gauss(l, f)
    return gauss(l.T, y)


A = np.random.randint(low=1, high=10, size=(SIZE, SIZE))/100
A += A.T
for i in range(SIZE):
    A[i][i] += 2 * sum(A[i])
f = np.random.randint(low=1, high=10, size=SIZE)
x0 = np.zeros(SIZE)

vals = np.linalg.eigh(A)[0]
min_v, max_v = min(vals), max(vals)
t = 2/(min_v + max_v)

xes = np.arange(8)

t_min, t_opt, t_max = 0.2/max_v, 2/(min_v + max_v), 1.8/max_v

print(np.linalg.norm(np.matmul(A, holeckiy(A, f))-f))

j_y, z_y = [], []

for i in (0.1**j for j in range(2, 10)):
    print(i)
    st_time = time.time()
    jacobi(A, f, x0, i)
    en_time = time.time()
    j_y.append(en_time - st_time)

    st_time = time.time()
    zeidel(A, f, x0, i)
    en_time = time.time()
    z_y.append(en_time - st_time)


pyplot.plot(xes, j_y, label='jacobi')
pyplot.plot(xes, z_y, label='zeidel')
#pyplot.plot(xes, one_param(A, f, x0, t_max, ITERATIONS), label='t_max')
pyplot.grid()
pyplot.legend()
pyplot.show()
