import numpy as np
from L1.solution import gauss
from L2.solution import gauss_boost
from matplotlib import pyplot


def calc_matrix(x):
    m = np.zeros((len(x), len(x)), dtype=float)
    for i in range(1, len(x)):
        for j in range(1, i+1):
            m[i, j] = np.prod([x[i]-x[k] for k in range(j)])

    for i in range(len(m)):
        m[i, 0] = 1.0

    return m


x = np.array([0.0, -1.0, 1.0])
f = np.array([0.0, 2.0, 2.0])
C = calc_matrix(x)

res_gauss = gauss(C, f)
res_gauss_boost = gauss_boost(C, f)
res_np = np.linalg.solve(C, f)


def get_y(x, A):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = np.sum([A[j]*(x[i]**j) for j in range(len(A))])

    return y


print(res_gauss)
print(res_gauss_boost)
print(res_np)

xes = np.linspace(-10.0, 10.0, num=20)

pyplot.plot(xes, get_y(xes, res_gauss), label='gauss', lw=2)
pyplot.plot(xes, get_y(xes, res_gauss_boost), label='gauss_boost', lw=2)
pyplot.plot(xes, get_y(xes, res_np), label='numpy', lw=2)
pyplot.grid()
pyplot.legend()
pyplot.show()


