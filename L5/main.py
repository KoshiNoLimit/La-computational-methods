import numpy as np
from L5.solution import strassen

SIZE = 10

A = np.random.randint(low=1, high=10, size=(SIZE, SIZE))/10
for i in range(SIZE):
    A[i][i] += 2 * sum(A[i])

B = np.random.randint(low=1, high=10, size=(SIZE, SIZE))/10
for i in range(SIZE):
    B[i][i] += 2 * sum(B[i])

strassen()

