import numpy as np
from tests import TestLab2, TestLab1
import L2
import L1


test1 = TestLab1()
test2 = TestLab2()
for i in range(100, 501, 100):
    A = np.random.randint(low=1, high=10, size=(i, i)) / 10000
    f = np.random.randint(low=1, high=10, size=i) / 10000
    flag = True
    for row in A:
        m = max(abs(i) for i in row)
        if m < sum(abs(i) for i in row)/2:
            flag = False
            break

    test1.test_gauss(A, f)
    test2.test_gauss(A, f)

    res1 = L1.solution.gauss(A, f)
    res2 = L2.solution.gauss_boost(A, f)

    check1 = np.matmul(A, res1)
    check2 = np.matmul(A, res2)

    length1 = np.linalg.norm(check1 - f)
    length2 = np.linalg.norm(check2 - f)

    print('Диагональное преобладание: ', flag)
    print('L2 оценка погрешности стандартного Гаусса:     ', length1)
    print('L2 оценка алгоритма модифицированного Гаусса:  ', length2)
    print('Разность 1-го и 2-го: ', length1 - length2, '\n')
