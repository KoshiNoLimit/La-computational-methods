import numpy as np


def gauss(A0, f0, boost=False):
    """Метод Гаусса"""

    def gauss_forward(A, f):
        """Прямой ход метода Гаусса"""
        for i in range(0, len(A) - 1):
            for j in range(i + 1, len(A)):
                if A[i, i] == 0:
                    raise KeyError('can not to solve with zero')
                coef = (A[j][i] / A[i][i])
                f[j] -= f[i] * coef
                A[j] -= A[i] * coef
        return A, f

    def gauss_forward_boost(A, f):
        """Прямой ход метода гаусса"""
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

    def gauss_reverse(A, f):
        """Обратный ход метода Гаусса"""
        for i in range(0, len(A)):
            f[i] /= A[i][i]
            A[i] /= A[i][i]

        for i in range(len(A) - 1, 0, -1):
            for j in range(i, len(A)):
                f[i - 1] -= (A[i - 1][j]) * f[j]
        return f

    if boost:
        A, f = gauss_forward_boost(np.copy(A0), np.copy(f0))
    else:
        A, f = gauss_forward(np.copy(A0), np.copy(f0))

    return gauss_reverse(A, f)


def jacobi(A0, b0, x0, eps):
    """Метод Якоби решения СЛАУ"""
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
            x[i] = np.inner(B[i], x_copy) + c[i]
        # x = np.matmul(B, x) + c

        if np.linalg.norm(x - x_copy, ord=1) < eps:
            return x
        else:
            x_copy = np.copy(x)


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

        if np.linalg.norm(x - x_copy, ord=1) < eps:
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


def one_param(A0, f0, x0, t, eps):
    A, f, x = np.copy(A0), np.copy(f0), np.copy(x0)

    x_temp = np.copy(x)
    E = np.eye(len(A))

    while True:
        x = np.matmul(E - A*t, x) + f*t
        if np.linalg.norm(x - x_temp, ord=1) < eps:
            return x
        x_temp = np.copy(x)


def mul_m_m(A, B):
    """Умножение матриц"""
    size = len(A)
    C = []
    for a in A:
        C.append([
            sum([a[i]*B[j][i] for i in range(size)]) 
            for j in range(size)
        ])
    
    return C


def sum_m_m(A, B):
    return [ 
        [A[i][j]+B[i][j] for j in range(len(A))] for i in range(len(A))
    ]


def diff_m_m(A, B):
    return [ 
        [A[i][j]-B[i][j] for j in range(len(A))] for i in range(len(A))
    ]


def split_matrix(M):
    mid = int(len(M)/2)
    M11, M12, M21, M22 = [[0]*mid]*mid, [[0]*mid]*mid, [[0]*mid]*mid, [[0]*mid]*mid
    for i in range(mid):
        for j in range(mid):
            M11[i][j] = M[i][j]
            M22[i][j] = M[i+mid][j+mid]
            M12[i][j] = M[i][j+mid]
            M21[i][j] = M[i+mid][j]   

    return M11, M12, M21, M22    


def shtrassen(A, B):
    mid = int(len(A)/2)
    #print(len(A), len(A[1]))
    if mid == 1:
        return mul_m_m(A, B)
    # P1 = shtrassen((A[:mid, :mid] + A[mid:, mid:]), (B[:mid, :mid] + B[mid:, mid:]))
    # P2 = shtrassen((A[mid:, :mid] + A[mid:, mid:]), B[:mid, :mid])
    # P3 = shtrassen(A[:mid, :mid], (B[:mid, mid:] - B[mid:, mid:]))
    # P4 = shtrassen(A[mid:, mid:], (B[mid:, :mid] - B[:mid, :mid]))
    # P5 = shtrassen((A[:mid, :mid] + A[:mid, mid:]), B[mid:, mid:])
    # P6 = shtrassen((A[mid:, :mid] - A[:mid, :mid]), (B[:mid, :mid] + B[:mid, mid:]))
    # P7 = shtrassen((A[:mid, mid:] - A[mid:, mid:]), (B[mid:, :mid] + B[mid:, mid:]))
    # print(A[:mid][:mid], A[mid:][mid:])

    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    P1 = shtrassen(sum_m_m(A11, A22), sum_m_m(B11, B22))
    P2 = shtrassen(sum_m_m(A21, A22), B11)
    P3 = shtrassen(A11, diff_m_m(B12, B22))
    P4 = shtrassen(A22, diff_m_m(B21, B11))
    P5 = shtrassen(sum_m_m(A11, A12), B22)
    P6 = shtrassen(diff_m_m(A21, A11), sum_m_m(B11, B12))
    P7 = shtrassen(diff_m_m(A12, A22), sum_m_m(B21, B22))


    # C = np.zeros((len(A), len(A)))
    # C[:mid, :mid] = P1 + P4 - P5 + P7
    # C[:mid, mid:] = P3 + P5
    # C[mid:, :mid] = P2 + P4
    # C[mid:, mid:] = P1 - P2 + P3 + P6

    C = [[0] * len(A)] * len(A)

    for i in range(mid):
        for j in range(mid):
            C[i][j] = P1[i][j] + P4[i][j] - P5[i][j] + P7[i][j]
            C[i][mid+j] = P3[i][j] + P5[i][j]
            C[mid+i][j] = P2[i][j] + P4[i][j]
            C[mid+i][mid+j] = P1[i][j] - P2[i][j] + P3[i][j] + P6[i][j]

    return C


def holeskiy(A):
    n = len(A)
    C = np.zeros((n, n))

    for i in range(n):
        s = A[i, i]
        for ip in range(i):
            s -= C[i, ip]*C[i, ip]
        
        C[i, i] = s**0.5
        for j in range(i+1, n):
            s = A[j, i]
            for ip in range(0, i):
                s -= C[i, ip] * C[j, ip]
            C[j, i] = s / C[i, i]
    return C
# import time

# SIZES = [2**i for i in range(1, 12)]
# y = {'shtrassen': [], 'simple': [], 'shtr_distrib': []}

# for size in SIZES:
#     A = np.random.randint(low=1, high=100, size=(size, size))
#     B = np.random.randint(low=1, high=100, size=(size, size))

#     # A = A.tolist()
#     # B = B.tolist()

#     start_time = time.time()
#     shtrassen(A, B)
#     y['shtrassen'].append(time.time() - start_time)

#     start_time = time.time()
#     mul_m_m(A, B)
#     y['simple'].append(time.time() - start_time)

#     start_time = time.time()
#     np.matmul(A, B)
#     y['shtr_distrib'].append(time.time() - start_time)


#     print(size, y['simple'][-1], y['shtrassen'][-1], y['shtr_distrib'][-1])
    

# A = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])   
# B = np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]])  
# print(np.matmul(A, B))

