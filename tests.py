import L1.solution
import L2.solution
import numpy as np
import unittest


class TestLab1(unittest.TestCase):

    def test_vector_mul(self, matrix):
        """Тест скалярного умножения"""
        for row in range(1, len(matrix)):
            with self.subTest(i=row):
                self.assertEqual(
                    np.sum(np.array(matrix[row])*np.array(matrix[row-1])),
                    L1.solution.mul_v_v(matrix[row], matrix[row-1])
                )

    def test_matrix_mul(self, A, B):
        """Тест матричного умножения"""
        nA, nB = np.array(A), np.array(B)
        self.assertTrue(np.array_equal(
            np.matmul(nA, nB),
            np.array(L1.solution.mul_m_m(A, B))
        ))

    def test_gauss(self, A, f):
        """Тест метода Гаусса"""
        self.assertTrue(np.sum(np.absolute(np.linalg.solve(A, f) - L1.solution.gauss(A, f))) < 0.0001)


class TestLab2(unittest.TestCase):

    def test_gauss(self, A, f):
        """Тест метода Гаусса"""
        self.assertTrue(np.sum(np.absolute(np.linalg.solve(A, f) - L2.solution.gauss_boost(A, f))) < 0.0001)
