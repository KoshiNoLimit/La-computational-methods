import numpy as np

delta = 2
A = [[12, 24], [52, 83]]
f = [23, 32]

A, f = np.copy(A), np.copy(f)
A_p, f_p = np.copy(A) + delta, np.copy(f) + delta

x = np.linalg.solve(A, f)
x_p = np.linalg.solve(A_p, f_p)

print('Величина погрешности по норме L1: ', np.linalg.norm(x-x_p, ord=1))

norm_x = np.linalg.norm(x)
norm_x_p = np.linalg.norm(x_p)
norm_f = np.linalg.norm(f)
norm_f_p = np.linalg.norm(f_p)
norm_A = np.linalg.norm(A)
norm_A_p = np.linalg.norm(A_p)

cond_A = np.linalg.cond(A)

print('Расхождение по формуле: ', (norm_x_p/norm_x) - ((norm_f_p/norm_f)+(norm_A_p/norm_A)) * cond_A)
