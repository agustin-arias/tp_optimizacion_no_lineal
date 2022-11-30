import numpy as np
import matplotlib.pyplot as plt


def metodo_gradiente(
    A,
    b=None,
    x0=None,
    max_iter=100,
    gamma_k=1,
):
    b = np.zeros(np.shape(A)[0])
    x0 = np.ones(np.shape(A)[0])
    k = 0
    xk = x0
    d = -A @ x0 - b
    while k <= max_iter and np.linalg.norm(d) > 10 ** -8:
        t = d.T @ d / (d.T @ A @ d)
        t *= gamma_k if gamma_k != -1 else np.random.rand()
        xk += t * d
        d = -A @ xk - b
        k += 1
    return xk, k - 1


def derivada_parcial(f, x, i):
    h = 0.1
    e_i = np.zeros(len(x))  # COMPLETAR: i-esimo vector canonico
    e_i[i - 1] = 1
    z = (f(x + h * e_i) - f(x - h * e_i)) / (2 * h)  # COMPLETAR: formula del metodo
    h = h / 2
    y = (f(x + h * e_i) - f(x - h * e_i)) / (2 * h)  # COMPLETAR: formula del metodo
    error = np.linalg.norm(y - z)
    eps = 1e-8
    while error > eps and (y != np.nan) and (y != np.inf):
        error = np.linalg.norm(y - z)
        z = y
        h = h / 2
        y = (f(x + h * e_i) - f(x - h * e_i)) / (2 * h)  # COMPLETAR: formula del metodo
    return z


def gradiente(f, x):
    return np.array([derivada_parcial(f, x, i + 1) for i in range(len(x))])


def longitud_armijo(f, x, d, eta=0.2, beta=2):
    t = 1
    grad = gradiente(f, x)
    if f(x + t * d) <= f(x) + eta * t * grad.T @ d:
        while f(x + t * d) <= f(x) + eta * t * grad.T @ d:
            t = beta * t
        return t / beta
    else:
        while f(x + t * d) > f(x) + eta * t * grad.T @ d:
            t = t / beta
        return t


def metodo_cn_bbr(A, b, x0, eps=10 ** -8, k_max=100, secuencia=False):
    f = lambda x: 0.5 * x @ (A @ x) + b @ x
    d0 = A @ x0 + b
    t0 = longitud_armijo(f, x0, d0)
    x1 = x0 + t0 * d0
    xk = x1
    k = 1
    sk_minus_1 = t0 * d0
    if secuencia:
        valores_de_fx = [f(x0)]
    while np.linalg.norm(A @ xk + b) > eps and k < k_max:
        if secuencia:
            valores_de_fx.append(f(xk))
        dk = -(A @ xk + b)
        tk = sk_minus_1 @ sk_minus_1 / (sk_minus_1 @ A @ sk_minus_1)
        sk_minus_1 = tk * dk
        xk += tk * dk
        k += 1
    if secuencia:
        return xk, k, valores_de_fx
    return xk, k


def linspace_diagonal_matrix(matrix_size=10, max=1):
    """
    n is the size of the matrix A \in R^nxn
    min is the minimum of all eigenvalues
    max is the maximum of all eigenvalues
    """
    MIN = 1
    matrix = np.diag([int(aval) for aval in np.linspace(MIN, max, num=matrix_size)])
    return matrix


def significant_figures(number):
    return "{0:.3e}".format(number)


def run_tests(coefficients, min=2, max=6, print_along=False, size=10, k_max=1000):
    columns = [f"coeff={coefficient}" for coefficient in coefficients]
    columns.append("BBR")
    if -1 in coefficients:
        columns[coefficients.index(-1)] = "coeff random"
    rows = [f"max_aval entre 1 y 10^{i}" for i in range(min, max)]
    data = []

    for i in range(min, max):
        results = []
        A = linspace_diagonal_matrix(matrix_size=size, max=10 ** i)
        for coefficient in coefficients:
            _, iteraciones = metodo_gradiente(A, gamma_k=coefficient, max_iter=k_max)
            results.append(iteraciones)
        _, iteraciones = metodo_cn_bbr(A, np.zeros(size), np.ones(size), k_max=k_max)
        results.append(iteraciones)
        data.append(results)
    return rows, columns, data


def pretty_print_table(table):
    columns = table[0]
    rows = table[1]
    data = table[2]
    format_row = "{:>25}" * (len(rows) + 1)
    print(format_row.format("", *rows))
    for column, row in zip(columns, data):
        print(format_row.format(column, *row))


K_MAX = 10000


# 1
coefficients_1 = [3 / 4, 1 / 2, 1 / 4, -1]
table = run_tests(coefficients_1, k_max=K_MAX)
pretty_print_table(table)
"""
    Notamos que el mejor en todos los casos es el de BBR
"""


# 2
matrix1 = np.diag([0.01, 0.02, 0.03, 0.04, 0.05, 10, 11, 12, 13, 14])

size = 10
_, ktop, fs = metodo_cn_bbr(
    matrix1, np.zeros(size), 50 * np.ones(size), k_max=20, secuencia=True
)
f = lambda x: 0.5 * x @ matrix1 @ x
ks = np.arange(ktop)
plt.plot(ks, np.array(fs))
plt.show()
"""
    En la 11va iteracion f(xk) pega un salto
"""


TEST_MATRICES = [
    np.eye(10),
    np.diag([1, 1.1, 1, 1.01, 0.99, 0.8, 1.01, 1, 0.9, 0.95]),
    np.diag([1, 1.5, 0.8, 0.7, 1.12, 0.95, 1, 1.1, 1.6, 0.75]),
    np.diag([2, 0.4, 0.8, 0.5, 1, 1.8, 1, 0.55, 1.99, 0.90]),
    np.diag([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    np.diag([0.1, 0.25, 0.5, 0.75, 1, 2, 4, 8, 1, 1]),
    np.diag([0.01, 0.02, 0.03, 0.04, 0.05, 10, 11, 12, 13, 14]),
    np.diag([0.001, 512, 50, 12, 0.45, 0.3, 56, 12, 1.5, 1.1]),
]
