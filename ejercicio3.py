import numpy as np
from time import time


def pretty_print_table(table):
    rows = table[0]
    columns = table[1]
    data = table[2]
    format_row = "{:>28}" * (len(rows) + 1)
    print(format_row.format("", *rows))
    for column, row in zip(columns, data):
        print(format_row.format(column, *row))


def significant_figures(number, scientific=True):
    if scientific:
        return "{0:.3e}".format(number)
    return "{0:.3g}".format(number)


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


# EJERCICIO 3 - CHOLESKY
# Sea A simetrica definida positiva, efectúa la descomposición de Cholesky: devuelve L tal que A=L.Lt
def chol(A):
    n = A.shape[0] - 1
    L = np.zeros(A.shape)
    L[0, 0] = np.sqrt(A[0, 0])

    for j in range(1, n + 1):
        L[j, 0] = A[j, 0] / L[0, 0]

    for i in range(1, n + 1):
        L[i, i] = np.sqrt(A[i, i] - sum(L[i, p] ** 2 for p in range(i)))
        if i != n:
            for j in range(i + 1, n + 1):
                L[j, i] = (A[j, i] - sum(L[i, p] * L[j, p] for p in range(i))) / L[i, i]
    return L


# Sea L diagonal inferior, devuelve el resultado de Lx=b
def sust_forward(L, b):
    n = L.shape[0] - 1
    sol = np.zeros(L.shape[0])
    sol[0] = b[0] / L[0, 0]
    for i in range(1, n + 1):
        sol[i] = (b[i] - sum(L[i, j] * sol[j] for j in range(i))) / L[i, i]
    return sol


# Sea L diagonal superior, devuelve el resultado de Lx=b
def sust_backward(L, b):
    n = L.shape[0] - 1
    sol = np.zeros(L.shape[0])
    sol[n] = b[n] / L[n, n]
    for i in reversed(range(n)):
        sol[i] = (b[i] - sum(L[i, j] * sol[j] for j in range(i + 1, n + 1))) / L[i, i]
    return sol


# Sea A simetrica definida positiva, devuelve el resultado de Ax=b
def eq_lineal_chol(A, b):
    L = chol(A)
    y = sust_forward(L, b)
    x = sust_backward(L.T, y)
    return x


def metodo_newton(A, b, x0, eps=10 ** -8, k_max=100):
    k = 0
    xk = x0
    while np.linalg.norm(A @ xk + b) > eps and k < k_max:
        # print(f"np.linalg.norm(A @ xk + b): {np.linalg.norm(A @ xk + b)}")
        # print(f"xk: {xk}")
        dk = eq_lineal_chol(A, -(A @ xk + b))
        # print(f"dk: {dk}")
        xk += dk
        k += 1
    return xk, k


def calculo_de_H(metodo, Hk, yk, sk):
    output = Hk
    if metodo == "Broyden":
        # print("Metodo Broyden")
        output += ((sk - Hk @ yk) @ (sk - Hk @ yk)) / (yk @ (sk - Hk @ yk))
    elif metodo == "DFP":
        # print("Metodo DFP")
        output += np.outer(sk, sk) / (yk @ sk)
        output -= Hk @ np.outer(yk, yk) @ Hk / (yk @ Hk @ yk)
    elif metodo == "BFGS":
        # print("Metodo BFGS")
        output += (1 + (yk @ Hk @ yk) / (sk @ yk)) * (np.outer(sk, sk)) / (sk @ yk)
        output -= (np.outer(sk, yk) @ Hk + Hk @ np.outer(yk, sk)) / (sk @ yk)
    return output


def metodo_cn_bbr(A, b, x0, eps=10 ** -8, k_max=100):
    f = lambda x: 0.5 * x @ (A @ x) + b @ x
    d0 = A @ x0 + b
    t0 = longitud_armijo(f, x0, d0)
    x1 = x0 + t0 * d0
    xk = x1
    k = 1
    sk_minus_1 = t0 * d0
    while np.linalg.norm(A @ xk + b) > eps and k < k_max:
        dk = -(A @ xk + b)
        tk = sk_minus_1 @ sk_minus_1 / (sk_minus_1 @ A @ sk_minus_1)
        sk_minus_1 = tk * dk
        xk += tk * dk
        k += 1
    return xk, k


# Broyden Mala, DFP, BFGS, BBR
def metodo_cuasi_newton(A, b, x0, H0, eps=10 ** -8, k_max=100, metodo="Broyden"):
    if metodo == "BBR":
        return metodo_cn_bbr(A, b, x0, k_max=k_max)
    k = 0
    xk = x0
    f = lambda x: A @ x + b
    Hk = H0
    while np.linalg.norm(A @ xk + b) > eps and k < k_max:
        dk = -Hk @ (A @ xk + b)
        if np.linalg.norm(dk) > 10 ** 150:
            break  # si la norma de dk crece demasiado, evitamos que el codigo nos tire un error de overflow
        tk = -(A @ xk + b) @ dk / (dk @ (A @ dk))
        yk = A @ (tk * dk)
        sk = tk * dk
        xk += tk * dk
        Hk = calculo_de_H(metodo, Hk, yk, sk)
        k += 1
    return xk, k


K_MAX = 5000

A = np.load("./matriz.npy")
size = np.shape(A)[0]
b = np.zeros(size)
x0 = np.ones(size)
identity = np.eye(size)

rows = ["Newton", "Broyden", "DFP", "BFGS", "BBR"]
columns = ["||xk||", "# iteraciones", "tiempo de ejecucion (s)"]

data = []

s = time()
x, iteraciones = metodo_newton(A, b, x0, k_max=K_MAX)


data.append(
    [
        significant_figures(np.linalg.norm(x)),
        iteraciones,
        significant_figures(time() - s, scientific=False),
    ]
)

for metodo in rows[1:-1]:
    A = np.load("./matriz.npy")
    size = np.shape(A)[0]
    b = np.zeros(size)
    x0 = np.ones(size)
    identity = np.eye(size)
    s = time()
    x, iteraciones = metodo_cuasi_newton(A, b, x0, identity, metodo=metodo, k_max=K_MAX)
    data.append(
        [
            significant_figures(np.linalg.norm(x)),
            iteraciones,
            significant_figures(time() - s, scientific=False),
        ]
    )

A = np.load("./matriz.npy")
size = np.shape(A)[0]
b = np.zeros(size)
x0 = np.ones(size)
identity = np.eye(size)
s = time()
x, iteraciones = metodo_cuasi_newton(A, b, x0, identity, metodo="BBR", k_max=K_MAX)
data.append(
    [
        significant_figures(np.linalg.norm(x)),
        iteraciones,
        significant_figures(time() - s, scientific=False),
    ]
)

pretty_print_table((columns, rows, data))

"""
    diferencias entre tiempo de ejecucion se debe a que en el metodo de Newton
    se usa la funcion eq_lineal_chol que es mas exacta que np.linalg.solve?
"""
