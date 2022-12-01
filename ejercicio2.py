import numpy as np
import matplotlib.pyplot as plt


def random_discrete_diagonal_matrix(size=10, max_value=8):
    """
    n is the size of the matrix A \in R^nxn
    min is 10 times the minimum of all eigenvalues
    max is 10 times the maximum of all eigenvalues
    """
    values = np.random.randint(max_value - 1, size=size) + 1
    number_of_different_eigenvalues = len(set(values))
    random_matrix = np.diag(values)
    return random_matrix, number_of_different_eigenvalues


def linspace_diagonal_matrix(matrix_size=10, max=1):
    """
    n is the size of the matrix A \in R^nxn
    min is the minimum of all eigenvalues
    max is the maximum of all eigenvalues
    """
    MIN = 1
    matrix = np.diag([int(aval)
                     for aval in np.linspace(MIN, max, num=matrix_size)])
    return matrix


def significant_figures(number, scientific=True):
    if scientific:
        return "{0:.3e}".format(number)
    return "{0:.3g}".format(number)


def gradiente_conjugado(A, b, x0, k_max=10000, gamma_k=(1, 1), secuencia=False):
    d0 = -A @ x0 - b
    k = 0
    xk = x0
    dk = d0
    if secuencia:
        valores_de_xk = [np.linalg.norm(xk)]
    while np.linalg.norm(A @ xk + b) > 10 ** -8 and k < k_max:
        tk = -(A @ xk + b).T @ dk / (dk.T @ A @ dk)
        tk *= gamma_k[0]
        xk += tk * dk
        if secuencia:
            valores_de_xk.append(np.linalg.norm(xk))
        betak = dk.T @ A @ (A @ xk + b) / (dk.T @ A @ dk)
        betak *= gamma_k[1]
        dk = -(A @ xk + b) + betak * dk
        k += 1
    if secuencia:
        return xk, k - 1, valores_de_xk
    return xk, k - 1


def metodo_gradiente(A, b=None, x0=None, k_max=10000):
    b = np.zeros(np.shape(A)[0])
    x0 = np.ones(np.shape(A)[0])
    k = 0
    xk = x0
    d = -A @ x0 - b
    while k <= k_max and np.linalg.norm(d) > 10 ** -8:
        t = d.T @ d / (d.T @ A @ d)
        xk += t * d
        d = -A @ xk - b
        k += 1

    return np.linalg.norm(xk), k - 1


def pretty_print_table(table, ej2=False):
    columns = table[0]
    rows = table[1]
    data = table[2]
    if ej2:
        format_row = "{:>15}" * (len(rows) + 1)
    else:
        format_row = "{:>28}" * (len(rows) + 1)
    print(format_row.format("", *rows))
    for column, row in zip(columns, data):
        print(format_row.format(column, *row))


def run_tests_parte_1_1(cantidad_de_tests, matrix_size):
    columns = ["# de avalores", "# de iteraciones",
               "||xk-x*||", "||xk-x*|| < 10^-8 ? "]
    rows = [f"matriz {i+1}" for i in range(cantidad_de_tests)]
    data = []

    for _ in range(cantidad_de_tests):
        matriz, different_eigenvalues = random_discrete_diagonal_matrix(
            size=matrix_size
        )
        xk, iterations = gradiente_conjugado(
            matriz, np.zeros(matrix_size), np.ones(matrix_size)
        )
        result = [
            different_eigenvalues,
            iterations,
            significant_figures(np.linalg.norm(xk)),  # x* = 0
            "Si" if np.linalg.norm(xk) < 10 ** -8 else "No",
        ]
        data.append(result)
    return rows, columns, data


def run_tests_parte_1_2(matrix_size):
    matriz = linspace_diagonal_matrix(
        matrix_size=matrix_size, max=10
    )
    xk, iterations, valores_xk = gradiente_conjugado(
        matriz, np.zeros(matrix_size), np.ones(matrix_size), secuencia=True
    )
    return valores_xk


def add_graphics(fs):
    ks = range(1, len(fs)+1)
    print(fs)
    plt.bar(ks, np.array(fs))
    plt.xlabel("Numero de Iteracion")
    plt.ylabel("Valor de ||xk - x*||")
    plt.title("Convergencia Cuadratica")
    plt.show()


# 1
table = run_tests_parte_1_1(cantidad_de_tests=8, matrix_size=10)
pretty_print_table(table)
fs = run_tests_parte_1_2(matrix_size=10)
add_graphics(fs)

# 2
coefficients = [
    (1, 1),
    (0.9, 1),
    (0.99, 1),
    (1, 0.9),
    (1, 0.99),
    (0.9, 0.9),
    (0.99, 0.99),
]
"""
  comprobar que todos las otras tuplas dan mas iteraciones que (tk*, betak)
"""


def run_tests_parte_2(coefficients, size_of_matrix=10, cantidad_de_tests=8):
    columns = [f"({coeff[0]}t, {coeff[1]}b)" for coeff in coefficients]
    rows = [f"matriz {i+1}" for i in range(cantidad_de_tests)]
    data = []
    for _ in range(cantidad_de_tests):
        matrix, _ = random_discrete_diagonal_matrix(size=size_of_matrix)
        results = []
        for coefficient in coefficients:
            _, iterations = gradiente_conjugado(
                matrix,
                np.zeros(size_of_matrix),
                np.ones(size_of_matrix),
                gamma_k=coefficient,
            )
            results.append(iterations)
        data.append(results)
    return (rows, columns, data)


table = run_tests_parte_2(coefficients)
pretty_print_table(table, ej2=True)


# 3
def run_tests_parte_3(cantidad_de_tests, matrix_size):
    print("\t" * 7 + "# de iteraciones")
    columns = ["Metodo del gradiente",
               "Gradiente conjugado", "lambda_max / lambda_min"]
    rows = [f"matriz {i+1}" for i in range(cantidad_de_tests)]
    data = []

    for i in range(cantidad_de_tests):
        matriz = linspace_diagonal_matrix(
            matrix_size=matrix_size, max=10 ** (i + 1))
        _, iteraciones_gradiente_conjugado = gradiente_conjugado(
            matriz,
            np.zeros(matrix_size),
            np.ones(matrix_size),
        )
        _, iteraciones_metodo_gradiente = metodo_gradiente(matriz)
        result = [
            iteraciones_metodo_gradiente,
            iteraciones_gradiente_conjugado,
            significant_figures(10 ** (i + 1)),
        ]
        data.append(result)
    return rows, columns, data


table = run_tests_parte_3(cantidad_de_tests=4, matrix_size=12)
pretty_print_table(table)
