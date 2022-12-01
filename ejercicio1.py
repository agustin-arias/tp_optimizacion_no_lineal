import numpy as np
import math
import warnings

warnings.filterwarnings("ignore")

# Ejercicio 1
"""
Sabemos que si A es diagonal, entonces f(x) = lambda_1/2 * x1^2 + ... lambda_n/2 * xn^2
Donde lambda_1, ... lambda_n son los autovalores de A. 
Sabemos que el minimo se alcanzara cuando x1=x2=...=xn=0, siendo este minimo x=(0,0,...,0)
Para tener una metrica de comparacion entre distintos metodos, tomaremos la norma de xk (la ultima
iteracion). Tendremos una mejor aproximacion si ||xk|| esta mas cercano a 0.
Tomaremos al minimo autovalor como 1 siempre, y al maximo autovalor variable.
"""


def metodo_gradiente(
    A, b=None, x0=None, max_iter=10000, gamma_k=1, angle_of_rotation=None
):
    b = np.zeros(np.shape(A)[0])
    x0 = np.ones(np.shape(A)[0])
    k = 0
    xk = x0
    d = -A @ x0 - b
    if angle_of_rotation != None:
        rotation_matrix = np.array(
            [
                [math.cos(angle_of_rotation), -math.sin(angle_of_rotation)],
                [math.sin(angle_of_rotation), math.cos(angle_of_rotation)],
            ]
        )

    while k <= max_iter and np.linalg.norm(d) > 10 ** -8:
        t = d.T @ d / (d.T @ A @ d)
        t *= gamma_k if gamma_k != -1 else np.random.rand()
        xk += t * d
        d = -A @ xk - b
        if angle_of_rotation != None:
            gradient = -d
            d = rotation_matrix @ d
            if gradient @ d >= 0:
                d = -gradient
        k += 1

    return xk, k - 1


def linspace_diagonal_matrix(n=10, max=1):
    """
    n is the size of the matrix A \in R^nxn
    min is the minimum of all eigenvalues
    max is the maximum of all eigenvalues
    """
    MIN = 1
    matrix = np.diag([int(aval) for aval in np.linspace(MIN, max, num=n)])
    return matrix


def significant_figures(number):
    return "{0:.3e}".format(number)


def run_tests(coefficients, angle=None, min=1, max=5, print_along=False, size=10):
    columns = [f"coeff={coefficient}" for coefficient in coefficients]
    if -1 in coefficients:
        columns[coefficients.index(-1)] = "coeff random"
    rows = [f"aval entre 1 y 10^{i}" for i in range(min, max)]
    data = []
    if angle != None:
        columns = np.concatenate((["coeff=1 (sin rotar)"], columns))

    for i in range(min, max):
        if print_along:
            print(f"Para A con valores entre 0 y 10^-{i}", end="\n\t")
        results = []
        A = linspace_diagonal_matrix(n=size, max=10 ** i)
        if angle != None:
            solucion, iteraciones = metodo_gradiente(A, gamma_k=1)
            results.append(iteraciones)

        for coefficient in coefficients:
            solucion, iteraciones = metodo_gradiente(
                A, gamma_k=coefficient, angle_of_rotation=angle
            )
            coefficient_string = (
                "= " + str(coefficient) if coefficient != -1 else "aleatorio"
            )
            # results.append(significant_figures(np.linalg.norm(solucion)))
            results.append(iteraciones)
            if print_along:
                print(
                    f"Para el coeficiente {coefficient_string}: {resultado_3_cifras}",
                    end="\n\t",
                )
        data.append(results)
        if print_along:
            print()
    if print_along:
        print("\n" * 3)
    return rows, columns, data


def pretty_print_table(table):
    columns = table[0]
    rows = table[1]
    data = table[2]
    format_row = "{:>20}" * (len(rows) + 1)
    print(format_row.format("", *rows))
    for column, row in zip(columns, data):
        print(format_row.format(column, *row))


# 1
coefficients_1 = [1, 3 / 4, 1 / 2, 1 / 4, -1]
table = run_tests(coefficients_1)
pretty_print_table(table)


print("\n" * 2)
# 2
coefficients_2 = [1, 0.99, 0.9]
table = run_tests(coefficients_2)
pretty_print_table(table)
print("\n" * 2)

# 3
PI = math.pi
angles_of_rotation = [
    -PI / 3,
    -PI / 6,
    PI / 6,
    PI / 3,
]

for angle_of_rotation in angles_of_rotation:
    print("\t" * 8 + f"Theta = {round(math.degrees(angle_of_rotation))}°")
    table = run_tests(coefficients=coefficients_1,
                      angle=angle_of_rotation, size=2)
    pretty_print_table(table=table)
    print("\n" * 2)
