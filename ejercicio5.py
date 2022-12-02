import numpy as np
import warnings
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import iplot

warnings.filterwarnings("ignore")


# EJERCICIO 5 - FUNCIÓN BIRD Y SUS MINIMIZADORES


def plot_fun(f, limites, points=None):
    """
    f : función a graficar
    limites : toma una tupla (x1,x2,y1,y2) de los límites del gráfico: grafica en el dominio [x1,x2] x [y1,y2]
    points : lista de puntos a graficar sobre la superficie; se ingresa como una lista de tuplas (x,y,z)
    """

    x = np.linspace(limites[0], limites[1], 1000)
    y = np.linspace(limites[2], limites[3], 1000)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))
    data = [go.Surface(x=x, y=y, z=Z)]
    if points is not None:
        for p in points:
            data.append(go.Scatter3d(
                x=[p[0]], y=[p[1]], z=[p[2]], mode="markers"))
    fig = go.Figure(data=data)
    iplot(fig)


def bird(x):
    t1 = (x[0] - x[1]) ** 2
    t2 = np.sin(x[0]) * np.exp((1 - np.cos(x[1])) ** 2)
    t3 = np.cos(x[1]) * np.exp((1 - np.sin(x[0])) ** 2)
    return t1 + t2 + t3


def derivada_parcial(f, x, i):
    h = 0.1
    e_i = np.zeros(len(x))  # COMPLETAR: i-esimo vector canonico
    e_i[i - 1] = 1
    z = (f(x + h * e_i) - f(x - h * e_i)) / \
        (2 * h)  # COMPLETAR: formula del metodo
    h = h / 2
    y = (f(x + h * e_i) - f(x - h * e_i)) / \
        (2 * h)  # COMPLETAR: formula del metodo
    error = np.linalg.norm(y - z)
    eps = 1e-8
    while error > eps and (y != np.nan) and (y != np.inf):
        error = np.linalg.norm(y - z)
        z = y
        h = h / 2
        y = (f(x + h * e_i) - f(x - h * e_i)) / \
            (2 * h)  # COMPLETAR: formula del metodo
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


def metodo_cn_bbr(f, x0, eps=10 ** -8, k_max=100):
    d0 = -gradiente(f, x0)
    t0 = longitud_armijo(f, x0, d0)
    tk = t0
    dk = d0
    x1 = x0 + t0 * d0
    xk = x1
    k = 1
    sk_minus_1 = tk * dk
    while np.linalg.norm(gradiente(f, xk)) > eps and k < k_max:
        dk = -gradiente(f, xk)
        yk_minus_1 = gradiente(f, xk) - gradiente(f, xk - tk * dk)
        tk = sk_minus_1 @ sk_minus_1 / (sk_minus_1 @ yk_minus_1)
        sk_minus_1 = tk * dk
        xk += tk * dk
        k += 1
    return xk, k


minimizadores = np.array(
    [[4.701055751981055, 3.152946019601391],
        [-1.582142172055011, -3.130246799635430]]
)


def calculo_de_H(metodo, Hk, yk, sk):
    output = Hk
    if metodo == "Broyden":
        # print("Metodo Broyden")
        output += np.outer(sk - Hk @ yk, sk - Hk @ yk) / (yk @ (sk - Hk @ yk))
    elif metodo == "DFP":
        # print("Metodo DFP")
        output += np.outer(sk, sk) / (yk @ sk)
        output -= Hk @ np.outer(yk, yk) @ Hk / (yk @ Hk @ yk)
    elif metodo == "BFGS":
        # print("Metodo BFGS")
        output += (1 + (yk @ Hk @ yk) / (sk @ yk)) * \
            (np.outer(sk, sk)) / (sk @ yk)
        output -= (np.outer(sk, yk) @ Hk + Hk @ np.outer(yk, sk)) / (sk @ yk)
    return output


# Broyden Mala, DFP, BFGS, BBR
def metodo_cuasi_newton(f, x0, H0, eps=10 ** -4, k_max=100, metodo="Broyden"):
    k = 0
    xk = x0
    Hk = H0  # Initialize H_0 as the Identity Matrix
    gk = gradiente(f, xk)
    dk = -Hk @ gk
    while k < k_max and np.linalg.norm(dk) > eps:
        if np.linalg.norm(dk) > 2 ** 32 or np.linalg.norm(xk) > 2 ** 32:
            break  # si dk es demasiado grande, salir
        tk = longitud_armijo(f, xk, dk)
        xk += tk * dk
        sk = tk * dk
        old_gk = gk
        gk = gradiente(f, xk)
        yk = gk - old_gk
        Hk = calculo_de_H(metodo, Hk, yk, sk)
        dk = -Hk @ gk
        k += 1
    return xk, k


# 1

PLOT_BIRD = False
if PLOT_BIRD:
    plot_fun(bird, limites=(-10, 10, -10, 10))

# 2


def random_vector(dim=2, max_value=1):
    return np.random.random(2) * 2 * max_value - max_value


def random_restart(f, N, c, k_max=100):
    mejor_valor_f_en_solucion = np.inf
    mejor_sol_x = np.zeros(2)
    for _ in range(N):
        random_initial_point = random_vector(max_value=c)
        solucion, _ = metodo_cuasi_newton(
            f, random_initial_point, np.eye(2), k_max=k_max)
        if f(solucion) < mejor_valor_f_en_solucion:
            mejor_valor_f_en_solucion = f(solucion)
            mejor_sol_x = solucion
    return mejor_sol_x


def obtener_promedio(soluciones, repeticiones):
    return sum([x for x in soluciones if x < 2**32]) / repeticiones


def obtener_frecuencia(soluciones, repeticiones):
    return sum([1 for x in soluciones if abs(x - (-106.764536)) < 10 ** -2]) / repeticiones


def run_tests_parte_2():
    MAX_N = 20
    REPETICIONES = 20
    c = 15
    frecuencias = []
    promedios = []
    for N in range(1, MAX_N + 1):
        soluciones = []
        for _ in range(REPETICIONES):
            solucion = random_restart(bird, N, c)
            soluciones.append(bird(solucion))
            #print(solucion, bird(solucion))
        promedios.append(obtener_promedio(soluciones, REPETICIONES))
        frecuencias.append(obtener_frecuencia(soluciones, REPETICIONES))
    return promedios, frecuencias


def add_bar_plot(frecuencias):
    ks = range(1, len(frecuencias)+1)
    plt.bar(ks, np.array(frecuencias))
    plt.xlabel("N (cantidad de puntos iniciales)")
    plt.ylabel("Frecuencia del optimo global")
    plt.title("Frecuencia del optimo global")
    plt.show()


def add_line_plot(promedios):
    ks = range(1, len(promedios)+1)
    plt.plot(ks, promedios)
    plt.xlabel("N (cantidad de puntos iniciales)")
    plt.ylabel("Media aritmetica de los optimos")
    plt.title("Media aritmetica de los optimos")
    plt.show()


promedios, frecuencias = run_tests_parte_2()
print(promedios, frecuencias)
add_bar_plot(frecuencias)
add_line_plot(promedios)
