"""
Titlu: Calcul Numeric, Laboratorul #05
Autor: Alexandru Ghita, aprilie 2021
"""
import numpy as np
import matplotlib.pyplot as plt


def interp_directa(X, Y, z):
    """ Author: Stefan Catalin Cucu (163) -- version from Lab#5

       :param X: X = [X0, X1, ..., Xn]
       :param Y: [Y0=f(X0), Y1=F(X1), ..., Yn=f(Xn)]
       :param z: Punct in care doresti o valoare aproximata a functiei
       """
    A = np.vander(X, increasing=True)
    a = np.linalg.solve(A, Y)
    t = np.polyval(np.flip(a), z)

    return t


def interp_lagrange(X, Y, z):

    return 0


def interp_newton(X, Y, z):

    return 0


def interp_newton_dd(X, Y, z):

    return 0


def f(x):
    y = np.e ** (2 * x)
    return y


def ex1():

    """ (a) Generare date client si vizualizarea acestora. """
    # Genereaza datele

    domain = np.linspace(-1, 1, 100)
    f_domain = f(domain)

    N = 16

    # Creaza o figura noua in care sa vizualizezi datele generate
    plt.figure(0)

    """ (b) Aproximarea valorilor lipsa. """
    # Discretizeaza domeniul
    X = np.full(shape=N + 2, fill_value=np.nan)
    X[1:-1] = np.linspace(-1, 1, N)
    h = X[2] - X[1]
    X[0] = X[1] - h
    X[-1] = X[-2] + h
    Y = f(X)
    # Calculeaza aproximarea in fiecare punct din domeniu

    inter_lagr_direct = np.zeros_like(domain)  # TODO: To be modified   - Modificat
    for i in range(len(inter_lagr_direct)):
        inter_lagr_direct[i] = interp_directa(X, Y, domain[i])

    """ (c) Generare grafic (verificare). """
    # Ploteaza in figura de la punctul (a) (plt.show() trebuie sa fie activ doar aici acum)

    plt.figure(0)
    plt.scatter(X[1:-1], Y[1:-1], marker='*', c='red', s=10, label='Clicks')
    plt.plot(domain, f_domain, c='blue', label='functia exacta')
    plt.plot(domain, inter_lagr_direct, label='interp_directa')
    plt.legend()
    plt.show()
    """ (d) Graficul erorii de interpolare. """
    # Calculeaza eroarea

    # Genereaza o figura noua si afiseaza graficul erorii


if __name__ == '__main__':
    ex1()
