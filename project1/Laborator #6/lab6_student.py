"""
Titlu: Calcul Numeric, Laboratorul #06
Autor: Alexandru Ghita, mai 2021
"""
import numpy as np
import matplotlib.pyplot as plt


def deriv_num(X, Y, metoda='ascendente'):

    return 0


def func1(x):

    return 0


def dfunc1(x):
    return 1


def ex1():
    """ Hint: Tine cont ca, in discretizarea intervalului ales, sunt create doua noduri suplimentare
    la capetele intervalului. """
    pass


def interp_directa(X, Y, z):
    """ Stefan Catalin Cucu (163) version from Lab#5 """
    A = np.vander(X, increasing=True)
    a = np.linalg.solve(A, Y)
    t = np.polyval(np.flip(a), z)

    return t


def spline_liniara(X, Y, z):
    """ Metoda de interpolare spline liniara.

    :param X: X = [X0, X1, ..., Xn]
    :param Y: [Y0=f(X0), Y1=F(X1), ..., Yn=f(Xn)]
    :param z: Punct in care doresti o valoare aproximata a functiei
    :return: t: Valoarea aproximata calculata in z
    """

    return 0


def spline_patratica(X, Y, dfa, z):
    """ Metoda de interpolare spline patratica.

    :param X: X = [X0, X1, ..., Xn]
    :param Y: [Y0=f(X0), Y1=F(X1), ..., Yn=f(Xn)]
    :param dfa: Valoarea derivatei in capatul din stanga al intervalului folosit
    :param z: Punct in care doresti o valoare aproximata a functiei
    :return: t: Valoarea aproximata calculata in z
    """

    return 0, 0


def func2(x):
    return 0


def dfunc2(x):
    return 0


def ex2():
    pass


def ex3():
    x_clicks_all = [55., 69, 75, 81, 88, 91, 95, 96, 102, 108, 116, 126, 145, 156, 168, 179, 193, 205, 222, 230, 235,
                    240, 242, 244, 253, 259]

    y_clicks_all = [162., 176, 188, 209, 229, 238, 244, 256, 262, 259, 254, 260, 262, 265, 263, 260, 259, 252, 244, 239,
                    233, 227, 226, 224, 224, 221]

    select_from = 1  # Extrage click-uri din 'select_from' in 'select_from'

    x_clicks = x_clicks_all[::select_from]
    y_clicks = y_clicks_all[::select_from]

    domain = np.linspace(x_clicks[0], x_clicks[-1], 100)

    N = len(x_clicks) - 1

    inter_lagr_direct = np.zeros_like(domain)  # TODO: To be modified
    inter_spline_liniara = np.zeros_like(domain)  # TODO: To be modified
    dfa = 0  # TODO: To be modified
    inter_spline_patratica = np.zeros_like(domain)  # TODO: To be modified

    # Afisare grafic figura
    plt.figure(5)

    # Afisare date client pe grafic
    plt.scatter(x_clicks, y_clicks, marker='*', c='red', s=10, label='Clicks')

    # Afisare doggo
    image = np.load('frida_doggo.npy')
    plt.imshow(image, extent=[0, 300, 0, 300])

    # Afisare grafic aproximare
    plt.plot(domain, inter_lagr_direct, c='w', linewidth=2, linestyle='-.', label='Metoda Lagrange')
    plt.plot(domain, inter_spline_liniara, c='g', linewidth=2, linestyle='-', label='Spline Liniara')
    plt.plot(domain, inter_spline_patratica, c='b', linewidth=2, linestyle='--', label='Spline Patratica')
    plt.title('Interpolare, N={}'.format(N))
    plt.legend()
    plt.xlim([-1, 305])
    plt.ylim([-1, 305])
    plt.xlabel('x')
    plt.ylabel('y = f(x)')
    plt.show()


if __name__ == '__main__':
    # ex1()
    # ex2()
    ex3()
