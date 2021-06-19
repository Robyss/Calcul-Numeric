"""
Titlu: Calcul Numeric, Laboratorul #02
Autor: Alexandru Ghita, februarie 2021
"""
import sys
import numpy as np
import matplotlib.pyplot as plt


def newton_raphson(f, df, x0, eps, max_iter):
    # Pasul 1
    i = 1

    # Pasul 2
    while i <= max_iter:
        x1 = x0 - f(x0)/df(x0)  # Pasul 3

        # Pasul 4 -> Criteriul de oprire
        if np.abs(x1 - x0) < eps:
            return x1, i

        # Pasul 5
        i += 1

        # Pasul 6
        x0 = x1

    # Pasul 7
    # print('Metoda N-R nu a atins convergenta dupa {} iteratii.'.format(i))
    print(f'Metoda N-R nu a atins convergenta dupa {i} iteratii.')
    sys.exit(1)  # Intrerupe rularea script-ului cu Status Code 1

def secanta(f, x0, x1, eps, max_iter):
    i = 2
    y0 = f(x0)
    y1 = f(x1)

    while i<=max_iter:
        x_aprox = x1 - y1*(x1 - x0)/(y1 - y0)

        if np.abs(x1 - x_aprox) < eps:
            return x_aprox, i

        i += 1
        x0 = x1
        y0 = y1
        x1 = x_aprox
        y1 = f(x_aprox)

    print(f'Metoda Secantei nu a atins convergenta dupa {i} iteratii.')
    sys.exit(1)

def pozitie_falsa(f, x0, x1, eps, max_iter):
    i = 2
    y0 = f(x0)
    y1 = f(x1)

    while i<=max_iter:
        x_aprox = x1 - y1*(x1 - x0)/(y1 - y0)

        if np.abs(x_aprox - x1) < eps:
            return x_aprox, i

        i += 1
        y_aprox = f(x_aprox)

        if y_aprox * y1 < 0:
            y0 = y1
            x0 = x1

        x1 = x_aprox
        y1 = y_aprox

    print(f'Metoda Pozitiei False nu a atins convergenta dupa {i} iteratii')
    sys.exit(1)


def func1(x):
    y = -x**3 - 2*np.cos(x)
    return y


def dfunc1(x):
    y = - 3 * x**2 + 2*np.sin(x)
    return y


def plot_function(interval, functions, points=None, fig_num=0, title=None):
    plt.figure(fig_num)
    domain = np.linspace(interval[0], interval[1])

    for func in functions:
        plt.plot(domain, func[0](domain), label=func[1])

    if points:
        for point in points:
            plt.scatter(point[0], 0, label=point[1])

    plt.axhline(c='Black')
    plt.axvline(c='Black')

    plt.xlabel('points')
    plt.ylabel('values')

    if title:
        plt.title(title)

    plt.grid()
    plt.legend()
    plt.show()


def ex2_newton_raphson():
    X0 = -3
    EPS = 1e-5
    MAX_ITER = 1000
    x_num, steps = newton_raphson(f=func1, df=dfunc1, x0=X0, eps=EPS, max_iter=MAX_ITER)
    print(f'Solutia ecuatiei f(x) = 0 cu metoda Newton-Raphson este x_sol = {x_num:.5f} gasita in N = {steps} pasi.')

    plot_function(interval=[-3, 3],
                  functions=[(func1, 'f(x)'), (dfunc1, "f'(x)")],
                  points=[(x_num, 'xnum')],
                  title='Exercitiul 1, Metoda N-R')


def ex2_secanta():
    EPS = 1e-5
    MAX_ITER = 1000
    x_num, steps = secanta(f=func1, x0=0, x1=1, eps=EPS, max_iter=MAX_ITER)

    print(f'Solutia ecuatiei f(x) = 0 cu metoda secantei este x_sol = {x_num:.5f} gasita in N = {steps} pasi.')

    plot_function(interval=[-3, 3],
                  functions=[(func1, 'f(x)')],
                  points=[(x_num, 'xnum')],
                  title='Exercitiul 1, Metoda Secantei')


def ex2_pozitie_falsa():
    EPS = 1e-5
    MAX_ITER = 1000
    x_num, steps = pozitie_falsa(f=func1, x0=-3, x1=3, eps=EPS, max_iter=MAX_ITER)

    print(f'Solutia ecuatiei f(x) = 0 cu metoda pozitiei false este x_sol = {x_num:.5f} gasita in N = {steps} pasi.')

    plot_function(interval=[-3, 3],
                  functions=[(func1, 'f(x)')],
                  points=[(x_num, 'xnum')],
                  title='Exercitiul 1, Metoda Pozitiei False')


if __name__ == '__main__':  # Stand alone
    # ex2_newton_raphson()
    # ex2_secanta()
    ex2_pozitie_falsa()
