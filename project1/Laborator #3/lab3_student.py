"""
Titlu: Calcul Numeric, Laboratorul #03
Autor: Alexandru Ghita, martie 2021
"""
# Importarea librariei 'numpy' cu alias-ul 'np'
import numpy as np


def subs_desc(U, b, vectorized=True):
    """
        Metoda Substitutiei Descendente: Determina solutia sistemului superior triunghiular U * x = b
        folosind metoda substitutiei descendente.

    :param U (numpy square matrix) = matrice superior triunghiulara
    :param b (numpy column vector) = coloana termenilor liberi
    :param vectorized (bool) = choose whether to vectorize or not
    :return x (numpy column vector) = solutia sistemului
    """
    # Verifica daca matricea U este patratica
    assert np.shape(U)[0] == np.shape(U)[1], "Matricea nu este patratica!"
    n = np.shape(U)[0]

    # Verifica daca matricea U este superior triunghiulara
    assert np.allclose(U, np.triu(U), atol=1e-3), "Matricea nu este superior triunghiulara!"

    # Verifica daca U si b au acelasi numar de linii
    assert n == b.shape[0], "A si b nu au acelasi numar de linii"

    # Verifica daca L este inversabila
    assert np.prod(np.diagonal(U)) != 0, "Matricea nu este inversabila"

    # Initializeaza vectorul x ca vector coloana numpy
    x = np.full(shape=b.shape, fill_value=np.nan)

    if vectorized:
        """ Metoda substitutiei descendente, vectorizata si optimizata. """
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

    else:
        """ Metoda substitutiei descendente: Varianta explicita, nevectorizata. """
        x[-1] = b[-1] / U[-1, -1]
        for i in range(n-2, -1, -1):
            suma = 0
            for j in range(i + 1, n):
                suma += U[i, j] * b[j]
            x[i] = (b[i] - suma) / U[i, i]

    return x


def meg_fp(a, b):

    return 0


def meg_pp(a, b):

    return 0


def ex2():
    A = np.array([
        [2, 3, 0],
        [3, 4, 2],
        [1, 3, 1]
    ], dtype=np.float32)
    b = np.array([
        [8],
        [17],
        [10],
    ], dtype=np.float32)

    c = np.eye(3, 3)

    d = subs_desc(c, b, vectorized=True)
    print(d)
    print('Solutia gasita cu MEG_FP este x_sol = ', None)
    print('Solutia gasita cu MEG_PP este x_sol = ', None)


if __name__ == '__main__':
    ex2()
