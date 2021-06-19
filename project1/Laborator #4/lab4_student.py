"""
Titlu: Calcul Numeric, Laboratorul #04
Autor: Alexandru Ghita, aprilie 2021
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
    assert np.shape(U)[0] == np.shape(U)[1], "U nu este matrice patratica!"
    n = np.shape(U)[0]  # Salvam in variabila n dimensiunea matricei

    # Verifica daca matricea U este superior triunghiulara
    # assert np.all(U == np.triu(U)), "U nu este superior triunghiulara!"
    assert np.allclose(U, np.triu(U), atol=1e-7), "U nu este superior triunghiulara!"

    # Verifica daca U si b au acelasi numar de linii
    assert n == np.shape(b)[0], "U si b nu au acelasi numar de linii!"

    # Verifica daca L este inversabila:
    # Determinant matrice superior triunghiulara = Produs elemente diagonala
    assert np.abs(np.prod(U.diagonal())) > 0, "Matricea U nu este inversabila!"

    # Initializeaza vectorul x ca vector coloana numpy
    x = np.full(shape=b.shape, fill_value=np.nan)

    if vectorized:
        """ Metoda substitutiei descendente, vectorizata si optimizata. """
        # Mergem de la ultima linie la prima
        for k in range(n-1, -1, -1):
            x[k] = (b[k] - U[k, k+1:] @ x[k+1:]) / U[k, k]
    else:
        """ Metoda substitutiei descendente: Varianta explicita, nevectorizata. """
        x[n-1] = b[n-1] / U[n-1, n-1]
        for k in range(n-2, -1, -1):
            suma = 0
            for j in range(n-1, k, -1):
                suma += U[k, j] * x[j]
            x[k] = (b[k] - suma) / U[k, k]

    return x


def subs_asc(L, b):
    """
        Metoda Substitutiei Ascendente: Determina solutia sistemului inferior triunghiular L * x = b
        folosind metoda substitutiei ascendente.

    :param L (numpy square matrix) = matrice inferior triunghiulara
    :param b (numpy column vector) = coloana termenilor liberi
    :return x (numpy column vector) = solutia sistemului
    """

    return 0


def fact_lu(A):

    return 0, 0, 0


def fact_qr(A):

    return 0, 0


def ex1():
    A = np.array([
        [2., 0., 0.],
        [3., 4., 0.],
        [1., 3., 1.]
    ], dtype=np.float32)

    b = np.array([
        [2],
        [11],
        [10]
    ], dtype=np.float32)

    x = np.array([
        [1],
        [2],
        [3]
    ], dtype=np.float32)

    x_sol = subs_asc(L=A, b=b)

    is_correct = np.allclose(x, x_sol)
    print(f'Am implementat corect metoda substitutiei ascendente? {is_correct}')


def ex2():
    """ ! Pana nu implementam, metoda o sa dea eroare ! """
    A = np.array([
        [2., 3., 0.],
        [3., 4., 2.],
        [1., 3., 1.]
    ], dtype=np.float32)

    b = np.array([
        [8],
        [17],
        [10]
    ], dtype=np.float32)

    x = np.array([
        [1],
        [2],
        [3]
    ], dtype=np.float32)

    L, U, w = fact_lu(A=A)

    # Modificarea lui b
    b_prime = b[w]

    # Primul sistem
    y = subs_asc(L=L, b=b_prime)

    # Al doilea sistem
    x_sol = subs_desc(U=U, b=y)

    # Verificarea implementarii
    is_correct = np.allclose(x, x_sol)
    print(f'Am implementat corect factorizarea LU? {is_correct}')


def ex3():
    """ ! Pana nu implementam, metoda o sa dea eroare ! """
    A = np.array([
        [2., 3., 0.],
        [3., 4., 2.],
        [1., 3., 1.]
    ], dtype=np.float32)

    b = np.array([
        [8],
        [17],
        [10]
    ], dtype=np.float32)

    x = np.array([
        [1],
        [2],
        [3]
    ], dtype=np.float32)

    Q, R = fact_qr(A=A)

    # Modificarea lui b
    b_mod = Q.T @ b

    # Rezolvarea sistemului
    x_sol = subs_desc(U=R, b=b_mod)

    # Verificarea implementarii
    is_correct = np.allclose(x, x_sol)
    print(f'Am implementat corect factorizarea QR? {is_correct}')


if __name__ == '__main__':
    ex1()
    # ex2()
    # ex3()
