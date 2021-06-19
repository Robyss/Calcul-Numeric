import numpy as np
import matplotlib.pyplot as plt

#################################################
# Exercitiul 1
#################################################


def metoda_bisectiei(a, b, f, epsilon):

    assert a < b, f'Intervalul [ {a}, {b} ] este ales gresit'

    assert np.sign(f(a)) * np.sign(f(b)) < 0, f'Nu exista radacina pe intervalul [ {a}, {b} ]'

    x_num = a + (b - a) / 2

    N = int(np.log2((b - a) / epsilon))
    for _ in range(1, N):
        if f(x_num) == 0:
            break
        elif np.sign(f(a)) * np.sign(f(x_num)) < 0:
            b = x_num
        else:
            a = x_num

        x_num = a + (b - a) / 2
    return x_num


def function(x):
    y = x**2 - 31
    return y


def ex1():

    sol = [np.sqrt(31), -np.sqrt(31)]

    # Alegem un interval ce indeplineste toate conditiile metodei
    A = -10  # Capatul din stanga al intervalului
    B = 10  # Capatul din dreapta al intervalului

    EPSILON = 1e-3
    NUM_POINTS = 50  # In cate puncte sa discretizam intervalul
    x = np.linspace(A, B, NUM_POINTS)  # Discretizare a intervalului [A, B] in NUM_POINTS puncte
    y = function(x)  # Valorile functiei pentru punctele din discretizare
    x_num1 = metoda_bisectiei(A, 0, function, EPSILON)
    x_num2 = metoda_bisectiei(0, B, function, EPSILON)

    # Afisare grafic

    plt.figure(0)  # Initializare figura
    plt.plot(x, y, label='f(x)')  # Plotarea functiei
    plt.scatter(x_num1, 0, label='sol1')  # Adaugare solutia 1 in grafic
    plt.scatter(x_num2, 0, label='sol2')  # Adaugare solutia 2 in grafic

    plt.axhline(0, c='black')  # Adauga axa OX
    plt.axvline(0, c='black')  # Adauga axa OY
    plt.xlabel('x')  # Label pentru axa OX
    plt.ylabel('f(x) = y')  # Label pentru axa OY
    plt.title('Metoda Bisectiei')  # Titlul figurii
    plt.grid(b=True)  # Adauga grid
    plt.legend()  # Arata legenda
    plt.show()  # Arata graficul

    return x


#########################################################
# Exercitiul 2
#########################################################
def functie(x):
    y = np.e ** (2*x)
    return y


def interp_neville(X, Y, z):
    n = len(X)
    Q = np.zeros((n, n))

    # Pasul 1
    Q[:, 0] = Y[:]

    for i in range(1, n):
        for j in range(1, i + 1):
            Q[i, j] = (Q[i, j - 1] * (z - X[i - j]) - Q[i - 1, j - 1] * (z - X[i]))/(X[i] - X[i - j])   # Bug
            # Eroare pentru / (X[i] - X[i-j + 1])

    t = Q[-1, -1]
    return t


def ex2():
    # Genereaza datele
    X = np.linspace(-1, 1, 16)
    Y = functie(X)
    # Creaza o figura noua in care sa vizualizezi datele generate
    plt.figure(0)

    plt.plot(X, Y, label='f(x)')
    plt.scatter(X, Y, label='Date Client', marker='*', color='red')

    plt.axhline(0, c='black')  # Adauga axa OX
    plt.axvline(0, c='black')  # Adauga axa OY
    plt.xlabel('x')  # Label pentru axa OX
    plt.ylabel('f(x) = y')  # Label pentru axa OY
    plt.title('Metoda de Interpolare Neville')  # Titlul figurii
    plt.grid(b=True)  # Adauga grid
    # plt.legend()  # Arata legenda
    # plt.show()  # Arata graficul

    # Discretizare domeniu

    X_2 = np.linspace(-1, 1, 75)
    Y_2 = functie(X_2)

    aprox = np.zeros(75).astype(np.float32)
    for i in range(75):
        aprox[i] = interp_neville(X_2, Y_2, i)

    plt.plot(X_2, aprox, label='Aproximare', linestyle='--', color='black')
    plt.legend()
    plt.show()

    return 0

#################################################
# Exercitiul 3
#################################################

def subs_desc(U, b, vectorized=True):
    # Verifica daca matricea U este patratica
    assert np.shape(U)[0] == np.shape(U)[1], "U nu este matrice patratica!"
    n = np.shape(U)[0]  # Salvam in variabila n dimensiunea matricei

    # Verifica daca matricea U este superior triunghiulara

    assert np.allclose(U, np.triu(U), atol=1e-7), "U nu este superior triunghiulara!"

    # Verifica daca U si b au acelasi numar de linii
    assert n == np.shape(b)[0], "U si b nu au acelasi numar de linii!"

    # Verifica daca U este inversabila:
    # Determinant matrice superior triunghiulara = Produs elemente diagonala
    assert np.abs(np.prod(U.diagonal())) > 0, "Matricea U nu este inversabila!"

    # Initializeaza vectorul x ca vector coloana numpy
    x = np.full(shape=b.shape, fill_value=np.nan)

    if vectorized:
        # Mergem de la ultima linie la prima
        for k in range(n-1, -1, -1):
            x[k] = (b[k] - U[k, k+1:] @ x[k+1:]) / U[k, k]
    else:
        x[n-1] = b[n-1] / U[n-1, n-1]
        for k in range(n-2, -1, -1):
            suma = 0
            for j in range(n-1, k, -1):
                suma += U[k, j] * x[j]
            x[k] = (b[k] - suma) / U[k, k]

    return x


def fact_qr_new(A):

    # a)
    assert A.shape[0] == A.shape[1], 'Matricea A nu este patratica'

    n = A.shape[0]  # Salvam numarul de linii
    # Asiguram ca matricea este inversabila
    assert np.linalg.det(A) != 0, 'Sistemul nu este compatibil'

    R = np.zeros((n, n)).astype(np.float32)
    Q = np.zeros((n, n)).astype(np.float32)

    # Pasul 1
    for i in range(n):
        R[0, 0] += A[i, 1] ** 2
    R[0, 0] = np.sqrt(R[0, 0])

    Q[:, 0] = A[:, 0] / R[0, 0]

    for j in range(1, n):
        R[0, j] += Q[:, 0] @ A[:, j]

    # Pasul 2
    for k in range(1, n):
        suma1 = 0
        suma2 = 0
        for i in range(n):
            suma1 += A[i, k] ** 2
        for s in range(k-1):
            suma2 += R[s, k] ** 2

        R[k, k] = np.sqrt(suma1 - suma2)    # Atribuim diagonala lui R

        for i in range(n):
            Q[i, k] = (A[i,k] - Q[i, :k-1] @ R[:k-1, k]) / R[k, k]  # Completam coloana K a matricei Q

        for j in range(k, n):
            R[k, j] = Q[:, k] @ A[:, j] # Completam linia k a matricei R

    return Q, R


def ex3():

    A = np.array([
        [0, -7, -10, 9],
        [0, -8, 1, -4],
        [-8, -3, -5, -5],
        [-5, -10, 0, -8],
    ], dtype=np.float32)

    b = np.array([
        [-8],
        [-29],
        [-49],
        [-57],
    ], dtype=np.float32)

    Q, R = fact_qr_new(A=A)
    # b)
    # Modificarea lui b
    b_mod = Q.T @ b

    # Rezolvarea sistemului
    x_sol = subs_desc(U=R, b=b_mod)
    print(f'Solutia sistemului folosind factorizarea QR este:\n {x_sol}')


if __name__ == '__main__':
    ex1()
    ex2()
    ex3()