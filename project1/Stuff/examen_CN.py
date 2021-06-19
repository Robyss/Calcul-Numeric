import numpy as np
import matplotlib.pyplot as plt

######################################
#   Exercitiul 1
######################################


def function(x):
    y = 1/(1 + x**2)
    return y


def int_romberg(a, b, n, f):

    assert a < b, 'Intervalul este eronat'
    # Pasul 1
    h = b - a + 1   # lungimea intervalului

    # Pasul 2
    Q = np.full(shape=(n, n), fill_value=np.nan)    # Construim matricea Q

    Q[0, 0] = (f(a) + f(b)) * h / 2

    for i in range(1, n):
        suma = 0
        for k in range(1, 2**i):
            suma += f(a + (k) * h / (2**i))

        Q[i, 0] = h * (f(a) + 2 * suma + f(b)) / 2**(i+1)

    for i in range(1, n):
        for j in range(1, i + 1):
            Q[i,j] = (4**j * Q[i,j-1] - Q[i-1,j-1]) / (4**j - 1)

    I = Q[-1, -1]
    return I


def ex1():

    # b) I = arctan(5) == 1.3734
    integrala = 1.3734
    # c) Aproximam integrala folosind metoda Romberg

    # Capetele integralelor
    a = -5
    b = 5
    n = 4
    I_romberg = int_romberg(a, b, n, function)
    print(I_romberg)

    # d)
    error_romberg = np.abs(integrala - I_romberg)
    print(f'Eroare aproximarii integralei este: {error_romberg}')

######################################
#   Exercitiul 2
######################################

def f(x):
    y = np.cos(3 * x)   # Functia f(x) = cos(3x)
    return y


def interp_lagr(X, Y, z):
    # Retinem dimensiune vectorilor
    n = np.shape(X)[0] - 1  # -1 deoarece n-ul vine cu +1

    L = np.full(shape=(n + 1, n + 1), fill_value=np.nan)  # Initializam matricea L

    # Pasul 1
    for k in range(n + 1):
        produs= 1
        for j in range(n + 1):
            if k == j:  # Numitor != 0
                continue
            produs *= (z - X[j]) / (X[k] - X[j])
            L[n, k] = produs

    # Pasul 2
    t = 0
    t = L[n, :] @ Y[:]

    return t


def ex2():

    # b) Generarea datelor cunoscute X si Y
    N = 13

    X = np.full(shape=N+1, fill_value=np.nan)
    X[:-1] = np.linspace(0, np.pi, N)
    h = X[2] - X[1]
    X[-1] = X[-2] + h

    Y = f(X)

    print(f'Datele generate cunoscute sunt X:{X}')
    print(f'Datele generate cunoscute sunt Y={Y}')

    # c) Afisarea graficului cu datele generate
    plt.figure(0)
    plt.scatter(X[:-1], Y[:-1], marker='*', c='red', s=10, label='Puncte')


    # d)

    domain = np.linspace(0, np.pi, 80)  # 80 de puncte echidistante
    f_domain = f(domain)

    f_interp_lagr = np.array([interp_lagr(X, Y, z) for z in X[:-1]])

    # e) Afisarea functiei exacte si a aproximarilor de la pasul d)

    plt.plot(domain, f_domain, label='functia exacta')
    plt.plot(X[:-1], f_interp_lagr, label='interp lagrange')

    # Aleg sa pastrez aceste instructiuni la finalul generarii graficului

    plt.grid()
    plt.axhline(0, c='black')
    plt.axvline(0, c='black')
    plt.xlabel('Points')
    plt.ylabel('Values')
    plt.legend()
    plt.title('Graficul de interpolare')

    # f) Graficul erorii de interpolare
    plt.figure(1)



    error = np.full(shape=80, fill_value=np.nan)  # eroare reprezinta e_t
    for i in range(80):
        error[i] = np.abs(Y[i] - f_interp_lagr[i])
    plt.plot(X, error, label='Valoare eroare')
    plt.grid()
    plt.axhline(0, c='black')
    plt.axvline(0, c='black')
    plt.xlabel('Points')
    plt.ylabel('Values')
    plt.legend()
    plt.title('Erorile interpolarii')

    plt.show()  # Un singur show() la finalul functiei


if __name__ == '__main__':
    ex1()
    ex2()
