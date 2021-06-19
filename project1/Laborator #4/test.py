import numpy as np
import matplotlib.pyplot as plt


def ex1():

    x = np.sqrt(13)

    return 0


def ex2():                  # Exercitiul 2
    A = np.array([          # Datele problemei
        [-1, -4, -7],
        [-2, 4, -6],
        [-3, 5, -6]
    ], dtype = np.int32)
    dim = A.shape           # dim = [linii, coloane]
    print(f'Numarul de linii: {dim[0]} \nNumarul de coloane: {dim[1]}\n')   # a)

    A = np.array(A).astype(np.int64)  # b)

    temp = A[1]
    print(f'Linia 2: {temp}\n')     # c)

    prod = A[1] @ A.T[1]
    print(f'Produsul scalar intre linia 2 si coloana 2: {prod}\n')

    return 0


def func(x):
    y = x ** 3 + 8 * x ** 2 + 17 * x + 10
    return y


def ex3():
    interval = [-5, 5]
    plt.figure(0)
    x = np.linspace(interval[0], interval[1], 77)   # Discretizarea de 77 de puncte echidistante
    y = func(x)
    plt.plot(x, y, label='f(x)')

    plt.axhline(0, c='black')  # Adauga axa OX
    plt.axvline(0, c='black')  # Adauga axa OY

    plt.xlabel('x')
    plt.ylabel('f(x) = y')

    plt.grid(b = True)


    plt.title('Grafic')
    plt.legend()

    plt.show(c='cyan', l='dot')  # Am incercat


if __name__ == '__main__':
    # ex1()
    # ex2()
    ex3()