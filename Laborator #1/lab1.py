import numpy as np
import matplotlib.pyplot as plt

A = 0   # Capatul din stanga al intervalului
B = 4   # Capatul din stanga al intervalului
epsilon = 10**(-5)  # '1e-5' Eroarea maxima acceptata


def function(x):
    y = x**3 - 7* x**2 + 14*x - 6

    return y


def metoda_bisectiei(a, b, f, eps):
    # x_num = (a + b) / 2
    x_num = a + (b - a) / 2

    N = np.log2((b - a) / eps) # N:18.0
    N = np.floor(N)
    N = int(N)

    #   Iteratiile algoritmului

    for i in range (1, N):
        # Verificam daca nu cumva am dat fix peste solutie
        if f(x_num) == 0:
            break
        elif np.sign( f(a) ) * np.sign( f(x_num) ) < 0:
            b = x_num
        else:
            a = x_num

        x_num = a + (b - a) / 2

    return x_num, N


print(   metoda_bisectiei(a = A, b = B, f = function, eps = epsilon)  )

domain = np.linspace(A, B, 500)

plt.figure(0)
plt.plot(domain, function(domain))
plt.show()


