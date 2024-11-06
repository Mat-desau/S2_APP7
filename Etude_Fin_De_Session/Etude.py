import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

def hp_yeet():
    b = [((1.937e12)*pi)]
    a = [1, (1562*pi), (3449468*pi), (3949075461*pi), ((1.937e12)*pi)]

    p = np.roots(a)

    print('Poles\n', p, '\n')

    a1 = np.poly((p[0], p[1]))
    a2 = np.poly((p[2], p[3]))
    b1 = [a1[2]]
    b2 = [a2[2]]

    print('Poly\n', b1)
    print(a1, '\n')
    print(b2)
    print(a2, '\n')

    wc1 = np.sqrt(b1[0])
    wc2 = np.sqrt(b2[0])

    print('Wc\n', wc1)
    print(wc2, '\n')

    Q1 = wc1/a1[1]
    Q2 = wc2/a2[1]

    print('Q\n', Q1)
    print(Q2, '\n')

def hp_yeet2():
    b = [1]
    a = [1, 3.124, 4.392, 3.201, 1]
    s = 1/(250*2*pi)

    a[0] = a[0] * (s**4)
    a[1] = a[1] * (s**3)
    a[2] = a[2] * (s**2)
    a[3] = a[3] * (s**1)
    a[4] = a[4] * (s**0)

    a[1] = a[1] / a[0]
    a[2] = a[2] / a[0]
    a[3] = a[3] / a[0]
    a[4] = a[4] / a[0]
    a[0] = a[0] / a[0]

    b = a[4]

    P = np.roots(a)

    a1 = np.poly((P[0], P[1]))
    b1 = a1[2]
    a2 = np.poly((P[2], P[3]))
    b2 = a2[2]

    Q1 = np.sqrt(b1) / a1[1]
    Q2 = np.sqrt(b2) / a2[1]

    print(Q1)
    print(Q2)

def main():
    hp_yeet2()

if __name__ == '__main__':
    main()