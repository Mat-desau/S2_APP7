import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import sounddevice as sd
import helpers as hp

pi = np.pi

def Question1_1():
    max = 0.001
    f0 = 500
    f1 = 10000
    fe = 44100
    t = np.linspace(0, max, int(max*fe))

    w = signal.chirp(t, f0, max, f1, method='quadratic')

    sd.play(w)

    plt.figure()
    plt.plot(t, w)

    plt.title('Numero 1.1')

def Question1_2():
    max = 0.0010
    f0 = 500
    f1 = 10000
    fe = 11000
    t = np.linspace(0, max, int(max*fe))

    w = signal.chirp(t, f0, max, f1, method='quadratic')

    sd.play(w)
    plt.figure()

    plt.plot(t, w)

    plt.title('Numero 1.2')

def Question1_2():
    max = 0.020
    f0 = 500
    f1 = 500
    fe1 = 50000
    fe2 = 600
    t1 = np.linspace(0, max, int(max*fe1))
    t2 = np.linspace(0, max, int(max*fe2))


    w1 = signal.chirp(t1, f0, max, f1, method='quadratic')
    w2 = signal.chirp(t2, f0, max, f1, method='quadratic')

    sd.play(w1)
    sd.play(w2)

    plt.figure()

    plt.plot(t1, w1)
    plt.plot(t2, w2)
    plt.title('Numero 1.3')

def Question2_1_2():
    max = 10
    f0 = 500
    f1 = 10000
    fe = 44100
    t = np.linspace(0, max, int(max*fe))

    w = signal.chirp(t, f0, max, f1, method='quadratic')

    sd.play(w)

    plt.figure()

    y8 = hp.digitize(w, 8)
    y4 = hp.digitize(w, 4)
    y2 = hp.digitize(w, 2)

    plt.plot(t, w)
    plt.plot(t, y8)
    plt.plot(t, y4)
    plt.plot(t, y2)
    plt.xlim((0, 0.001))
    plt.legend(['Normal', '8 bits', '4 bits', '2 bits'])

    plt.title('Numero 2.1')

    E8 = np.sqrt((1/len(w))*((np.sum((w-y8)**2))))
    E4 = np.sqrt((1/len(w))*((np.sum((w-y4)**2))))
    E2 = np.sqrt((1/len(w))*((np.sum((w-y2)**2))))

    print('E8 = ', E8, '\nE4 = ', E4, '\nE2 = ', E2)

def Question3():
    b, a = signal.bessel(4, 1, btype='low', analog=True)

    print('1.\nb = ', b, '\na = ', a, '\n')

    #hp.bodeplot(b, a, 'Question 3.2')

    p = np.roots(a)
    a1 = np.poly([p[0], p[1]])
    a2 = np.poly([p[2], p[3]])

    print('3.\na1 = ', a1, '\na2 = ', a2, '\n')

    fc = 1500*2*pi
    b_4, a_4 = signal.bessel(4, fc, btype='low', analog=True)

    print('4.\nb = ', b_4, '\na = ', a_4, '\n')

    #hp.bodeplot(b_4, a_4, 'Question 3.5')



def main():
    Question3()
    plt.show()

if __name__ == "__main__":
    main()
