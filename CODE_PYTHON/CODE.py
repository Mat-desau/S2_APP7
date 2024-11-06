import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import helpers as hp

pi = np.pi

def dB(Gain):
    dB = 20*np.log10(Gain)

    return dB

def Bode(min, max, nb, Wn, A0, A1, rangemin, rangetop, prt):
    w = np.linspace(min, max, nb)

    Rdb = 20*np.log10(A0/A1)

    place = ((2*Wn)*nb)/max
    incertude = 0.05*2*Wn

    Legende = []

    figure = plt.figure()
    figure.suptitle('Lieu de Bode')


    for x in range(rangemin, rangetop):
        b1, a1 = signal.butter(x, Wn, btype='low', analog=True)
        b2, a2 = signal.cheby1(x, Rdb, Wn, btype='low', analog=True)
        b3, a3 = signal.bessel(x, Wn, btype='low', analog=True)

        w1, h1 = signal.freqs(b1, a1, w)
        w2, h2 = signal.freqs(b2, a2, w)
        w3, h3 = signal.freqs(b3, a3, w)

        h1 = 20*np.log10(np.abs(h1))
        h2 = 20*np.log10(np.abs(h2))
        h3 = 20*np.log10(np.abs(h3))

        string = 'Ordre ' + str(x)
        fig1 = plt.subplot(3, 1, 1)
        fig2 = plt.subplot(3, 1, 2)
        fig3 = plt.subplot(3, 1, 3)

        fig1.semilogx(w1, h1)
        fig2.semilogx(w2, h2)
        fig3.semilogx(w3, h3)

        fig1.set_title('Butter')
        fig2.set_title('Cheby')
        fig3.set_title('Bessel')

        Legende = np.append(Legende, string)

        if prt:
            print('\nValeur à ', Wn, ' (2*fc) de l\'', string, '\nButter : ', h1[int(place)], '\nCheby : ', h2[int(place)], '\nBessel : ', h3[int(place)])

    #fig1.hlines(-25, (2*Wn)-incertude, (2*Wn)+incertude, colors='black')
    #fig2.hlines(-25, (2*Wn)-incertude, (2*Wn)+incertude, colors='black')
    #fig3.hlines(-25, (2*Wn)-incertude, (2*Wn)+incertude, colors='black')

    debutpente = Wn-(Wn*0.25)
    limmaxx = [0, (Wn)]
    limmaxy = [dB(A0), dB(A0)]
    limminx = [0, debutpente]
    limminy = [dB(A1), dB(A1)]
    limminpentex = [debutpente, Wn]
    limminpentey = [dB(A1), dB(A1-(A1*0.5))]
    limmaxx2 = [(Wn), (2*Wn)]
    limmaxy2 = [dB(A0), -25]
    limminx2 = [Wn, Wn]
    limminy2 = [dB(A1-(A1*0.5)), -50]

    largeur = 2

    fig1.legend(Legende)
    fig1.semilogx(limmaxx, limmaxy, color='black', linewidth=largeur)
    fig1.semilogx(limmaxx2, limmaxy2, color='black', linewidth=largeur)
    fig1.semilogx(limminx, limminy, color='black', linewidth=largeur)
    fig1.semilogx(limminx2, limminy2, color='black', linewidth=largeur)
    fig1.semilogx(limminpentex, limminpentey, color='black', linewidth=largeur)
    fig1.set_ylim((-55, 10))
    fig1.grid(color='grey')

    fig2.legend(Legende)
    fig2.semilogx(limmaxx, limmaxy, color='black', linewidth=largeur)
    fig2.semilogx(limmaxx2, limmaxy2, color='black', linewidth=largeur)
    fig2.semilogx(limminx, limminy, color='black', linewidth=largeur)
    fig2.semilogx(limminx2, limminy2, color='black', linewidth=largeur)
    fig2.semilogx(limminpentex, limminpentey, color='black', linewidth=largeur)
    fig2.set_ylim((-55, 10))
    fig2.grid(color='grey')

    fig3.legend(Legende)
    fig3.semilogx(limmaxx, limmaxy, color='black', linewidth=largeur)
    fig3.semilogx(limmaxx2, limmaxy2, color='black', linewidth=largeur)
    fig3.semilogx(limminx, limminy, color='black', linewidth=largeur)
    fig3.semilogx(limminx2, limminy2, color='black', linewidth=largeur)
    fig3.semilogx(limminpentex, limminpentey, color='black', linewidth=largeur)
    fig3.set_ylim((-55, 10))
    fig3.grid(color='grey')

def GrpDelai(min, max, nb, Wn, A0, A1, rangemin, rangetop, prt):
    w = np.linspace(min, max, nb)

    Rdb = 20*np.log10(A0/A1)

    place = ((2*Wn)*nb)/max

    Legende = []

    figure = plt.figure()
    figure.suptitle('Delai de groupe')

    for x in range(rangemin, rangetop):
        b1, a1 = signal.butter(x, Wn, btype='low', analog=True)
        b2, a2 = signal.cheby1(x, Rdb, Wn, btype='low', analog=True)
        b3, a3 = signal.bessel(x, Wn, btype='low', analog=True)

        w1, mag1, ph1 = signal.bode((b1, a1), w)
        w2, mag2, ph2 = signal.bode((b2, a2), w)
        w3, mag3, ph3 = signal.bode((b3, a3), w)

        string = 'Ordre ' + str(x)
        fig1 = plt.subplot(3, 1, 1)
        fig2 = plt.subplot(3, 1, 2)
        fig3 = plt.subplot(3, 1, 3)

        ph1 = -np.diff(ph1)/np.diff(w1)
        ph2 = -np.diff(ph2)/np.diff(w2)
        ph3 = -np.diff(ph3)/np.diff(w3)

        fig1.plot(w1[:-1], ph1)
        fig2.plot(w2[:-1], ph2)
        fig3.plot(w3[:-1], ph3)

        fig1.set_title('Butter')
        fig2.set_title('Cheby')
        fig3.set_title('Bessel')

        Legende = np.append(Legende, string)

        if prt:
            print('\nValeur de phase à ', Wn, ' (2*fc) de l\'', string, '\nButter : ', ph1[int(place)], '\nCheby : ', ph2[int(place)], '\nBessel : ', ph3[int(place)])

    fig1.legend(Legende)
    fig1.grid(color='grey')
    fig2.legend(Legende)
    fig2.grid(color='grey')
    fig3.legend(Legende)
    fig3.grid(color='grey')

def Impulse(Wn, A0, A1, rangemin, rangetop):
    Rdb = 20 * np.log10(A0 / A1)

    Legende = []

    figure = plt.figure()
    figure.suptitle('Reponse impulsionnel')

    for x in range(rangemin, rangetop):
        b1, a1 = signal.butter(x, Wn, btype='low', analog=True)
        b2, a2 = signal.cheby1(x, Rdb, Wn, btype='low', analog=True)
        b3, a3 = signal.bessel(x, Wn, btype='low', analog=True)

        tout1, yout1 = signal.impulse((b1, a1))
        tout2, yout2 = signal.impulse((b2, a2))
        tout3, yout3 = signal.impulse((b3, a3))

        string = 'Ordre ' + str(x)

        fig1 = plt.subplot(3, 1, 1)
        fig2 = plt.subplot(3, 1, 2)
        fig3 = plt.subplot(3, 1, 3)


        fig1.plot(tout1, yout1)
        fig2.plot(tout2, yout2)
        fig3.plot(tout3, yout3)

        fig1.set_title('Butter')
        fig2.set_title('Cheby')
        fig3.set_title('Bessel')

        Legende = np.append(Legende, string)

    fig1.legend(Legende)
    fig1.grid(color='grey')
    fig2.legend(Legende)
    fig2.grid(color='grey')
    fig3.legend(Legende)
    fig3.grid(color='grey')

def Bessel(N, fc, K, prt, fig):
    Wc = 2*pi*fc
    b, a = signal.bessel(N, Wc, analog=True, btype='low')

    p = np.roots(a)

    a1 = np.poly([p[0], p[1]])
    b1 = np.array([a1[2]*K])
    a2 = np.poly([p[3], p[4]])
    b2 = np.array([a2[2]*K])
    a3 = np.poly([p[2]])
    b3 = np.array([a3[1]*K])

    if prt:
        print('b = ', b, '\na = ', a, '\n')
        print('Pole : ', p, '\n')
        print('b1 = ', b1, '\na1 = ', a1, '\nb2 = ', b2, '\na2 = ', a2, '\nb3 = ', b3, '\na3 = ', a3, '\n')
    if fig:
        hp.bodeplot(b, a, 'Modified Bode of')



    return a1, b1, a2, b2, a3, b3

def MFB_R2(K, R1):
    R2 = K*R1

    return R2

def MFB_R3C1(a, R2, C2):

    R3C1 = 1/((a[2])*R2*C2)

    return R3C1

def MFB_R3(R1, R2, C2, a):
    R3 = 1/((a[1]*C2)-(1/R2)-(1/R1))
    return R3

def MFB_C1(R3C1, R3):
    C1 = R3C1 / R3

    return C1

def MFB_Wc_Q(C1, C2, R1, R2, R3, a):
    Q = np.sqrt(C2/C1)*(1/((np.sqrt(R2*R3)/R1)+(np.sqrt(R3/R2))+(np.sqrt(R2/R3))))
    Wc_Q = np.sqrt(a[2])/Q

    return Q, Wc_Q

def AfficherTout(C1, C2, R1, R2, R3, Q, SR1, SR2, SR3, SC1, SC2, fig):
    print(fig, '\n\tR1\t\t\tR2\t\t\tR3\t\t\tC1\t\t\tC2\t\t\tQ\t\t\tSR1\t\t\tSR2\t\t\tSR3\t\t\tSC1\t\t\tSC2\n')
    for x in range(0, len(R1)):
        print(R1[x], '\t', R2[x], '\t', R3[x], '\t', C1[x], '\t', C2[x], '\t', Q[x], '\t', SR1[x], '\t', SR2[x], '\t', SR3[x], '\t', SC1[x], '\t', SC2[x], '\n')

def AfficherToutRC(C, R, fig):
    print(fig, '\n\tR\t\tC\n')
    for x in range(0, len(C)):
        print(R[x], '\t', C[x], '\n')

def CalculDeTout(minR1, maxR1, K, a, prt, fig):
    Valeur = [1e-9, 1.2e-9, 1.8e-9, 3.9e-9, 4.7e-9, 6.8e-9, 10e-9, 22e-9, 33e-9, 39e-9, 47e-9, 68e-9, 82e-9]

    TC1 = []
    TC2 = []
    TR1 = []
    TR2 = []
    TR3 = []
    TQ = []
    TSR1 = []
    TSR2 = []
    TSR3 = []
    TSC1 = []
    TSC2 = []

    for x in range(0, 13):
        C2 = Valeur[x]
        for R1 in range(minR1, maxR1, 1000):
            # Calcul de R2
            R2 = MFB_R2(K, R1)

            # calcul R3 C1
            R3C1 = MFB_R3C1(a, R2, C2)

            # calcul de R3
            R3 = MFB_R3(R1, R2, C2, a)

            # calcul de C1
            C1 = MFB_C1(R3C1, R3)

            if(R2 > 0 and C1 > 0 and R3 > 0):
                for z in range(0, 13):
                    if ((C1 + C1*0.005) > Valeur[z] and (C1 - C1*0.005) < Valeur[z]):
                        Q, temp = MFB_Wc_Q(C1, C2, R1, R2, R3, a)
                        if ((temp < a[1]+(a[1]*0.005)) and ((temp > a[1]-(a[1]*0.005)))):
                            TC1 = np.append(TC1, C1)
                            TR3 = np.append(TR3, R3)
                            TC2 = np.append(TC2, C2)
                            TR1 = np.append(TR1, R1)
                            TR2 = np.append(TR2, R2)
                            TQ = np.append(TQ, Q)
    for x in range(0, len(TC1)):
        denom = ((np.sqrt(TR2[x]*TR3[x]) / TR1[x]) + (np.sqrt(TR1[x]/TR2[x])) + (np.sqrt(TR2[x]/TR3[x])))
        SR1 = (np.sqrt(TR2[x]*TR3[x])) / (TR1[x]*denom)
        SR2 = -1*(TR2[x]*((TR3[x]/(2*TR1[x]*np.sqrt(TR2[x]*TR3[x]))) + (TR3[x]/(2*((TR2[x])**2)*np.sqrt(TR3[x]/TR2[x]))) + (1/(2*TR3[x]*np.sqrt(TR2[x]/TR3[x])))))/(denom)
        SR3 = -1 * (TR3[x] * ((TR2[x] / (2 * TR1[x] * np.sqrt(TR2[x] * TR3[x]))) + (TR2[x] / (2 * ((TR3[x]) ** 2) * np.sqrt(TR3[x] / TR3[x]))) + (1 / (2 * TR2[x] * np.sqrt(TR3[x] / TR2[x]))))) / (denom)
        SC1 = -1/2
        SC2 = 1/2

        TSR1 = np.append(TSR1, SR1)
        TSR2 = np.append(TSR2, SR2)
        TSR3 = np.append(TSR3, SR3)
        TSC1 = np.append(TSC1, SC1)
        TSC2 = np.append(TSC2, SC2)

    if prt:
        AfficherTout(TC1, TC2, TR1, TR2, TR3, TQ, TSR1, TSR2, TSR3, TSC1, TSC2, fig)

def CalculDeToutRC(a, prt, fig):
    Valeur = [1e-9, 1.2e-9, 1.8e-9, 3.9e-9, 4.7e-9, 6.8e-9, 10e-9, 22e-9, 33e-9, 39e-9, 47e-9, 68e-9, 82e-9]

    TC = []
    TR = []

    for x in range(0, 13):
        C = Valeur[x]

        R = 1/(a[1]*(C))
        TC = np.append(TC, C)
        TR = np.append(TR, R)

    if prt:
        AfficherToutRC(TC, TR, fig)

def GraphiqueDeValeurs(file, prt):

    File = np.loadtxt(file)

    TX = []
    for i in range(1, len(File)):
        if(File[i-1] != File[i]):
            X = File[i]-File[i-1]
            X = X/1000000
            TX = np.append(TX, X)


    t = np.arange(0, TX[-1], 0.05)
    hist, bin_edges = np.histogram(TX, bins=t)
    hist = np.insert(hist, 0, 0)

    if prt:
        plt.figure()
        plt.step(t, hist)

def main():
    #min X \ Max X \ nombre de points \ A0 \ A1 \ A2 \ Y minimum \ Y maximum \ Ordre minimum \ Ordre maximum
    #Bode(0.1, 10, 10000, 1, 1, 0.56, 1, 10, True)
    #GrpDelai(0, 2.5, 10000, 1, 10, 7, 1, 10, True)
    #Impulse(1, 10, 7, 1, 10)

    file = 'Data.txt'
    w = 15000
    K = 1

    #Bessel Mettre vrai pour voir les fonction de transfert, mettre 2eme vrai pour graphique
    a1, b1, a2, b2, a3, b3 = Bessel(5, w, K, False, False)

    #Test des valeurs mettre vrai pour voir les resultats
    CalculDeTout(1, 500000, K, a1, True, 'Filtre 1')

    # Test des valeurs mettre vrai pour voir les resultats
    CalculDeTout(1, 500000, K, a2, True, 'Filtre 2')

    # Test des valeurs mettre vrai pour voir les resultats
    CalculDeToutRC(a3, True, 'Filtre RC')

    GraphiqueDeValeurs(file, False)

    plt.show()

#Valeur = [1e-9, 1.2e-9, 1.8e-9, 3.9e-9, 4.7e-9, 6.8e-9, 10e-9, 22e-9, 33e-9, 39e-9, 47e-9, 68e-9, 82e-9]

if __name__ ==  "__main__":
    main()

