"""
Fichier de fonctions utiles pour la problÃ©matique de l'APP6 (S2)
(c) JB Michaud, Sylvain Nicolay UniversitÃ© de Sherbrooke
v 1.0 Hiver 2023
v 1.1 - CorrigÃ© un cas limite dans simplifytf
      - UtilisÃ© des fonctions et une logique plus intuitive Ã  lire dans simplifytf
      - ImplÃ©mentÃ© un workaround pour np.unwrap pour d'anciennes versions de numpy
      - AjustÃ© adÃ©quatement l'utilisation de period= dans np.unwrap
      - GÃ©nÃ©ralisÃ© le code correctdelaybug au cas oÃ¹, mais cette fonction ne devrait plus servir, a Ã©tÃ© mise en commentaire

Fonctions de visualisation
pzmap: affiche les pÃ´les et les zÃ©ros dÃ©jÃ  calculÃ©s
bode1: affiche un lieu de bode dÃ©jÃ  calculÃ©
bodeplot: calcule et affiche le lieu de bode d'une FT
grpdel1: affiche le dÃ©lai de groupe dÃ©jÃ  calculÃ©
timeplt1: affiche une rÃ©ponse temporelle dÃ©jÃ  calculÃ©e
timepltmutlti1: affiche plusieurs rÃ©ponses temporelles dÃ©jÃ  calculÃ©es Ã  diffÃ©rentes frÃ©quences
timeplotmulti2: affiche plusieurs rÃ©ponses temporelles dÃ©jÃ  calculÃ©es pour diffÃ©rents systÃ¨mes

Fonctions de manipulation de FT
paratf: calcule la FT simpifiÃ©e Ã©quivalente Ã  2 FT en parallÃ¨le
seriestf: calcule la FT simplifiÃ©e Ã©quivalente Ã  2 FT en sÃ©rie (i.e. en cascade)
simplifytf: simplifie les pÃ´les et les zÃ©ros d'une FT, et arrondis les parties rÃ©elles et imaginaires Ã  l'entier lorsque pertinent
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


###############################################################################
def pzmap1(z, p, title):
    """
    Affiche les pÃ´les et les zÃ©ros sur le plan complexe

    :param z: liste des zÃ©ros
    :param p: liste des pÃ´les
    :param title: titre du graphique
    :return: handles des Ã©lÃ©ments graphiques gÃ©nÃ©rÃ©s
    """

    if len(p) == 0:     # safety check cas limite aucun pÃ´le
        return
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if len(z):
        ax.plot(np.real(z), np.imag(z), 'o', fillstyle='none', label='ZÃ©ros')  # affichage des zeros avec marqueurs 'o' ouverts
    ax.plot(np.real(p), np.imag(p), 'x', fillstyle='none', label='PÃ´les')  # affichage des poles avec des marqueurs 'x'
    fig.suptitle('PÃ´le/zÃ©ros de ' + title)
    ax.set_xlabel("Partie rÃ©elle ($Re(s)$)")
    ax.set_ylabel("Partie imaginaire ($Im(s)$)")
    # Recherche des min et max pour ajuster les axes x et y du graphique
    # longue histoire courte, concatÃ¨ne toutes les racines dans 1 seule liste, puis rÃ©serve une marge de chaque cÃ´tÃ©
    rootslist = []
    if len(z):
        rootslist.append(z)
    rootslist.append(p)
    rootslist = [item for sublist in rootslist for item in sublist]
    ax.set_xlim(np.amin(np.real(rootslist)) - .5, np.amax(np.real(rootslist)) + .5)
    ax.set_ylim(np.amin(np.imag(rootslist)) - .5, np.amax(np.imag(rootslist)) + .5)
    return fig, ax


###############################################################################
def bode1(w, mag, phlin, title):
    """
    Affiche le lieu un lieu de bode dÃ©jÃ  calculÃ©

    :param w: vecteur des frÃ©quences du lieu de bode
    :param mag: vecteur des amplitudes, assumÃ©es en dB, doit Ãªtre de mÃªme longueur que w
    :param phlin: vecteur des phases, assumÃ©es en degrÃ©s, doit Ãªtre de mÃªme longueur que w
    :param title: titre du graphique
    :return: handles des Ã©lÃ©ments graphiques gÃ©nÃ©rÃ©s
    """

    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    fig.suptitle(title + ' Frequency Response')

    ax[0].plot(w, mag)
    ax[0].set_xscale('log')
    ax[0].grid(visible=None, which='both', axis='both', linewidth=0.5)
    # fixe les limites du graphiques en gardant une marge minimale
    ax[0].set_xlim(10 ** (np.floor(np.log10(np.amin(w))) - 0.1), 10 ** (np.ceil(np.log10(np.amax(w))) + .1))
    ax[0].set_ylim(20 * (np.floor(np.amin(mag) / 20 - 0.1)), 20 * (np.ceil(np.amax(mag) / 20 + .1)))
    ax[0].set_ylabel('Amplitude [dB]')

    ax[1].plot(w, phlin)
    ax[1].set_xscale('log')
    ax[1].grid(visible=None, which='both', axis='both', linewidth=0.5)
    ax[1].set_xlabel('Frequency [rad/s]')
    ax[1].set_ylabel('Phase [deg]')
    # fixe les limites du graphiques en gardant une marge minimale
    ax[1].set_xlim(10 ** (np.floor(np.log10(np.amin(w))) - 0.1), 10 ** (np.ceil(np.log10(np.amax(w))) + .1))
    ax[1].set_ylim(20 * (np.floor(np.amin(phlin) / 20) - 1), 20 * (np.floor(np.amax(phlin) / 20) + 2))
    return fig, ax


###############################################################################
def bodeplot(b, a, title):
    """
    Calcule et affiche le lieu de bode d'une FT

    :param b: numÃ©rateur de la FT sous forme np.poly
    :param a: dÃ©nominateur de la FT sous forme np.poly
    :param title: titre du graphique
    :return: amplitude (dB) et phase (radians) calculÃ©s aux frÃ©quences du vecteur w (rad/s) et les handles des Ã©lÃ©ments
        graphiques gÃ©nÃ©rÃ©s
    """

    w, h = signal.freqs(b, a, 5000)  # calcul la rÃ©ponse en frÃ©quence du filtre (H(jw)), frÃ©quence donnÃ©e en rad/sec
    mag = 20 * np.log10(np.abs(h))
    ph = np.unwrap(np.angle(h), period=np.pi) if np.__version__ > '1.21' else \
        np.unwrap(2*np.angle(h))/2  # calcul du dÃ©phasage en radians
    phlin = np.rad2deg(ph)  # dÃ©phasage en degrÃ©s
    fig, ax = bode1(w, mag, phlin, title)
    return mag, ph, w, fig, ax

###############################################################################
def grpdel1(w, delay, title):
    """
    Affiche le dÃ©lai de groupe dÃ©jÃ  calculÃ©

    :param w: vecteur des frÃ©quences, assumÃ©es en rad/s
    :param delay: vecteur des dÃ©lais de groupe, assumÃ© en secondes, doit Ãªtre de longueur len(w)-1
    :param title: titre du graphique
    :return: handles des Ã©lÃ©ments graphiques gÃ©nÃ©rÃ©s
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle('Group Delay ' + title)
    ax.plot(w[:len(w) - 1], delay)
    ax.set_xscale('log')
    ax.set_xlabel('FrÃ©quence [rad/s]')
    ax.set_ylabel('DÃ©lai de groupe [s]')
    ax.grid(which='both', axis='both')
    ax.set_xlim(10 ** (np.floor(np.log10(np.amin(w))) - 0.1), 10 ** (np.ceil(np.log10(np.amax(w))) + .1))
    return fig, ax


###############################################################################
def timeplt1(t, u, tout, yout, title):
    """
    Affiche le rÃ©sultat de  la simulation temporelle d'un systÃ¨me

    :param t: vecteur de temps en entrÃ©e de lsim, assumÃ© en secondes
    :param u: vecteur d'entrÃ©e du systÃ¨me, doit Ãªtre de mÃªme longueur que t
    :param tout: vecteur de temps en sortie de lsim, assumÃ© en secondes
    :param yout: vecteur de rÃ©ponse du systÃ¨me, doit Ãªtre de mÃªme longueur que tout
    :return: handles des Ã©lÃ©ments graphiques gÃ©nÃ©rÃ©s
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle('RÃ©ponse temporelle '+title)
    ax.plot(t, u, 'r', alpha=0.5, linewidth=1, label='input')
    ax.plot(tout, yout, 'k', linewidth=1.5, label='output')
    ax.legend(loc='best', shadow=True, framealpha=1)
    ax.grid(alpha=0.3)
    ax.set_xlabel('t (s)')
    return fig, ax


###############################################################################
def timepltmulti1(t, u, w, tout, yout, title):
    """
    Affiche la rÃ©ponse d'un mÃªme systÃ¨me Ã  N entrÃ©es assumÃ©es sinusoÃ®dales, chacune dans un subplot

    :param t: vecteur de temps fourni Ã  lsim, assumÃ© en secondes
    :param u: liste de N vecteurs d'entrÃ©e, doivent tous Ãªtre de mpeme longueur que t
    :param w: liste de la frÃ©quence des N sinusoÃ®des
    :param tout: vecteur de temps en sortie de lsim, assumÃ© en secondes
    :param yout: liste de N vecteurs de sortie de lsim, doivent tous Ãªtre de mÃªme longueur que tout
    :param title: titre du graphique
    :return: handles des Ã©lÃ©ments graphiques gÃ©nÃ©rÃ©s
    """

    fig, ax = plt.subplots(len(w), 1, figsize=(6, 6))
    fig.suptitle('RÃ©ponses temporelles de ' + title)
    for i in range(len(w)):
        ax[i].plot(t, u[i], 'r', alpha=0.5, linewidth=1, label=f'Input {w[i]} rad/s')
        ax[i].plot(tout[i], yout[i], 'k', linewidth=1.5, label=f'Output {w[i]} rad/s')
        ax[i].legend(loc='best', shadow=True, framealpha=1)
        ax[i].grid(alpha=0.3)
        if i == len(w) - 1:
            ax[i].set_xlabel('t (s)')
    return fig, ax


###############################################################################
def timepltmulti2(t, u, tout, yout, title, systems):
    """
    Affiche N rÃ©sultats de simulation temporelle de N systÃ¨mes dans N subplots

    :param t: vecteur de temps fourni Ã  lsim pour tous les systÃ¨mes, assumÃ© en secondes
    :param u: vecteur d'entrÃ©e de tous les systÃ¨mes, doit Ãªtre de mÃªme longueur que t
    :param tout: vecteur de temps en sortie de lsim pour tous les systÃ¨mes, assumÃ© en secondes
    :param yout: liste de N vecteurs de sortie de lsim pour chacun des systÃ¨mes, chaque vecteur de mÃªme longueur que tout
    :param title: titre du graphique
    :param systems: liste de N noms des systÃ¨mes simulÃ©s
    :return: handles des Ã©lÃ©ments graphiques gÃ©nÃ©rÃ©s
    """

    fig, ax = plt.subplots(len(yout), 1, figsize=(6, 6))
    fig.suptitle('RÃ©ponses temporelles de ' + title)
    for i in range(len(yout)):
        ax[i].plot(t, u, 'r', alpha=0.5, linewidth=1, label=f'Input {systems[i]}')
        ax[i].plot(tout, yout[i], 'k', linewidth=1.5, label=f'Output {systems[i]}')
        ax[i].legend(loc='best', shadow=True, framealpha=1)
        ax[i].grid(alpha=0.3)
        if i == len(yout) - 1:
            ax[i].set_xlabel('t (s)')
    return fig, ax


###############################################################################
def paratf(z1, p1, k1, z2, p2, k2):
    """
    Calcule la FT rÃ©sultante simplifiÃ©e des 2 FT fournies en argument en parallÃ¨le

    :param z1: zÃ©ros de la FT #1
    :param p1: pÃ´les de la FT #1
    :param k1: gain de la FT #1, tel que retournÃ© par signal.tf2zpk par exemple
    :param z2: idem FT #2
    :param p2:
    :param k2:
    :return: z, p, k simplifiÃ©s de la FT rÃ©sultante
    """
    b1, a1 = signal.zpk2tf(z1, p1, k1)
    b2, a2 = signal.zpk2tf(z2, p2, k2)
    # en parallÃ¨le, il faut mettre sur dÃ©nominateur commun et faire le produit croisÃ© au numÃ©rateur
    bleft = np.convolve(b1, a2) # calcule les 2 termes du numÃ©rateur
    bright = np.convolve(b2, a1)
    b = np.polyadd(bleft, bright)
    a = np.convolve(a1, a2)
    z, p, k = signal.tf2zpk(b, a)
    z, p, k = simplifytf(z, p, k)
    return z, p, k


###############################################################################
def seriestf(z1, p1, k1, z2, p2, k2):
    """
    Calcule la FT rÃ©sultante simplifiÃ©e des 2 FT fournies en argument en cascade

    :param z1: zÃ©ros de la FT #1
    :param p1: pÃ´les de la FT #1
    :param k1: gain de la FT #1, tel que retournÃ© par signal.tf2zpk par exemple
    :param z2: idem FT #2
    :param p2:
    :param k2:
    :return: z, p, k simplifiÃ©s de la FT rÃ©sultante
    """
    # Plus facile de travailler en polynÃ´me?
    b1, a1 = signal.zpk2tf(z1, p1, k1)
    b2, a2 = signal.zpk2tf(z2, p2, k2)
    # en sÃ©rie les numÃ©rateurs et dÃ©nominateurs sont simplement multipliÃ©s
    b = np.convolve(b1, b2)  # convolve est Ã©quivalant Ã  np.polymul()
    a = np.convolve(a1, a2)
    z, p, k = signal.tf2zpk(b, a)
    z, p, k = simplifytf(z, p, k)
    return z, p, k


###############################################################################
def simplifytf(z, p, k):
    """
    - simplifie les racines identiques entre les zÃ©ros et les pÃ´les
    - arrondit les parties rÃ©elles et imaginaires de tous les termes Ã  l'entier

    :param z: zÃ©ros de la FT Ã  simplifier
    :param p: pÃ´les de la FT Ã  simplifier
    :param k: k de la FT Ã  simplifier, tel que retournÃ©e par signal.tf2zpk par exemple
    :return: z, p, k simplifiÃ©s
    """

    tol = 1e-6  # tolÃ©rance utilisÃ©e pour dÃ©terminer si un pÃ´le et un zÃ©ro sont identiques ou un nombre est entier

    # cast tout en complexe d'abord couvre les cas oÃ¹ z ou p est complÃ¨tment rÃ©el pour les comparaisons qui suivent
    z = z.astype(complex)
    p = p.astype(complex)
    # algorithme de simplification des pÃ´les et des zÃ©ros
    # compliquÃ© pcq que la comparaison de nombres en points flottants ne se pythonify pas trÃ¨s bien
    # et que plusieurs cas limites (e.g. FT rÃ©sultantes avec aucune racine) nÃ©cessitent des contorsions
    while len(p) and len(z):  # tant que le numÃ©rateur et le dÃ©nominateur contiennent encore des racines
        match = False
        for i, zval in enumerate(z[:]):     # itÃ¨re sur les zÃ©ros
            for j, pval in enumerate(p[:]):     # itÃ¨re sur les pÃ´les
                if np.isclose(zval, pval, atol=tol, rtol=tol):  # si le zÃ©ro est identique au pÃ´le
                    p = np.delete(p, j)     # enlÃ¨ve ce zÃ©ro et ce pÃ´le
                    z = np.delete(z, i)
                    match = True    # poutine pour repartir la recherche en cas de match, (pour les cas limites)
                    break
            if match:
                break
        else:
            break
    # itÃ¨re sur les zÃ©ros, les pÃ´les et enfin le gain pour arrondir Ã  l'unitÃ© lorsque pertinent
    for i, zval in enumerate(z):
        if np.isclose(zval.real, np.round(zval.real), atol=tol, rtol=tol):   # teste si la valeur est identique Ã  un entier
            z[i] = complex(np.round(z[i].real), z[i].imag)
        if np.isclose(zval.imag, np.round(zval.imag), atol=tol, rtol=tol):
            z[i] = complex(z[i].real, np.round(z[i].imag))
    for i, pval in enumerate(p):
        if np.isclose(pval.real, np.round(pval.real), atol=tol, rtol=tol):   # teste si la valeur est identique Ã  un entier
            p[i] = complex(np.round(p[i].real), p[i].imag)
        if np.isclose(pval.imag, np.round(pval.imag), atol=tol, rtol=tol):
            p[i] = complex(p[i].real, np.round(p[i].imag))
    if np.isclose(k, np.round(k), atol=tol, rtol=tol):
        k = np.round(k)
    return z, p, k


def digitize(x, b, xmin=-1, xmax=1):
    """
    - numÃ©rise le signal x selon le nombre de bits b

    :param x: signal Ã  numÃ©riser
    :param b: nombre de bits
    :param xmin: valeur minimal de la plage de x
    :param xmax: valeur maximale de la plage de x

    :return: y, signal numÃ©risÃ©
    """
    bitbins = np.linspace(xmin, xmax, (2 ** b + 1))
    y = (bitbins[1] - bitbins[0]) * (np.digitize(x, bitbins) - 2 ** (b-1))
    return y