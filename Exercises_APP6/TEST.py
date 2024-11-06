import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
pi = np.pi

num = [1, 1, 0]
den = [1, 2, 3]

B, A = signal.butter(1, [2*pi*980, 2*pi*1020], btype='bandpass', analog=True)

w = np.arange(0, 10000)
W1, H1 = signal.freqs(B, A, worN=w)

zero = np.roots(B)
pole = np.roots(A)

#puizero = np.sqrt((np.real(zero)*np.exp(2))+(np.imag(zero)*np.exp(2)))
#puipole = np.sqrt((np.real(pole)*np.exp(2))+(np.imag(pole)*np.exp(2)))



plt.figure()
plt.scatter(np.real(zero), np.imag(zero), marker='o')
plt.scatter(np.real(pole), np.imag(pole), marker='x')
plt.axis([-100, 100, -100, 100])
plt.figure()

plt.plot(W1, H1)

plt.show()

print(A)
print(B)