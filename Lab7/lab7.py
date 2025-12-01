from scipy import ndimage, datasets
import numpy as np
import matplotlib.pyplot as plt

# X = datasets.face(gray=True)
# plt.imshow(X, cmap=plt.cm.gray)
# plt.show()

# Exercitiul 1
# a)

n1 = 128
n2 = 128

X = np.zeros((n1, n2))

for i in range (n1):
    for j in range (n2):
        X[i][j] = np.sin(2 * np.pi * i + 3 * np.pi * j)

Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y))

fig, ax = plt.subplots(1, 2, figsize=(12, 12))

spectru_X = ax[0].imshow(X, cmap='hot')
spectru_frec_Y = ax[1].imshow(freq_db, cmap='hot')
ax[0].set_title("a) Semnal X(i, j)")
ax[1].set_title("a) Spectru Fourier |Y| dB")
plt.colorbar(spectru_X, ax=ax[0])
plt.colorbar(spectru_frec_Y, ax=ax[1])
plt.tight_layout()
plt.savefig("./Lab7/ex1_a.pdf", format="pdf")
plt.show()


# b)

n1 = 128
n2 = 128

X = np.zeros((n1, n2))

for i in range (n1):
    for j in range (n2):
        X[i][j] = np.sin(4 * np.pi * i) + np.cos(6 * np.pi * j)

Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y) + 1e-10)

fig, ax = plt.subplots(1, 2, figsize=(12, 12))

spectru_X = ax[0].imshow(X, cmap='hot')
spectru_frec_Y = ax[1].imshow(freq_db, cmap='hot')

ax[0].set_title("b) Semnal X(i, j)")
ax[1].set_title("b) Spectru Fourier |Y| dB")
plt.colorbar(spectru_X, ax=ax[0])
plt.colorbar(spectru_frec_Y, ax=ax[1])
plt.tight_layout()
plt.savefig("./Lab7/ex1_b.pdf", format="pdf")
plt.show()


# c)

N = n1
Y = np.zeros((N, N), dtype=complex)
Y[0][5] = Y[0][N-5] = 1

X = np.fft.irfft2(Y)
freq_db = 20*np.log10(abs(Y))

fig, ax = plt.subplots(1, 2, figsize=(12, 12))

spectru_X = ax[0].imshow(X, cmap='hot')
spectru_frec_Y = ax[1].imshow(freq_db, cmap='hot')
ax[0].set_title("c) Semnal X generat prin IRFFT")
ax[1].set_title("c) Spectru Y cu componente in (0,5) si (0,N-5)")
plt.colorbar(spectru_X, ax=ax[0])
plt.colorbar(spectru_frec_Y, ax=ax[1])
plt.tight_layout()
plt.savefig("./Lab7/ex1_c.pdf", format="pdf")
plt.show()


# d)

N = n1
Y = np.zeros((N, N), dtype=complex)
Y[5][0] = Y[-5][0] = 1

X = np.fft.irfft2(Y)
freq_db = 20*np.log10(abs(Y))

fig, ax = plt.subplots(1, 2, figsize=(12, 12))

spectru_X = ax[0].imshow(X, cmap='hot')
spectru_frec_Y = ax[1].imshow(freq_db, cmap='hot')
ax[0].set_title("d) Semnal X - frecvente verticale")
ax[1].set_title("d) Spectru Y cu componente in (5,0) si (-5,0)")
plt.colorbar(spectru_X, ax=ax[0])
plt.colorbar(spectru_frec_Y, ax=ax[1])
plt.tight_layout()
plt.savefig("./Lab7/ex1_d.pdf", format="pdf")
plt.show()


# e)

N = n1
Y = np.zeros((N, N), dtype=complex)
Y[5][5] = Y[-5][-5] = 1

X = np.fft.irfft2(Y)
freq_db = 20*np.log10(abs(Y))

fig, ax = plt.subplots(1, 2, figsize=(12, 12))

spectru_X = ax[0].imshow(X, cmap='hot')
spectru_frec_Y = ax[1].imshow(freq_db, cmap='hot')
ax[0].set_title("e) Semnal X - frecvente diagonale")
ax[1].set_title("e) Spectru Y cu componente in (5,5) si (-5,-5)")
plt.colorbar(spectru_X, ax=ax[0])
plt.colorbar(spectru_frec_Y, ax=ax[1])
plt.tight_layout()
plt.savefig("./Lab7/ex1_e.pdf", format="pdf")
plt.show()