from scipy import ndimage, datasets
import numpy as np
import matplotlib.pyplot as plt

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
freq_db = 20*np.log10(abs(Y) + 1e-10)

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
freq_db = 20*np.log10(abs(Y) + 1e-10)

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
freq_db = 20*np.log10(abs(Y) + 1e-10)

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

# Exercitiul 2

freq_cutoff = 40
X = datasets.face(gray=True)
# Normalizare val pix intre 0 si 1
X = (X - X.min()) / (X.max() - X.min())
Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y) + 1e-10)
Y_cutoff = Y.copy()
Y_cutoff[freq_db < freq_cutoff] = 0 # In exemplul din lab am eliminat magnitudini mari, aici elimin magnitudini mici pentru a nu se duce informatia, ci doar calitatea pozei vine redusa
X_cutoff = np.fft.ifft2(Y_cutoff).real

snr_db = 10*np.log10(np.mean(X ** 2) / np.mean((X - X_cutoff) ** 2))
print(f"SNR obtinut: {snr_db:.2f} dB") 
fig, ax = plt.subplots(1, 2, figsize=(12, 12))

ax[0].imshow(X, cmap='gray')
ax[1].imshow(X_cutoff, cmap='gray')
ax[0].set_title("Imagine originala")
ax[1].set_title(f"Imagine comprimata\nSNR = {snr_db:.2f} dB")
plt.tight_layout()
plt.savefig("./Lab7/ex2.pdf", format="pdf")
plt.show()

# Exerctitiul 3 

pixel_noise = 200
X = datasets.face(gray=True)
noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise
snr_db_inainte = 10*np.log10(np.mean(X ** 2) / np.mean((noise) ** 2))
print("SNR inainte: ", snr_db_inainte)
Y_noisy = np.fft.fft2(X_noisy)
Y_shifted = np.fft.fftshift(Y_noisy)

rows, cols = X.shape
middle_row, middle_col = rows // 2, cols // 2
radius = 80  # mai mic => filtrarea este mai puternica
mask = np.zeros_like(X)
Y, X_grid = np.ogrid[:rows, :cols]
mask = (Y - middle_row) ** 2 + (X_grid - middle_col) ** 2 <= radius ** 2

Y_filtered = Y_shifted * mask
X_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(Y_filtered)))

snr_db_dupa = 10 * np.log10(np.mean(X**2) / np.mean((X - X_filtered)**2))
print("SNR dupa filtrare:", snr_db_dupa)

fig, ax = plt.subplots(3, 1, figsize=(10, 18))  # 3 rânduri, 1 coloană
ax[0].imshow(X, cmap='gray')
ax[0].set_title("Imagine originala", fontsize=12)
ax[0].axis('off')

ax[1].imshow(X_noisy, cmap='gray')
ax[1].set_title(f"Imagine zgomotoasa\nnoise = {pixel_noise}, SNR inainte = {snr_db_inainte:.2f} dB", fontsize=12)
ax[1].axis('off')

ax[2].imshow(X_filtered, cmap='gray')
ax[2].set_title(f"Imagine filtrata\nSNR dupa = {snr_db_dupa:.2f} dB", fontsize=12)
ax[2].axis('off')

plt.subplots_adjust(hspace=1)
plt.savefig("./Lab7/ex3.pdf", format="pdf")
plt.show()
