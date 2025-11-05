import numpy as np, matplotlib.pyplot as plt, time, matplotlib.image as img, math

# Ex 1
def DFT(x):
    N = len(x)
    X = []
    for k in range(N):
        suma = 0
        for n in range(N):
            suma += x[n] * np.exp(-2j * np.pi * k * n / N)
        X.append(suma)
    return np.array(X)

def FFT(x):
    n = len(x)
    if n <= 1:
        return x
    if n % 2 != 0:
        return [sum(x[m] * np.exp(-2j * np.pi * k * m / n) for m in range(n)) for k in range(n)]
    par = FFT(x[0::2])
    imp = FFT(x[1::2])
    T = [np.exp(-2j * np.pi * k / n) * imp[k] for k in range(n // 2)]
    return [par[k] + T[k] for k in range(n // 2)] + [par[k] - T[k] for k in range(n // 2)]

frecventa_de_esantionare = 600
durata = 1
t = np.linspace(0, durata, int(durata * frecventa_de_esantionare))

f1, f2, f3 = 5, 20, 200
A1, A2, A3 = 1.0, 2.0, 0.75

x = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t) + A3 * np.sin(2 * np.pi * f3 * t)


N_values = [128, 256, 512, 1024, 2048, 4096, 8192]

timp_DFT = []
timp_FFT = []
timp_numpy = []

for N in N_values:
    semnal = x[:N]  

    start = time.time()
    DFT(semnal)
    timp_DFT.append(time.time() - start)

    start = time.time()
    FFT(semnal)
    timp_FFT.append(time.time() - start)

    start = time.time()
    np.fft.fft(semnal)
    timp_numpy.append(time.time() - start)

plt.figure(figsize=(10, 5))
plt.plot(N_values, timp_DFT, "o-", label="DFT")
plt.plot(N_values, timp_FFT, "o-", label="FFT Manual")
plt.plot(N_values, timp_numpy, "o-", label="FFT Numpy")
plt.yscale('log')  
plt.xlabel("Dimensiunea vectorului N")
plt.ylabel("Timp de execuție log scale")
plt.title("Compararea timpiilor de execuție: DFT vs FFT")
plt.grid(True)
plt.legend()
plt.savefig(fname="./Lab4/dft_vs_fft_manual_vs_fft_numpy.pdf", format="pdf")
plt.show()

# Ex 2

frecventa = 20
amplitudine = 1
faza = 0
durata = 3
frecventa_de_esantionare = 333
t = np.linspace(0, durata, int(durata * frecventa_de_esantionare))
semnal_sinus_continuu = amplitudine * np.sin(2 * np.pi * frecventa * t + faza)

f_s = 4
t_esantionat = np.arange(0, durata, 1/f_s)

semnal_sinus = amplitudine * np.sin(2 * np.pi * frecventa * t_esantionat + faza)

frec_1 = 4
frec_2 = 10
semnal_sinus_1 = amplitudine * np.sin(2 * np.pi * frec_1 * t_esantionat + faza)
semnal_sinus_2 = amplitudine * np.sin(2 * np.pi * frec_2 * t_esantionat + faza)
semnal_sinus_1_continuu = amplitudine * np.sin(2 * np.pi * frec_1 * t + faza)
semnal_sinus_2_continuu = amplitudine * np.sin(2 * np.pi * frec_2 * t + faza)

fig, ax = plt.subplots(6, 1, figsize=(12, 12))

ax[0].plot(t, semnal_sinus_continuu, label = "Continuu")
ax[0].scatter(t_esantionat, semnal_sinus, label = "Esantionat", color="green")
ax[0].set_title("Semnal original")
ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[0].grid(True)

ax[1].plot(t, semnal_sinus_1_continuu, label = "Continuu 1")
ax[1].scatter(t_esantionat, semnal_sinus_1, label = "Esantionat 1", color="green")
ax[1].set_title("Semnal 1")
ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[1].grid(True)

ax[2].plot(t, semnal_sinus_2_continuu, label = "Continuu 2")
ax[2].scatter(t_esantionat, semnal_sinus_2, label = "Esantionat 2", color="green")
ax[2].set_title("Semnal 2")
ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[2].grid(True)

ax[3].plot(t, semnal_sinus_continuu, label = "Continuu")
ax[3].scatter(t_esantionat, semnal_sinus, label = "Esantionat", color="green")
ax[3].plot(t, semnal_sinus_1_continuu, label = "Continuu 1")
ax[3].scatter(t_esantionat, semnal_sinus_1, label = "Esantionat 1", color="red")
ax[3].grid(True)
ax[3].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[3].set_title("Semnal original vs semnal 1")

ax[4].plot(t, semnal_sinus_continuu, label = "Continuu")
ax[4].scatter(t_esantionat, semnal_sinus, label = "Esantionat", color="green")
ax[4].plot(t, semnal_sinus_2_continuu, label = "Continuu 2")
ax[4].scatter(t_esantionat, semnal_sinus_2, label = "Esantionat 2", color="red")
ax[4].grid(True)
ax[4].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[4].set_title("Semnal original vs semnal 2")

ax[5].plot(t, semnal_sinus_continuu, label = "Continuu")
ax[5].scatter(t_esantionat, semnal_sinus, label = "Esantionat", color="green")
ax[5].plot(t, semnal_sinus_1_continuu, label = "Continuu 1")
ax[5].scatter(t_esantionat, semnal_sinus_1, label = "Esantionat 1", color="red")
ax[5].plot(t, semnal_sinus_2_continuu, label = "Continuu 2")
ax[5].scatter(t_esantionat, semnal_sinus_2, label = "Esantionat 2", color="black")
ax[5].grid(True)
ax[5].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[5].set_title("Semnal original vs semnal 1 vs semnal 2")
fig.tight_layout(pad=5.0)
plt.savefig(fname="./Lab4/fenomenul_de_aliere.pdf", format="pdf")
plt.show()

# Ex 3

f_s_mare = 333        
t_mare = np.linspace(0, durata, int(durata * f_s_mare))
semnal_continuu = amplitudine * np.sin(2 * np.pi * frecventa * t_mare + faza)

f_s_mic = 15          
t_mic = np.arange(0, durata, 1/f_s_mic)
semnal_esantionat_alias = amplitudine * np.sin(2 * np.pi * frecventa * t_mic + faza)

t_corect = np.arange(0, durata, 1/f_s_mare)
semnal_esantionat_corect = amplitudine * np.sin(2 * np.pi * frecventa * t_corect + faza)

frec_1 = 4
frec_2 = 10
semnal_1 = np.sin(2 * np.pi * frec_1 * t_mare)
semnal_2 = np.sin(2 * np.pi * frec_2 * t_mare)

fig, ax = plt.subplots(3, 1, figsize=(12, 10))

ax[0].plot(t_mare, semnal_continuu, label="Semnal continuu", color="blue")
ax[0].scatter(t_corect, semnal_esantionat_corect, label="Esantionat", color="red")
ax[0].set_title("Fara aliasing (f_s > f_Nyquist)")
ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[0].grid(True)

ax[1].plot(t_mare, semnal_continuu, label="Semnal continuu", color="blue")
ax[1].scatter(t_mic, semnal_esantionat_alias, label="Esantionat", color="orange")
ax[1].set_title("Cu aliasing (f_s < f_Nyquist)")
ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[1].grid(True)

ax[2].plot(t_mare, semnal_1, label="Semnal 1", color="red")
ax[2].plot(t_mare, semnal_2, label="Semnal 2", color="blue")
ax[2].plot(t_mare, semnal_continuu, label="Semnal 3", color="green")
ax[2].scatter(t_corect, np.sin(2 * np.pi * frec_1 * t_corect), color="red", label="Esantionare 1")
ax[2].scatter(t_corect, np.sin(2 * np.pi * frec_2 * t_corect), color="blue", label="Esantionare 2")
ax[2].scatter(t_corect, np.sin(2 * np.pi * frecventa * t_corect),  color="green", label="Esantionare 3")
ax[2].set_title("Comparare semnal 1 vs semnal 2 vs semnal 3")
ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax[2].grid(True)

fig.tight_layout(pad=4)
plt.savefig("./Lab4/lipsa_fenomenului_de_aliere.pdf", format="pdf")
plt.show()

# Ex 4

print(f"Pentru contrabas (40Hz - 200Hz) fs > 2 * fmax, fmax = 200 => minimul Nyquist este: 400Hz")

# Ex 5

im = img.imread("./Lab4/spectograma_audacity.png")
plt.imshow(im)
plt.title("Spectrograma inregistrarii vocalelor cu Audacity")
plt.axis('off')
plt.show()

# Ex 7

P_semnal = 90
Snr_db = 80
Snr = 10 ** (Snr_db / 10)
P_zgomot = P_semnal / Snr
P_zgomot_db = 10 * np.log10(P_zgomot)

print(f"Puterea zgomotului este: {P_zgomot_db}dB")