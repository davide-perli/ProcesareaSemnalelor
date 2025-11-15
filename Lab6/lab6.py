import scipy, matplotlib.pyplot as plt, numpy as np

# Ex 1

# sinc^2x = 1 (x=0) || sin^2x/x^2 (x!=0)

B = 1
frec_esantionare = 5000
t = np.linspace(-3, 3, frec_esantionare)
x = np.sinc(t * B) ** 2

fs_uri = [1.0, 1.5, 2.0, 4.0]
fix, axes = plt.subplots(2, 2)
for ax, fs in zip(axes.ravel(), fs_uri):
    Ts = 1 / fs
    t_s = np.arange(-3, 3 + Ts / 2, Ts)
    if 0 not in t_s:  
        t_s = np.concatenate((t_s, [0.0]))
        t_s.sort()
    x_s = np.sinc(t_s * B) ** 2

    x_recon = np.zeros_like(t)
    for n, xn in enumerate(x_s):
        x_recon += x_s[n] * np.sinc((t - t_s[n]) / Ts)
    ax.plot(t, x, color="blue", label="Semnal continuu")
    markerline, stemlines, baseline = ax.stem(t_s, x_s, linefmt=":", markerfmt="o", label="Esationare")
    stemlines.set_color("green")
    ax.plot(t, x_recon, color="pink", label="Reconstructia semnalului")
    ax.set_title(f"Fs = {fs} Hz")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

plt.tight_layout()
path = f"./Lab6/ex1_B={B}.pdf"
plt.savefig(path, format="pdf")
plt.show()

B = 2
frec_esantionare = 5000
t = np.linspace(-3, 3, frec_esantionare)
x = np.sinc(t * B) ** 2

fs_uri = [1.0, 1.5, 2.0, 4.0]
fix, axes = plt.subplots(2, 2)
for ax, fs in zip(axes.ravel(), fs_uri):
    Ts = 1 / fs
    t_s = np.arange(-3, 3 + Ts / 2, Ts)
    if 0 not in t_s:  
        t_s = np.concatenate((t_s, [0.0]))
        t_s.sort()
    x_s = np.sinc(t_s * B) ** 2

    x_recon = np.zeros_like(t)
    for n, xn in enumerate(x_s):
        x_recon += x_s[n] * np.sinc((t - t_s[n]) / Ts)
    ax.plot(t, x, color="blue", label="Semnal continuu")
    markerline, stemlines, baseline = ax.stem(t_s, x_s, linefmt=":", markerfmt="o", label="Esationare")
    stemlines.set_color("green")
    ax.plot(t, x_recon, color="pink", label="Reconstructia semnalului")
    ax.set_title(f"Fs = {fs} Hz")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

plt.tight_layout()
path = f"./Lab6/ex1_B={B}.pdf"
plt.savefig(path, format="pdf")
plt.show()


B = 7
frec_esantionare = 5000
t = np.linspace(-3, 3, frec_esantionare)
x = np.sinc(t * B) ** 2

fs_uri = [1.0, 1.5, 2.0, 4.0]
fix, axes = plt.subplots(2, 2)
for ax, fs in zip(axes.ravel(), fs_uri):
    Ts = 1 / fs
    t_s = np.arange(-3, 3 + Ts / 2, Ts)
    if 0 not in t_s:  
        t_s = np.concatenate((t_s, [0.0]))
        t_s.sort()
    x_s = np.sinc(t_s * B) ** 2

    x_recon = np.zeros_like(t)
    for n, xn in enumerate(x_s):
        x_recon += x_s[n] * np.sinc((t - t_s[n]) / Ts)
    ax.plot(t, x, color="blue", label="Semnal continuu")
    markerline, stemlines, baseline = ax.stem(t_s, x_s, linefmt=":", markerfmt="o", label="Esationare")
    stemlines.set_color("green")
    ax.plot(t, x_recon, color="pink", label="Reconstructia semnalului")
    ax.set_title(f"Fs = {fs} Hz")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

plt.tight_layout()
path = f"./Lab6/ex1_B={B}.pdf"
plt.savefig(path, format="pdf")
plt.show()
# Ex 2

N = 100
x = np.random.rand(N)
x_list = [x] 

x_curr = x.copy()
for i in range(3):
    x_curr = np.convolve(x_curr, x)
    x_list.append(x_curr)

fig, axes = plt.subplots(4, 1, figsize=(10, 12))
for i, xi in enumerate(x_list):
    axes[i].plot(xi)
    if i==0:
        axes[i].set_title("Semnal original")
    elif i == 1:
        axes[i].set_title(f"Convolutie o {i} data")
    else: 
        axes[i].set_title(f"Convolutie de {i} ori")
    axes[i].grid(True)
plt.tight_layout(pad=5.0)
plt.savefig("./Lab6/ex2_semnal_random.pdf", format="pdf")
plt.show()


x_block = np.zeros(N)
x_block[40:60] = 1  # bloc rectangular

x_list_block = [x_block]
x_curr = x_block.copy()
for i in range(3):
    x_curr = np.convolve(x_curr, x_block)
    x_list_block.append(x_curr)

fig, axes = plt.subplots(4, 1, figsize=(10, 12))
for i, xi in enumerate(x_list_block):
    axes[i].plot(xi)
    if i == 0:
        axes[i].set_title("Semnal original")
    elif i == 1:
        axes[i].set_title(f"Convolutie o {i} data")
    else: 
        axes[i].set_title(f"Convolutie de {i} ori")
    axes[i].grid(True)
plt.tight_layout(pad=5.0)
plt.savefig("./Lab6/ex2_semnal_bloc_rectangular.pdf", format="pdf")
plt.show()

# Ex 3

N = 4

coef_p = np.random.randint(-13, 12, N+1)
coef_q = np.random.randint(-13, 12, N+1)
p = np.poly1d(coef_p)
q = np.poly1d(coef_q)
r = np.polymul(p, q)
print(f"Numpy (p*q):\n{r}\n")

M = len(coef_p) + len(coef_q) - 1

fft_p = np.fft.fft(coef_p[::-1], M)  # Inversare ca sa fie ca la polymul
fft_q = np.fft.fft(coef_q[::-1], M)

fft_r = fft_p * fft_q

coef_r = np.fft.ifft(fft_r).real 

coef_r = np.round(coef_r)[::-1]
r_fft = np.poly1d(coef_r)
print(f"FFT (p*q):\n{r_fft}\n")

# Ex 4

n = 20
t = np.linspace(0, np.pi, n)  
x = np.sin(t)
d = 3
y = np.roll(x, d)
deplasarea_recuperata_fara_conjugata = np.fft.ifft(np.fft.fft(x) * np.fft.fft(y)).real.argmax()
deplasarea_recuperata_cu_conjugata = len(x) - np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(y))).real.argmax()
deplasare_recuperata_formula2 = np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(x)) /  np.abs(np.fft.fft(y) * np.conj(np.fft.fft(x)))).real.argmax()
print(f"Deplasarea originala: {d}\nDeplasarea recuperata cu IFFT(FFT(x) · FFT(y)) (aproximeaza deplasarea): {deplasarea_recuperata_fara_conjugata}\nDeplasarea recuperata cu len(semnal) - IFFT(FFT(x) · conjugata(FFT(y))) (deplasarea e precisa): {deplasarea_recuperata_cu_conjugata}\nDeplasarea recuperata cu IFFT(FFT(y) ⊘ FFT(x)) (deplasarea e precisa): {deplasare_recuperata_formula2}")

# Ex 5

f = 100
A = 1
phi = 0
perioada = 1
fs = 500
t = np.linspace(0, perioada, int(perioada * fs))
Nw = 200
semnal_sinusoidal = A * np.sin(2 * np.pi * f * t + phi)
hanning_window = np.hanning(Nw)
signal_through_hanning = semnal_sinusoidal[:Nw] * hanning_window
rectangular_window = np.ones(Nw)
signal_through_rect_windows = semnal_sinusoidal[:Nw] * rectangular_window

plt.plot(t[:Nw], semnal_sinusoidal[:Nw], label="Semnal original")
plt.plot(t[:Nw], signal_through_hanning, label="Semnal cu Hanning")
plt.plot(t[:Nw], signal_through_rect_windows, label="Semnal cu Rectangular window")
plt.title("Semnal original vs prin Hanning vs prin rectangular window")
plt.xlabel("Timp [s]")
plt.ylabel("Amplitudine")
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig("./Lab6/ex5_semnal_sinusoidal_prin_ferestre.pdf", format="pdf")
plt.show()

# Ex 6

x = np.genfromtxt('./Lab5/Train.csv', delimiter=",")
signal = x[:, 2]  
signal = signal[~np.isnan(signal)] # Sterg nan values

# a) pastrez datele pe 3 zile (72 de ore)
signal_3days = signal[:72]

plt.plot(signal_3days)
plt.title("Semnal original pe 3 zile")
plt.savefig("./Lab6/ex6_a_semnal_pe_3_zile.pdf", format="pdf")
plt.show()

# b)
w_list = [5, 9, 13, 17]
fig, axes = plt.subplots(len(w_list), 1)
for ax, w in zip(axes, w_list):
    semnal_netezit = np.convolve(signal_3days, np.ones(w), 'valid') / w
    ax.plot(semnal_netezit)
    ax.set_title(f"Semnal netezit cu w: {w}")
    ax.grid(True)
plt.tight_layout()
plt.savefig("./Lab6/ex6_b_semnal_netezit_cu_diferite_w.pdf", format="pdf")
plt.show()

# c)

print("Semnalul e esantionat la fiecare ora => fs = 1 => f_Nyquist = fs / 2 = 0.5. Frecventa de taiere f_t < f_Nyquist (f_t prea mica poate elimina si variatiile utile ale semnalului, una prea mare lasa sa treaca zgomot). Frecventa normalizata f_norm = f_t / f_Nyquist")
f_t = 0.1

# d, e)

b_butter, a_butter = scipy.signal.butter(5, f_t, btype='low')
semnal_filtrat_butter = scipy.signal.lfilter(b_butter, a_butter, signal_3days)

rp_list = [5, 1, 10]
plt.plot(semnal_filtrat_butter)
plt.grid(True)
plt.title("Filtru butter ordin 5")
plt.savefig("./Lab6/ex6_f_semnal_filtrat_butter_ordin_5.pdf", format="pdf")
plt.show()
b_cheby, a_cheby = scipy.signal.cheby1(5, 5, f_t, btype='low')
semnal_filtrat_cheby = scipy.signal.lfilter(b_cheby, a_cheby, signal_3days)
plt.plot(semnal_filtrat_cheby)
plt.grid(True)
plt.title("Filtru cheby1 cu rp 5, ordin 5")
plt.savefig("./Lab6/ex6_f_semnal_filtrart_cheby1_rp5_ordin_5.pdf", format="pdf")
plt.show()

fig, ax = plt.subplots(5, 1, figsize=(10, 12))
ax[0].plot(signal_3days)
ax[0].set_title("Semnal original")
ax[0].grid(True)
ax[1].plot(semnal_filtrat_butter)
ax[1].set_title("Semnal filtrat cu filtru butter ordin 5")
ax[1].grid(True)
for i, rp in enumerate(rp_list):
    b_cheby, a_cheby = scipy.signal.cheby1(5, rp, f_t, btype='low')
    semnal_filtrat_cheby = scipy.signal.lfilter(b_cheby, a_cheby, signal_3days)
    ax[i + 2].plot(semnal_filtrat_cheby)
    ax[i + 2].set_title(f"Semnal filtrat cu filtru cheby1 cu rp: {rp} ordin 5")
    ax[i + 2].grid(True)
plt.tight_layout(pad=4.0)
plt.savefig("./Lab6/ex6_d_e_semnal_filtrat_cu_filtru_cheby1(dif_rp)_si_butter_ordin_5.pdf", format="pdf")
plt.show()

# Aleg filtrul butter deoarece pare sa se aproprie mai mult de semnalul original

# f) ordin 3

b_butter, a_butter = scipy.signal.butter(3, f_t, btype='low')
semnal_filtrat_butter = scipy.signal.lfilter(b_butter, a_butter, signal_3days)

rp_list = [5, 1, 10]
plt.plot(semnal_filtrat_butter)
plt.grid(True)
plt.title("Filtru butter ordin 3")
plt.savefig("./Lab6/ex6_f_semnal_filtrat_butter_ordin_3.pdf", format="pdf")
plt.show()
b_cheby, a_cheby = scipy.signal.cheby1(3, 5, f_t, btype='low')
semnal_filtrat_cheby = scipy.signal.lfilter(b_cheby, a_cheby, signal_3days)
plt.plot(semnal_filtrat_cheby)
plt.grid(True)
plt.title("Filtru cheby1 cu rp 5, ordin 3")
plt.savefig("./Lab6/ex6_f_semnal_filtrart_cheby1_rp5_ordin_3.pdf", format="pdf")
plt.show()

fig, ax = plt.subplots(5, 1, figsize=(10, 12))
ax[0].plot(signal_3days)
ax[0].set_title("Semnal original")
ax[0].grid(True)
ax[1].plot(semnal_filtrat_butter)
ax[1].set_title("Semnal filtrat cu filtru butter ordin 3")
ax[1].grid(True)
for i, rp in enumerate(rp_list):
    b_cheby, a_cheby = scipy.signal.cheby1(3, rp, f_t, btype='low')
    semnal_filtrat_cheby = scipy.signal.lfilter(b_cheby, a_cheby, signal_3days)
    ax[i + 2].plot(semnal_filtrat_cheby)
    ax[i + 2].set_title(f"Semnal filtrat cu filtru cheby1 cu rp: {rp} ordin 3")
    ax[i + 2].grid(True)
plt.tight_layout(pad=4.0)
plt.savefig("./Lab6/ex6_d_e_semnal_filtrat_cu_filtru_cheby1(dif_rp)_si_butter_ordin_3.pdf", format="pdf")
plt.show()

# f) ordin 8

b_butter, a_butter = scipy.signal.butter(8, f_t, btype='low')
semnal_filtrat_butter = scipy.signal.lfilter(b_butter, a_butter, signal_3days)

rp_list = [5, 1, 10]
plt.plot(semnal_filtrat_butter)
plt.grid(True)
plt.title("Filtru butter ordin 8")
plt.savefig("./Lab6/ex6_f_semnal_filtrat_butter_ordin_8.pdf", format="pdf")
plt.show()
b_cheby, a_cheby = scipy.signal.cheby1(8, 5, f_t, btype='low')
semnal_filtrat_cheby = scipy.signal.lfilter(b_cheby, a_cheby, signal_3days)
plt.plot(semnal_filtrat_cheby)
plt.grid(True)
plt.title("Filtru cheby1 cu rp 5, ordin 8")
plt.savefig("./Lab6/ex6_f_semnal_filtrart_cheby1_rp5_ordin_8.pdf", format="pdf")
plt.show()

fig, ax = plt.subplots(5, 1, figsize=(10, 12))
ax[0].plot(signal_3days)
ax[0].set_title("Semnal original")
ax[0].grid(True)
ax[1].plot(semnal_filtrat_butter)
ax[1].set_title("Semnal filtrat cu filtru butter ordin 8")
ax[1].grid(True)
for i, rp in enumerate(rp_list):
    b_cheby, a_cheby = scipy.signal.cheby1(8, rp, f_t, btype='low')
    semnal_filtrat_cheby = scipy.signal.lfilter(b_cheby, a_cheby, signal_3days)
    ax[i + 2].plot(semnal_filtrat_cheby)
    ax[i + 2].set_title(f"Semnal filtrat cu filtru cheby1 cu rp: {rp} ordin 8")
    ax[i + 2].grid(True)
plt.tight_layout(pad=4.0)
plt.savefig("./Lab6/ex6_d_e_semnal_filtrat_cu_filtru_cheby1(dif_rp)_si_butter_ordin_8.pdf", format="pdf")
plt.show()

# consirder ca cel mai bine s-a descuract cu ordin mai mic, 3, filtru cheby1 cu rp = 5