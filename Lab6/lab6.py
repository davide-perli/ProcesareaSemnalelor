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