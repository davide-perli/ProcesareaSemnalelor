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

