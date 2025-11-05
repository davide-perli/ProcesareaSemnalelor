import numpy as np, matplotlib.pyplot as plt, csv

x = np.genfromtxt('./Lab5/Train.csv', delimiter=",")
signal = x[:, 2]  
signal = signal[~np.isnan(signal)] # Sterg nan values
print("Data: ", signal)
N = len(signal)

X = np.fft.fft(signal) # Transformata fourier
X_mod = abs(X / N) # Modulul transformatei
X_mod = X_mod[:N // 2] # Doar prima jumatate ca e simetrica
# Ex a 

# Semnalul a fost masurat din ora in ora (3600 de secunde). Sunt 18288 de esantionae
# fs = 1 / 3600 Hz
fs = round(1 / 3600, 6)
print("Frecventa de esantionare a semnalului Train.csv este: ", fs)

# Ex b
print(f"Un esantion pe ora => daca sunt 18288 de esantionae sunt 18288 ore, {int(18288 / 24)} zile sau {round(18288 / 24 / 365, 2)} ani")

# Ex c
fmax = fs / 2
print(f"Daca semnalul a fost esantionat corect(fara aliere) si optim, iar conform teoremei de esantionare Nyquist-Shannon f < fs / 2 => fmax = fs / 2 deci fmax = {fmax}")

# Ex d

f = np.linspace(0, fs / 2, N // 2) # Vec de frecvente
plt.figure(figsize=(10, 5))
# plt.plot(f, X_mod)
plt.semilogy(f, X_mod)
plt.title("Modulul Transformatei Fourier a semnalului")
plt.xlabel("Frecventa")
plt.ylabel("Modul transformata Fourier")
plt.grid(True)
# plt.xlim(0, fs/2)
plt.show()