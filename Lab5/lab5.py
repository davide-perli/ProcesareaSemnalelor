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
plt.savefig("./Lab5/modul_transformata_fourier_semnal.pdf", format="pdf")
plt.show()

# Ex e

comp_cont = np.mean(signal) # Componenta continua a semnalului daca e 0 nu exista daca != 0 exista
print("Componenta continua a semnalului:", comp_cont)
signal_fcomp_cont = signal - comp_cont
X = np.fft.fft(signal_fcomp_cont)
X_mod = abs(X / N) 
X_mod = X_mod[:N // 2]

plt.figure(figsize=(10, 5))
# plt.plot(f, X_mod)
plt.semilogy(f, X_mod)
plt.title("Modulul Transformatei Fourier fara componeneta_continua")
plt.xlabel("Frecventa")
plt.ylabel("Modul transformata Fourier")
plt.grid(True)
# plt.xlim(0, fs/2)
plt.savefig("./Lab5/modul_transformata_fourier_fara_componeneta_continua.pdf", format="pdf")
plt.show()

# Ex f

top_indici = np.argsort(X_mod)[-4:][::-1]
topf_vf_mod_trans_fourier = X_mod[top_indici]
top_frec = f[top_indici]

print("Primele 4 varfuri ale modulului transformatei Fourier si frecventele lor:")
for i in range(4):
    print(f"{i+1}. Amplitudine: {top_indici[i]}, Frecventa: {top_frec[i]} Hz")

# Ex g
#
# 1 esantione pe ora => 24 esantionae pe zi => 168 de esantionae pe sapatamana => 720 de esantionae pe luna

index_start = 168 
ore_luna = 720
end_index = index_start + ore_luna
semnal_luna = signal[index_start:end_index]
t = np.arange(len(semnal_luna))

plt.figure(figsize=(12, 5))
plt.plot(t, semnal_luna)
plt.title("Trafic intr-o luna")
plt.xlabel("Timp [ore]")
plt.ylabel("Semnal")
plt.grid(True)
plt.savefig("./Lab5/semnal_trafic_pe_o_luna_de_luni.pdf", format="pdf")
plt.show()