import numpy as np, matplotlib.pyplot as plt
from scipy import signal

# Exercitiul 1

# a)

N = 1000

# ecuatia de gradul 2: 0.33 * X ** 2 + 1.7 * X + 23

X = np.linspace(0.0, 7.0, N)
trend = 0.33 * X ** 2 + 1.7 * X + 23

frecventa_1 = 13
frecventa_2 = 141

sezon = np.sin(2 * np.pi * frecventa_1 * X) + np.cos(2 * np.pi * frecventa_2 * X)

# zgomot alb gaussian variatii mici

noise = np.random.normal(loc=0.0, scale=3.0, size=N)

serie_timp = trend + sezon + noise

fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

ax[0].plot(X, serie_timp, color='k', lw=1)
ax[0].set_title('Seria de timp (trend + sezon + zgomot)')
ax[0].set_ylabel('Amplitudine')

ax[1].plot(X, trend, color='tab:blue')
ax[1].set_title('Trend (polinom grad 2)')
ax[1].set_ylabel('Amplitudine')

ax[2].plot(X, sezon, color='tab:orange')
ax[2].set_title('Sezon (doua frecvente)')
ax[2].set_ylabel('Amplitudine')

ax[3].plot(X, noise, color='tab:green')
ax[3].set_title('Zgomot alb gaussian')
ax[3].set_xlabel('Timp')
ax[3].set_ylabel('Amplitudine')

fig.tight_layout()
plt.savefig("./Lab8/ex1_a.pdf", format="pdf")
plt.show()

# b)

serie_timp_centrata = serie_timp - np.mean(serie_timp)

T = len(serie_timp_centrata)

y = serie_timp_centrata
numitor = np.sum(y ** 2)

r = np.zeros(T)

for k in range(T):
    numarator = np.sum(y[k:T] * y[0:T-k])
    r[k] = numarator / numitor

lags = np.arange(T)
plt.figure(figsize=(12, 4))
plt.plot(lags, r)
plt.title('Autocorelatie manuala r_k')
plt.xlabel('Lag (esantioane)')
plt.ylabel('r_k')
plt.tight_layout()
plt.savefig("./Lab8/ex1_b_autocorelatie_manuala.pdf", format="pdf")
plt.show()

r_numpy = np.correlate(y, y, mode='full')
r_numpy = r_numpy[T-1:]           # esantionae de la 0 la T - 1 fara negative
r_numpy = r_numpy / r_numpy[0] # normalizare = impartirea la numitor de la cea manuala

lags_numpy = np.arange(r_numpy.size)
plt.figure(figsize=(12, 4))
plt.plot(lags_numpy, r_numpy)
plt.title('Autocorelatie (numpy.correlate)')
plt.xlabel('Lag (esantioane)')
plt.ylabel('r_k')
plt.tight_layout()
plt.savefig("./Lab8/ex1_b_autocorelatie_numpy.pdf", format="pdf")
plt.show()

# c)

p = 10

X_ar = np.column_stack([y[p - i:T - i] for i in range(1, p + 1)])
y_target = y[p:T]

phi = np.linalg.lstsq(X_ar, y_target, rcond=None)[0] # coeficeinti model ar

y_pred = np.dot(X_ar, phi) # X_ar @ phi

serie_pred = y_pred + np.mean(serie_timp) # medie pt. comp cu seria originala

plt.figure(figsize=(12, 4))
plt.plot(X, serie_timp, label='Original')
plt.plot(X[p:], serie_pred, label=f'AR({p}) predictii',)
plt.title(f'AR({p}): original vs predictii')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')
plt.legend()
plt.tight_layout()
plt.savefig("./Lab8/ex1_c_ar_predictii.pdf", format="pdf")
plt.show()

# d)

best_p = None
best_m = None
best_mse = float('inf')
# Normalizari cu o centrare valori in [0, 1]
y = (serie_timp_centrata - np.mean(serie_timp_centrata)) / np.std(serie_timp_centrata)
serie_timp = (serie_timp - np.mean(serie_timp)) / np.std(serie_timp)
for p in range(1, 31):  
    for m in range(1, 6): # predictie pe termen mai lung
        if p + m >= T:
            continue

        X_ar = np.column_stack([y[p - i:T - i - m] for i in range(1, p + 1)])
        y_target = y[p + m:T]

        phi = np.linalg.lstsq(X_ar, y_target, rcond=None)[0]

        y_pred = X_ar @ phi

        mse = np.mean((y_target - y_pred) ** 2)

        if mse < best_mse:
            best_p = p
            best_m = m
            best_mse = mse

print(f"Best p for AR(p): {best_p}, Best horizon (m): {best_m}, with MSE: {best_mse}")

X_ar = np.column_stack([y[best_p - i:T - i - best_m] for i in range(1, best_p + 1)])
y_target = y[best_p + best_m:T]
phi = np.linalg.lstsq(X_ar, y_target, rcond=None)[0]
y_pred = X_ar @ phi
serie_pred = y_pred + np.mean(serie_timp)

plt.figure(figsize=(12, 4))
plt.plot(X, serie_timp, label='Original')  
plt.plot(X[best_p + best_m:], serie_pred, label=f'AR({best_p}) predictii, m={best_m}')  
plt.title(f'AR({best_p}), m={best_m}: original vs predictii')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')
plt.legend()
plt.tight_layout()
plt.savefig("./Lab8/ex1_d_ar_predictii_cu_auto-hyperparameter-tunning.pdf", format="pdf")
plt.show()