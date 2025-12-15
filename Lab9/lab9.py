import matplotlib.pyplot as plt, numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Exercitiul 1

def genereaza_serie_timp(n=1000):
	X = np.linspace(0.0, 7.0, n)
	trend = 0.33 * X ** 2 + 1.7 * X + 23
	frecventa_1 = 13
	frecventa_2 = 141
	seasonal = np.sin(2 * np.pi * frecventa_1 * X) + np.cos(2 * np.pi * frecventa_2 * X)
	residuals = np.random.normal(loc=0.0, scale=3.0, size=n)
	observed = trend + seasonal + residuals
	return X, observed, trend, seasonal, residuals


N = 1000
X, observed, trend, seasonal, residuals = genereaza_serie_timp(n=N)

fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

ax[0].plot(X, observed)
ax[0].set_title('Observed (Trend + Seasonal + Residuals)')
ax[0].set_ylabel('Amplitudine')

ax[1].plot(X, trend)
ax[1].set_title('Trend')
ax[1].set_ylabel('Amplitudine')

ax[2].plot(X, seasonal)
ax[2].set_title('Seasonal')
ax[2].set_ylabel('Amplitudine')

ax[3].plot(X, residuals)
ax[3].set_title('Residuals')
ax[3].set_xlabel('Timp')
ax[3].set_ylabel('Amplitudine')

fig.tight_layout()
plt.savefig('./Lab9/ex1.pdf', format='pdf')
plt.show()


# Exercitiul 2
def mediere_exponentiala(serie, alpha):
	rezultat = np.zeros_like(serie)
	rezultat[0] = serie[0]
	for t in range(1, len(serie)):
		rezultat[t] = alpha * serie[t] + (1 - alpha) * rezultat[t-1]
	return rezultat

def eroare_mse(original, filtrat):
	return np.mean((original - filtrat) ** 2)

alphas = np.linspace(0.01, 1, 100)
mse_list = []
for alpha in alphas:
	filtrat = mediere_exponentiala(observed, alpha)
	mse_list.append(eroare_mse(observed, filtrat))
alpha_opt = alphas[np.argmin(mse_list)]

mediere_simpla = mediere_exponentiala(observed, alpha_opt)
mediere_dubla = mediere_exponentiala(mediere_simpla, alpha_opt)
mediere_tripla = mediere_exponentiala(mediere_dubla, alpha_opt)

fig, ax = plt.subplots(4, 1, figsize=(12, 8))

ax[0].plot(X, observed)
ax[0].set_title('Seria originala')
ax[0].set_ylabel('Amplitudine')

ax[1].plot(X, mediere_simpla)
ax[1].set_title(f'Mediere exponentiala simpla (alpha = {alpha_opt:.2f})')
ax[1].set_ylabel('Amplitudine')

ax[2].plot(X, mediere_dubla)
ax[2].set_title('Mediere exponentiala dubla')
ax[2].set_ylabel('Amplitudine')

ax[3].plot(X, mediere_tripla)
ax[3].set_title('Mediere exponentiala tripla')
ax[3].set_xlabel('Timp')
ax[3].set_ylabel('Amplitudine')

fig.tight_layout()
plt.savefig('./Lab9/ex2.pdf', format='pdf')
plt.show()

# Exercitiul 3

def model_MA(serie, q):
	n = len(serie)
	ma = np.zeros(n)
	epsilon = np.zeros(n)

	for t in range(n):
		if t < q:
			media_locala = np.mean(serie[:t+1])
		else:
			media_locala = np.mean(serie[t-q:t])
			
		epsilon[t] = serie[t] - media_locala
		ma[t] = media_locala + np.sum(epsilon[max(0, t-q):t])
	return ma, epsilon

q = 10
serie_MA, epsilon = model_MA(observed, q)
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

ax[0].plot(X, observed)
ax[0].set_title('Seria originala')
ax[0].set_ylabel('Amplitudine')

ax[1].plot(X, serie_MA)
ax[1].set_title(f'Model MA(q), q = {q}')
ax[1].set_ylabel('Amplitudine')

ax[2].plot(X, epsilon)
ax[2].set_title('Termeni de eroare epsilon')
ax[2].set_xlabel('Timp')
ax[2].set_ylabel('Eroare')

fig.tight_layout()
plt.savefig('./Lab9/ex3.pdf', format='pdf')
plt.show()

# Exercitiul 4
stationary = observed - trend
stationary = stationary - np.mean(stationary)

def calculeaza_epsilon(serie, orizont):
	n = len(serie)
	epsilon = np.zeros(n)

	for t in range(n):
		if t < orizont:
			media_locala = np.mean(serie[:t+1])
		else:
			media_locala = np.mean(serie[t-orizont:t])

		epsilon[t] = serie[t] - media_locala

	return epsilon

best_aic = np.inf
best_p, best_q = 0, 0
best_model = None

for p in range(21):
	for q in range(21):
		try:
			model = ARIMA(stationary, order=(p, 0, q))
			model_fit = model.fit()

			if model_fit.aic < best_aic:
				best_aic = model_fit.aic
				best_p = p
				best_q = q
				best_model = model_fit

		except:
			continue

print(f'Parametri optimi: p = {best_p}, q = {best_q}')
print(f'AIC minim: {best_aic:.2f}')

serie_ARMA = best_model.fittedvalues # estimare serie

# deviatie fata de media locala
orizont = max(best_p, best_q, 1)
epsilon_ARMA = calculeaza_epsilon(stationary, orizont)

fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

ax[0].plot(X, stationary)
ax[0].set_title('Seria stationara (fara trend si medie)')
ax[0].set_ylabel('Amplitudine')

ax[1].plot(X, serie_ARMA)
ax[1].set_title(f'Model ARMA(p, q) cu p = {best_p}, q = {best_q}')
ax[1].set_ylabel('Amplitudine')

ax[2].plot(X, epsilon_ARMA)
ax[2].set_title('Termeni de eroare epsilon')
ax[2].set_xlabel('Timp')
ax[2].set_ylabel('Eroare')

fig.tight_layout()
plt.savefig('./Lab9/ex4.pdf', format='pdf')
plt.show()
