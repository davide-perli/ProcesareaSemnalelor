import matplotlib.pyplot as plt, numpy as np


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
