import numpy as np, scipy, matplotlib.pyplot as plt

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
plt.savefig('./Lab11/ex1.pdf', format='pdf')
plt.show()


# Exercitiul 2

def build_hankel_matrix(y, L):

    N = len(y)
    K = N - L + 1
    
    X_hankel = np.zeros((L, K))
    for i in range(L):
        X_hankel[i, :] = y[i:i+K]
    
    return X_hankel


L = 200  
X_hankel = build_hankel_matrix(observed, L)

print(f"Time series dimension: N = {N}")
print(f"Window size: L = {L}")
print(f"Hankel matrix shape: {X_hankel.shape} (L × K, where K = N - L + 1 = {N - L + 1})")
print(f"\nHankel matrix:")
np.set_printoptions(precision=2, suppress=True, linewidth=100)
print(X_hankel)

fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))

ax2[0].plot(X, observed)
ax2[0].set_xlabel('Time')
ax2[0].set_ylabel('Value')
ax2[0].set_title('Observed Time Series')
ax2[0].grid(True)

im = ax2[1].imshow(X_hankel, aspect='auto', cmap='viridis')
plt.colorbar(im, ax=ax2[1], label='Value')
ax2[1].set_xlabel('Column (K)')
ax2[1].set_ylabel('Row (L)')
ax2[1].set_title('Hankel Matrix')

fig2.tight_layout()
plt.savefig('./Lab11/ex2.pdf', format='pdf')
plt.show()


# Exercitiul 3

def decomposition(X):
    L, K = X.shape
    # PCA X
    U, s, Vt = scipy.linalg.svd(X)

    XXT = X @ X.T
    val_descomp_XXT, vec_descomp_XXT = scipy.linalg.eigh(XXT)
    idx_XXT = val_descomp_XXT.argsort()[::-1] # desc ca sa fie ca PCA
    val_descomp_XXT = val_descomp_XXT[idx_XXT]
    vec_descomp_XXT = vec_descomp_XXT[:, idx_XXT]

    XTX = X.T @ X
    val_descomp_XTX, vec_descomp_XTX = scipy.linalg.eigh(XTX)
    idx_XTX = val_descomp_XTX.argsort()[::-1]
    val_descomp_XTX = val_descomp_XTX[idx_XTX]
    vec_descomp_XTX = vec_descomp_XTX[:, idx_XTX]

    return U, s, Vt, val_descomp_XXT, vec_descomp_XXT, val_descomp_XTX, vec_descomp_XTX


U, s, Vt, val_XXT, vec_XXT, val_XTX, vec_XTX = decomposition(X_hankel)

np.set_printoptions(precision=2, suppress=True, linewidth=100)

print("\nSingular values from PCA")
print(s[:10])

print("\nSQRT values X^XT")
print(np.sqrt(val_XXT[:10]))

print("\nSQRT values X^TX")
print(np.sqrt(val_XTX[:10]))

print("\nRelationships:")
print(" Sigma PCA = SQRT X^XT = SQRT X^TX")
print(" U PCA = vectors from the decomposition of XX^T")
print(" V PCA = vectors from the decomposition of X^TX")

# Exercitiul 4

def elementary_matrices(U, s, Vt):
    X_elem = []
    for i in range(len(s)):
        Xi = s[i] * np.outer(U[:, i], Vt[i, :])
        X_elem.append(Xi)
    return X_elem

# Construirea matricilor elementare Xi = sigma_i * u_i * v_i^T
X_elem = elementary_matrices(U, s, Vt)

def hankelizare(X):
    L, K = X.shape
    N = L + K - 1
    x_rec = np.zeros(N)
    counts = np.zeros(N)

    # From Hankel matrix back to time series
    for i in range(L):
        for j in range(K):
            x_rec[i + j] += X[i, j]
            counts[i + j] += 1

    return x_rec / counts

components = [hankelizare(Xi) for Xi in X_elem]

trend = components[0]

seasonal = components[1] + components[2]

residual = np.sum(components[3:], axis=0)

fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

ax[0].plot(observed)
ax[0].set_title("Original time series")

ax[1].plot(trend)
ax[1].set_title("SSA Trend")

ax[2].plot(seasonal)
ax[2].set_title("SSA Seasonal")

ax[3].plot(residual)
ax[3].set_title("SSA Residual")

plt.tight_layout()
plt.savefig("./Lab11/ex4_ssa.pdf")
plt.show()
