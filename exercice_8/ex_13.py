import numpy as np

a, b = -5., 5.
X = np.array([56, 101, 78, 67, 93, 87, 64, 72, 80, 69])
K = 100000 # number of bootstraps
n = X.shape[0]

samples = np.random.choice(X, size=(K, n), replace=True)

# estimate of the mean per bootstrap
mu_hat = samples.mean(axis=1)

# estimate of the mean over the bootstraps
mu_bar = mu_hat.mean()
acceptance = ((mu_hat - mu_bar) > a) & ((mu_hat - mu_bar) < b)
p = np.mean(acceptance)

print(f'{p=:.5f}')