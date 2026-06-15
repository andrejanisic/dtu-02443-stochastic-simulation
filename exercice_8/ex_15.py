import numpy as np
import matplotlib.pyplot as plt

X = np.array([5,4,9,6,21,17,11,20,7,10,21,15,13,16,8])
n = X.shape[0]

def var_var(X,K):
    var_hat = []
    for k in range(K):
        sample = np.random.choice(X, size=n, replace=True)
        var_hat_k = 1/(n-1)*np.sum(np.square(sample - sample.mean()))
        var_hat.append(var_hat_k)
    var_hat = np.array(var_hat)

    # estimate of the var over bootstrap
    var_of_the_var = 1/(K-1)*np.sum(np.square(var_hat - var_hat.mean()))
    #print(f'{K=}, {var_of_the_var=}')
    return var_of_the_var

K_values = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000])
results = []

for K in K_values:
    v_v = var_var(X, K)
    results.append(v_v)
    print(f"K = {K:<7} | var_of_the_var = {v_v:.4f}")

plt.figure(figsize=(10, 6))

# Plot the calculated values
plt.plot(K_values, results, marker='o', linewidth=2, color='#2c3e50', label='Estimated Variance of Variance')

# Add a horizontal baseline showing the final converged estimate for reference
plt.axhline(results[-1], color='#e74c3c', linestyle='--', alpha=0.8, 
            label=f'Converged Value (K=100k): {results[-1]:.2f}')

# Adjusting to a log scale on the x-axis makes it much easier to see the early stabilization
plt.xscale('log')
plt.title('Convergence of Bootstrap Variance Estimate ($Var(\\hat{\\sigma}^2)$)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Number of Bootstrap Replicates ($K$) - Log Scale', fontsize=11)
plt.ylabel('Variance of the Variance', fontsize=11)
plt.xticks(K_values, labels=[str(k) for k in K_values])
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()




