import numpy as np
import matplotlib.pyplot as plt

def compute_variance_par(X, K, verbose=False):
    mean_boot = []
    median_boot = []
    for k in range(K):
        sample = np.random.choice(X, replace=True, size=X.shape[0])
        mean_boot.append(sample.mean())
        median_boot.append(np.median(sample))
    mean_boot = np.array(mean_boot)
    median_boot = np.array(median_boot)
    var_mean = 1/(K-1)*np.sum(np.square(mean_boot-mean_boot.mean()))
    var_median = 1/(K-1)*np.sum(np.square(median_boot-median_boot.mean()))
    if verbose:
        print(f"mean extreme values for {K=} bootstraps and {n=}: min: {mean_boot.min()}, max: {mean_boot.max()}")
        print(f"variance of the mean for {K=} bootstraps and {n=}: {var_mean}")
        print(f"median extreme values for {K=} bootstraps and {n=}: min: {median_boot.min()}, max: {median_boot.max()}")
        print(f"variance of the median for {K=} bootstraps and {n=}: {var_median}")
    return var_mean, var_median

# Pareto
beta = 1.
k=1.05

# parameters of the first bootstrap
K=100
n=200

#parameter for the second bootstrap
M=1000 # We take it bigger 

X = (np.random.pareto(k, size=n) + 1) * beta

compute_variance_par(X, K, verbose=True)

boot_vars_mean = []
boot_vars_median = []

for m in range(M):
    # Resample from the original data to create a new bootstrap universe
    X_resampled = np.random.choice(X, replace=True, size=n)
    
    # Compute the variances for this universe
    v_mean, v_median = compute_variance_par(X_resampled, K)
    
    boot_vars_mean.append(v_mean)
    boot_vars_median.append(v_median)

boot_vars_mean = np.array(boot_vars_mean)
boot_vars_median = np.array(boot_vars_median)

# Calculate 95% Percentile Confidence Intervals 
ci_mean = np.percentile(boot_vars_mean, [2.5, 97.5])
ci_median = np.percentile(boot_vars_median, [2.5, 97.5])

print("\n=== 95% Bootstrap Confidence Intervals ===")
print(f"Variance of Mean  : [{ci_mean[0]:.5f}, {ci_mean[1]:.5f}]")
print(f"Variance of Median: [{ci_median[0]:.5f}, {ci_median[1]:.5f}]")

#  Visualization Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Variance of the Mean Distribution
axes[0].hist(boot_vars_mean, bins=40, color='#34495e', alpha=0.75, edgecolor='black')
axes[0].axvline(ci_mean[0], color='red', linestyle='--', linewidth=2, label=f'2.5th Pctl: {ci_mean[0]:.4f}')
axes[0].axvline(ci_mean[1], color='red', linestyle='--', linewidth=2, label=f'97.5th Pctl: {ci_mean[1]:.4f}')
axes[0].set_title("Bootstrap Distribution: Variance of the Mean", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Variance Value")
axes[0].set_ylabel("Frequency")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Variance of the Median Distribution
axes[1].hist(boot_vars_median, bins=40, color='#e67e22', alpha=0.75, edgecolor='black')
axes[1].axvline(ci_median[0], color='red', linestyle='--', linewidth=2, label=f'2.5th Pctl: {ci_median[0]:.4f}')
axes[1].axvline(ci_median[1], color='red', linestyle='--', linewidth=2, label=f'97.5th Pctl: {ci_median[1]:.4f}')
axes[1].set_title("Bootstrap Distribution: Variance of the Median", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Variance Value")
axes[1].set_ylabel("Frequency")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
