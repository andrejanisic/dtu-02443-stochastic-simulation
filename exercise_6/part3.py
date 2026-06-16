"""
Exercise 6, Part 3 - Bayesian inference for X ~ N(theta, psi) with MCMC
"""

import math
import numpy as np

RHO = 0.5
PRIOR_COV = np.array([[1.0, RHO], [RHO, 1.0]])
CONSTANT = 1.0 / (2.0 * (1.0 - RHO ** 2))


def sample_prior(rng):
    xi, gamma = rng.multivariate_normal([0.0, 0.0], PRIOR_COV)
    return xi, gamma, math.exp(xi), math.exp(gamma)


def simulate_data(theta, psi, n, rng):
    return rng.normal(theta, math.sqrt(psi), size=n)


def suff_stats(x):
    n = len(x)
    xbar = float(np.mean(x))
    S = float(np.sum((x - xbar) ** 2))
    return n, xbar, S


def log_posterior(xi, gamma, xbar, S, n):
    theta = math.exp(xi)
    psi = math.exp(gamma)
    log_prior = -CONSTANT * (xi * xi - 2 * RHO * xi * gamma + gamma * gamma)
    log_lik = -0.5 * n * gamma - (S + n * (xbar - theta) ** 2) / (2.0 * psi)
    return log_prior + log_lik


def metropolis_posterior(xbar, S, n, n_samples=20_000, burn_in=5_000,
                         prop_sd=0.3, start=(0.0, 0.0), rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    xi, gamma = start
    lp = log_posterior(xi, gamma, xbar, S, n)
    out = np.empty((n_samples, 2))
    rec = accepted = 0

    for t in range(burn_in + n_samples):
        dxi, dgamma = rng.normal(0.0, prop_sd, size=2)  
        xi2, gamma2 = xi + dxi, gamma + dgamma
        lp2 = log_posterior(xi2, gamma2, xbar, S, n)
        if math.log(rng.random()) < lp2 - lp:             
            xi, gamma, lp = xi2, gamma2, lp2
            if t >= burn_in:
                accepted += 1
        if t >= burn_in:
            out[rec] = (math.exp(xi), math.exp(gamma))
            rec += 1
    return out, accepted / n_samples


def plot(samples, true_pt, title, fname):
    import os
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.12,
                color="steelblue", label="posterior samples")
    plt.scatter([true_pt[0]], [true_pt[1]], color="red", marker="o", s=50,
                edgecolor="black", zorder=5, label="true (theta, psi)")
    plt.xlabel(r"$\theta$ (mean)")
    plt.ylabel(r"$\psi$ (variance)")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), fname)
    plt.savefig(out, dpi=120)
    plt.close()


# posterior narrows as n grows so smaller proposal SDs
PROP_SD = {10: 0.28, 100: 0.11, 1000: 0.035}


def sample_and_report(n, data_full, true_pt):
    x = data_full[:n]
    nn, xbar, S = suff_stats(x)

    samples, acc = metropolis_posterior(xbar, S, nn, prop_sd=PROP_SD[n],
                                        start=(math.log(xbar), math.log(S / nn)),
                                        rng=np.random.default_rng(100 + n))

    th, ps = samples[:, 0], samples[:, 1]
    th_ci, ps_ci = np.percentile(th, [5, 95]), np.percentile(ps, [5, 95])
    print(f"  n={n:<4} (acceptance {acc:.2f})")
    print(f"     theta: mean={th.mean():.3f}  sd={th.std():.3f}  "
          f"90%CI=[{th_ci[0]:.3f},{th_ci[1]:.3f}]")
    print(f"     psi  : mean={ps.mean():.3f}  sd={ps.std():.3f}  "
          f"90%CI=[{ps_ci[0]:.3f},{ps_ci[1]:.3f}]")

    plot(samples, true_pt, f"Posterior of (theta, psi),  n={n}",
         f"part3_posterior_n{n}.png")
    return samples


def main():

    #(a)
    rng_prior = np.random.default_rng(2025)
    xi0, gamma0, theta_true, psi_true = sample_prior(rng_prior)
    print("\n(a) PRIOR DRAW   (xi,gamma) ~ N(0,Sigma);  (theta,psi)=(e^xi,e^gamma):")
    print(f"    (xi,gamma)=({xi0:+.3f},{gamma0:+.3f})  ->  "
          f"theta(mean)={theta_true:.4f},  psi(variance)={psi_true:.4f}")

    #(b)
    rng_data = np.random.default_rng(6)   
        
    data_full = simulate_data(theta_true, psi_true, 1000, rng_data)
    x10 = data_full[:10]
    n10, xbar10, S10 = suff_stats(x10)
    print("\n(b) DATA   X_1..X_10 ~ N(theta, psi)   (n=10):")
    print("    " + ", ".join(f"{v:.2f}" for v in x10))
    print(f"    sufficient stats:  xbar={xbar10:.3f}   "
          f"S=Sum(xi-xbar)^2={S10:.3f}   s^2={S10/(n10-1):.3f}")

    #(d)
    print("\n(d) POSTERIOR SAMPLING with random-walk Metropolis-Hastings:")
    sample_and_report(10, data_full, (theta_true, psi_true))

    #(e)
    print("\n(e) REPEAT with larger samples (same true (theta,psi)):")
    for n in (100, 1000):
        sample_and_report(n, data_full, (theta_true, psi_true))


if __name__ == "__main__":
    main()
