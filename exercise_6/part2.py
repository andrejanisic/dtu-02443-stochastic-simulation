"""
Exercise 6, Part 2 - two call types sharing m lines.

The joint number of busy lines (i = type 1, j = type 2) has the distribution
    P(i, j) = c * A1^i / i! * A2^j / j! ,   on the triangle  0 <= i + j <= m.
Parameters:  A1 = A2 = 4,  m = 10 

We sample P(i, j) three ways and check each with a chi-square test:
    (a) direct Metropolis-Hastings
    (b) coordinatewise Metropolis-Hastings
    (c) Gibbs sampling

"""

import math
import numpy as np
from scipy import stats

A1 = 4      
A2 = 4      
M = 10      


W1 = np.array([A1 ** k / math.factorial(k) for k in range(M + 1)])
W2 = np.array([A2 ** k / math.factorial(k) for k in range(M + 1)])

# All (i, j) with i, j >= 0 and i + j <= M  (there are 66)
def triangle_states():
    return [(i, j) for i in range(M + 1) for j in range(M + 1) if i + j <= M]

# Unnormalised weight 
def g(i, j):
    return (A1 ** i / math.factorial(i)) * (A2 ** j / math.factorial(j))


def true_joint():
    states = triangle_states()
    w = np.array([g(i, j) for (i, j) in states], dtype=float)
    return states, w / w.sum()


def coord_ratio(k, d, A):
    if d == 1:
        return A / (k + 1) 
    elif d == -1:
        return k / A   
    else:
        return 1.0


#Part 2a
def metropolis_joint(n_samples, burn_in=5000, thin=50, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    i, j = 0, 0
    samples = np.empty((n_samples, 2), dtype=int)
    recorded = 0

    for step in range(burn_in + n_samples * thin):
        # propose a joint move: di, dj each uniform on {-1, 0, +1}  (symmetric)
        di = int(rng.integers(-1, 2))
        dj = int(rng.integers(-1, 2))
        i2, j2 = i + di, j + dj

        # outside the triangle
        if i2 < 0 or j2 < 0 or i2 + j2 > M:
            pass
        else:
            ratio = coord_ratio(i, di, A1) * coord_ratio(j, dj, A2)
            if rng.uniform(0, 1) < min(1.0, ratio):
                i, j = i2, j2

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples[recorded] = (i, j)
            recorded += 1

    return samples


#Part 2b
def metropolis_coord_step(k, other, A, rng):
    if rng.uniform(0, 1) < 0.5:
        kp, ratio = k + 1, A / (k + 1)     
    else:
        kp, ratio = k - 1, k / A           
    if kp < 0 or kp > M - other:           
        return k
    if rng.uniform(0, 1) < min(1.0, ratio):
        return kp                          
    return k                               


def metropolis_coordinatewise(n_samples, burn_in=5000, thin=50, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    i, j = 0, 0
    samples = np.empty((n_samples, 2), dtype=int)
    recorded = 0

    for step in range(burn_in + n_samples * thin):
        i = metropolis_coord_step(i, j, A1, rng)   
        j = metropolis_coord_step(j, i, A2, rng)   

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples[recorded] = (i, j)
            recorded += 1

    return samples


# Part 2c
def draw_truncated_poisson(K, weights, rng):
    c = np.cumsum(weights[:K + 1])
    u = rng.uniform(0, c[-1])
    return int(np.searchsorted(c, u))


def gibbs(n_samples, burn_in=5000, thin=50, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    i, j = 0, 0
    samples = np.empty((n_samples, 2), dtype=int)
    recorded = 0

    for step in range(burn_in + n_samples * thin):
        i = draw_truncated_poisson(M - j, W1, rng)   
        j = draw_truncated_poisson(M - i, W2, rng) 

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples[recorded] = (i, j)
            recorded += 1

    return samples


def chi_square_test_2d(samples):
    states, pmf = true_joint()
    N = len(samples)

    codes = samples[:, 0] * (M + 2) + samples[:, 1]
    state_codes = np.array([i * (M + 2) + j for (i, j) in states])
    observed = np.bincount(codes, minlength=(M + 2) ** 2)[state_codes].astype(float)

    expected = pmf * N
    chi2 = np.sum((observed - expected) ** 2 / expected)
    dof = len(states) - 1             
    p_value = stats.chi2.sf(chi2, dof)
    return chi2, dof, p_value, states, observed, expected


def plot_results(samples, observed, states, out_name):
    import os
    import matplotlib.pyplot as plt

    N = len(samples)
    _, pmf = true_joint()

    emp = np.full((M + 1, M + 1), np.nan)
    tru = np.full((M + 1, M + 1), np.nan)
    for k, (i, j) in enumerate(states):
        emp[i, j] = observed[k] / N
        tru[i, j] = pmf[k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))
    vmax = np.nanmax(tru)
    for ax, grid, title in ((ax1, emp, "MCMC empirical  P(i,j)"),
                            (ax2, tru, "true  P(i,j)")):
        im = ax.imshow(grid, origin="lower", vmin=0, vmax=vmax, cmap="viridis")
        ax.set_xlabel("j  (type-2 lines)")
        ax.set_ylabel("i  (type-1 lines)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), out_name)
    fig.savefig(out, dpi=120)
    plt.close()
    print(f"saved plot to {out}")


def generate(label, sampler, out_name):
    samples = sampler(n_samples=100_000, burn_in=5000, thin=50)
    chi2, dof, p_value, states, observed, expected = chi_square_test_2d(samples)
    _, pmf = true_joint()

    print(f"\n=== {label} ===")

    #most likely cells
    order = np.argsort(pmf)[::-1]  
    show = list(order[:8]) + [states.index((10, 0))]
    print(f"{'(i,j)':>8} {'observed':>10} {'expected':>10} {'P(i,j)':>9}")
    for k in show:
        i, j = states[k]
        print(f"{str((i, j)):>8} {observed[k]:>10.0f} {expected[k]:>10.1f} {pmf[k]:>9.5f}")
    print("-" * 42)
    print(f"chi^2 = {chi2:.2f},  dof = {dof},  p-value = {p_value:.3f}   "
          f"(min expected {expected.min():.1f})")

    plot_results(samples, observed, states, out_name)


def main():
    generate("(a) direct Metropolis-Hastings",
                   metropolis_joint, "part2a_distribution.png")
    generate("(b) coordinatewise Metropolis-Hastings",
                   metropolis_coordinatewise, "part2b_distribution.png")
    generate("(c) Gibbs sampling",
                   gibbs, "part2c_distribution.png")


if __name__ == "__main__":
    main()
