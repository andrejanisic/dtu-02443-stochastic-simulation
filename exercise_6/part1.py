"""
Exercise 6, Part 1
The number of busy lines in a trunk group (Erlang system) is given by a truncated Poisson distribution:
P(i) = c * A^i / i! ,   i = 0, 1, ..., m

using the Metropolis (random-walk) algorithm, then verify the fit with a
chi-square goodness-of-fit test.

Parameters (from Exercise 4): A = 8 (offered traffic), m = 10 servers.
This P(i) is exactly the steady-state "distribution of occupied servers" of the
M/M/m/m blocking system, and P(m) is the Erlang-B rejection probability (~0.1217).

"""

import math
import numpy as np
from scipy import stats

A = 8  
M = 10      


def g(i):
    """Unnormalised weight (handy for testing; the loop should use ratios)."""
    return A ** i / math.factorial(i)


def true_pmf():
    """The TRUE normalised P(i), used only for chi-square expected counts/plot."""
    w = np.array([g(i) for i in range(M + 1)], dtype=float)
    return w / w.sum()


def metropolis(n_samples, burn_in=2000, thin=1, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    x = 0                     
    samples = np.empty(n_samples, dtype=int)
    recorded = 0

    for step in range(burn_in + n_samples * thin):
        # propose y = x+1 or x-1, each with probability 1/2  (symmetric)
        if rng.uniform(low=0, high=1, size=1) < 0.5:
            y = x + 1
        else:
            y = x - 1
        
        if (y < 0) or (y > M):
            pass
        else:
            if y == x + 1:
                ratio = A / (x + 1)
            else:  
                ratio = x / A
            if rng.uniform(low=0, high=1, size=1) < min(1, ratio):
                x = y

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples[recorded] = x
            recorded += 1

    return samples



def chi_square_test(samples):
    observed = np.array([np.sum(samples == i) for i in range(M + 1)], dtype=float)
    expected = true_pmf() * len(samples)
    chi2 = np.sum((observed - expected) ** 2 / expected)
    dof = M                            
    p_value = stats.chi2.sf(chi2, dof)   
    return chi2, dof, p_value, observed, expected


def main():
    samples = metropolis(n_samples=100_000, burn_in=2000, thin=10)
    chi2, dof, p_value, observed, expected = chi_square_test(samples)

    pmf = true_pmf()
    print(f"{'i':>3} {'observed':>10} {'expected':>10} {'P(i)':>9}")
    for i in range(M + 1):
        print(f"{i:>3} {observed[i]:>10.0f} {expected[i]:>10.1f} {pmf[i]:>9.5f}")

    print("-" * 36)
    print(f"chi^2 = {chi2:.2f},  dof = {dof},  p-value = {p_value:.3f}")
    print(f"empirical P(busy={M}) = {observed[M] / len(samples):.4f}   "
          f"(Erlang-B reference = {pmf[M]:.4f})")

    import os
    import matplotlib.pyplot as plt

    here = os.path.dirname(__file__)
    N = len(samples)
    states = np.arange(M + 1)

    # empirical (MCMC) distribution vs the true truncated-Poisson pmf:
    plt.figure(figsize=(7, 4.5))
    width = 0.4
    plt.bar(states - width / 2, observed / N, width,
            label="MCMC (empirical)", color="steelblue")
    plt.bar(states + width / 2, pmf, width,
            label="truncated Poisson (true)", color="darkorange")
    plt.xlabel("number of busy lines  i")
    plt.ylabel("probability  P(i)")
    plt.title("Sampled vs. target distribution")
    plt.xticks(states)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(here, "part1_distribution.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"saved plot to {out}")

    # trace of a short RAW chain (no burn-in, no thinning) so the warm-up
    trace = metropolis(n_samples=1000, burn_in=0, thin=1,
                       rng=np.random.default_rng(0))
    plt.figure(figsize=(7, 4.5))
    plt.plot(trace, lw=0.8, color="steelblue")
    plt.axhline(A, color="grey", ls="--", lw=1, label=f"offered traffic A={A}")
    plt.xlabel("iteration")
    plt.ylabel("state (busy lines)")
    plt.title("Trace of the chain (no burn-in, no thinning)")
    plt.ylim(-0.5, M + 0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(here, "part1_trace.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"saved plot to {out}")


if __name__ == "__main__":
    main()
