import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from exercice_1.tests_utils import *

def LCG_hist(x, N=10000, a=5, c=1, M=16, n_bins=10, ax=None):
    # Fallback option: If no specific subplot axis is provided, create a standalone one
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=100)

    counts, bins, patches = ax.hist(
        x, 
        bins=n_bins, 
        color="#2b5c8f",    
        edgecolor="white",  
        linewidth=1.2, 
        alpha=0.85, 
        rwidth=0.95,
        zorder=2             
    )

    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0) 
    ax.set_axisbelow(True) 

    ax.set_title(f"Distribution of random numbers \n{n_bins=}")
    ax.set_xlabel("Value Range", fontsize=10, labelpad=8, color='#34495e')
    ax.set_ylabel("Frequency", fontsize=10, labelpad=8, color='#34495e')

    ax.tick_params(axis='both', which='major', labelsize=9, labelcolor='#2c3e50', bottom=True, left=True, color='#cccccc')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

def emp_vs_real(samples, N, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    uni_real = np.linspace(0,1, N)
    real_curve = st.uniform.cdf(uni_real)
    emp_curve = empiric_dist(samples, uni_real)
    ax.plot(uni_real, real_curve, label="real curve")
    ax.plot(uni_real, emp_curve, label="empiric curve")
    ax.set_title(f"Empirirc vs Real cumulative density curve")
    ax.legend()

def visual_corr(samples, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(samples[:-1], samples[1:])
    ax.grid()
    ax.set_title("Visual correlation")
    ax.set_xlabel("U_i")
    ax.set_ylabel("U_i+1")

def plot_corr_test(samples, min_h=1, max_h=100, step_h=1, ax=None):
    """
    p-value > 0.05 means that we reject the Null-Hypothesis being there is a correlation between X_i and X_i+h
    """
    if ax is None:
        fig, ax = plt.subplots()
    H=np.arange(min_h, max_h, step_h)
    Y=[correlation_test(samples, lag=h) for h in H]
    ax.scatter(H, Y)
    ax.set_xlabel("Lag h")
    ax.set_ylabel("p_value")
    ax.set_title("p-value as a function of the lag")
    ax.axhline(y=0.05, color='red', linestyle='--', label='Alpha = 0.05',)
    ax.legend()

def exercice_1_main_plot(samples, N, a=5, c=1, M=16, n_bins=10, systen_samples=False):
    fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    axes = axes.flatten()

    x0 = float(samples[0]*M) if M is not None else None
    period_length = floyd_lcg(lcg_step, x0, (a, c, M)) if M is not None else None
    fig.suptitle(f"{N=}, {a=}, {c=}, {M=}, {x0=}, period length = {period_length}\nChi-squared p-value: {float(chi_square_test(samples, n_bins, N)[0])}\nKolmogorov Smirnov test: D_n (adjusted) {Kolmogorov_Smirnov_test(samples, N)[0]:.3f}, threshold {Kolmogorov_Smirnov_test(samples, N)[1]:.3f},\nWald-Wolfowitz test p_value {run_test_1(samples):.3f}, Knuth’s run-length test p_value {run_test_2(samples):.3f}")
    LCG_hist(samples, N=N, a=a, c=c, M=M, n_bins=n_bins, ax=axes[0])
    emp_vs_real(samples, N, ax=axes[1])
    visual_corr(samples, ax=axes[2])
    plot_corr_test(samples, ax=axes[3])
    plt.tight_layout()
    plt.show()
        
