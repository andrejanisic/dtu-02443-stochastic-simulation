import numpy as np
import matplotlib.pyplot as plt

from exercice_1.plot_utils import *
from exercice_1.tests_utils import *

def LCG_max(x0, a=5, c=1, M=16):
    res = []
    x=x0
    while x not in res:
        res.append(x)
        x = (a*x +c) % M
    
    return np.array(res).shape[0]

#print(LCG_max(3, 5, 1, 16)) # example of the slide 17

def LCG(N, x0, a=5, c=1, M=16):
    res = [x0]
    x=x0
    for i in range(N-1):
        x = (a*x +c) % M
        res.append(x)
    return np.array(res)/M

if __name__=="__main__":
    # LCG implementation
    # print(LCG(10, 3, 5, 1, 16)) # example of the slide 17

    # a
    N,x0=10**4, 2
    a=123
    c=1
    M=2**10-1

    """N,x0=10**4, 2
    a=48271
    c=0
    M=2**31-1
    """
    n_bins = 10

    res_a = LCG(N, x0, a, c, M)
    #LCG_max(x0, a=a, c=c, M=M)
    #LCG_hist(res_a, n_bins=n_bins, a=a, N=N, c=c, M=M)

    # b
    #print(chi_square_test(res_a, n_bins, N))
    # print(empiric_dist_atom([5,1,2,3,2,3,4], 3))
    #emp_vs_real(res_a, N)
    #print(Kolmogorov_Smirnov_test(res_a, N))
    #print(visual_corr(res_a))
    #print(run_test_1(res_a))
    #print(run_test_2(res_a))
    #plot_corr_test(res_a)
    #exercice_1_main_plot(res_a, N, a=a, c=c, M=M)

    ## System available: np.random
    np.random.seed(42)
    #system_samples = np.random.uniform(0.0, 1.0, N)
    #exercice_1_main_plot(system_samples, N, a=None, c=None, M=None)

    ## Visual evidence of the uniformity of the three different generators     
    M_reps = 100     

    p_values_model1 = []
    p_values_model2 = []
    p_values_system = []

    # We vary the seed (x0) across iterations to get unique sequences
    for i in range(M_reps):
        current_seed = i + 2  
        
        # Model 1: Bad LCG
        res_m1 = LCG(N, x0=current_seed, a=123, c=1, M=2**10-1)
        p_values_model1.append(chi_square_test(res_m1, n_bins, N)[0])
        
        # Model 2: Good LCG 
        res_m2 = LCG(N, x0=current_seed, a=48271, c=0, M=2**31-1)
        p_values_model2.append(chi_square_test(res_m2, n_bins, N)[0])
        
        # Model 3: System NumPy Generator
        res_sys = np.random.uniform(0.0, 1.0, N)
        p_values_system.append(chi_square_test(res_sys, n_bins, N)[0])

    # Sort P-value and plot
    p_values_model1 = np.sort(p_values_model1)
    p_values_model2 = np.sort(p_values_model2)
    p_values_system = np.sort(p_values_system)

    expected_quantiles = np.linspace(0, 1, M_reps)


    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    models_data = [
        (p_values_model1, "LCG ($M=2^{10}-1$)"),
        (p_values_model2, "LCG ($M=2^{31}-1$)"),
        (p_values_system, "NumPy Uniform Engine")
    ]

    for idx, (p_vals, title) in enumerate(models_data):
        ax = axes[idx]
        
        # Plot the 45-degree ideal Uniform(0,1) baseline reference
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Uniform(0,1)')
        
        # Scatter the observed ordered p-values
        ax.scatter(expected_quantiles, p_vals, color='#1f77b4', s=15, zorder=3)
        
        # Aesthetics adjustments to match your screenshot layout precisely
        ax.set_title(f"Ordered p-values ({title})", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("iterations (sorted by p_value)", fontsize=10)
        ax.set_ylabel("Observed p-values", fontsize=10)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, linestyle='-', alpha=0.5)
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
   
        