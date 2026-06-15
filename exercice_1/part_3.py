import numpy as np
import matplotlib.pyplot as plt

from exercice_1.plot_utils import *
from exercice_1.tests_utils import *

def LCG_max(x0, a=5, c=1, M=16):
    res = []
    x = x0
    while x not in res:
        res.append(x)
        x = (a * x + c) % M
    return np.array(res).shape[0]

def LCG(N, x0, a=5, c=1, M=16):
    res = [x0]
    x = x0
    for i in range(N-1):
        x = (a * x + c) % M
        res.append(x)
    return np.array(res) / M

if __name__ == "__main__":
    K = 100 # Number of sub-samples (adjust as needed, e.g., 100 or 1000)
    N = 10000  # Sample size per sub-sample
    n_bins = 10
    
    # Define setups for our 3 generators: (a, c, M, initial_seed)
    generators = {
        "bad":  {"a": 123,   "c": 1, "M": 2**10 - 1, "seed": 2},
        "good": {"a": 48271, "c": 0, "M": 2**31 - 1, "seed": 2}
    }
    
    test_names = ["chi", "KS", "Wald", "RunLength"]
    results = {gen: {test: [] for test in test_names} for gen in ["bad", "good", "numpy"]}

    print(f"Running simulation over K = {K} sub-samples...")

    for k in range(K):
        # Evaluate "Bad" and "Good" LCGs
        for gen_name, params in generators.items():
            current_seed = params["seed"] + k * 12345  
            sample = LCG(N, current_seed, params["a"], params["c"], params["M"])
            
            # Extract p-values from your utility functions
            p_chi = chi_square_test(sample, n_classes=10, N=N)[0]
            p_ks = Kolmogorov_Smirnov_test(sample, N)
            p_wald = run_test_1(sample)   
            p_run = run_test_2(sample)      
            
            results[gen_name]["chi"].append(p_chi)
            results[gen_name]["KS"].append(p_ks)
            results[gen_name]["Wald"].append(p_wald)
            results[gen_name]["RunLength"].append(p_run)
            
        # Numpy
        system_samples = np.random.uniform(0.0, 1.0, N)

        p_chi_np = chi_square_test(system_samples, n_classes=10, N=N)[0]
        p_ks_np = Kolmogorov_Smirnov_test(system_samples, N)
        p_wald_np = run_test_1(system_samples)
        p_run_np = run_test_2(system_samples)

        results["numpy"]["chi"].append(p_chi_np)        
        results["numpy"]["KS"].append(p_ks_np)
        results["numpy"]["Wald"].append(p_wald_np)
        results["numpy"]["RunLength"].append(p_run_np)

    print("\n" + "="*60)
    print(f"{'Statistical Test':<20} | {'Bad LCG':<10} | {'Good LCG':<10} | {'NumPy':<10}")
    print("="*60)
    for test in test_names:
        mean_bad = np.mean(results["bad"][test])
        mean_good = np.mean(results["good"][test])
        mean_np = np.mean(results["numpy"][test])
        print(f"{test:<20} | {mean_bad:<10.3f} | {mean_good:<10.3f} | {mean_np:<10.3f}")
    print("="*60)