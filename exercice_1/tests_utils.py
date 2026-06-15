import numpy as np
import scipy.stats as st
import math

def chi_square_test(x, n_classes, N):
    """
    Under hypothesis x follows a Uniform(0,1) -> we want not to reject the hypothesis, p_value>0.05
    Return the p_value and True if we not reject the hypothesis.
    """
    x = np.asarray(x)
    
    n_expected = np.full(n_classes, N / n_classes)
    interval = 1 / n_classes

    if N/n_classes < 5:
        print("n_expected is lower than 5, not ideal")
    indices = np.clip((x // interval).astype(int), 0, n_classes - 1)
    n_observed = np.bincount(indices, minlength=n_classes)
    
    # Create the T statistic and get the p-value
    T = np.sum((n_observed - n_expected)**2 / n_expected)
    dof = n_classes - 1
    p_value = st.chi2.sf(T, df=dof)

    return p_value, p_value > 0.05

def empiric_dist(samples, x):
    """
    Computes the Empirical CDF of 'samples' evaluated at each point in 'x'.
    Both inputs should be 1D arrays or lists.
    """
    samples = np.asarray(samples)
    x = np.asarray(x)
    
    # Broadcasting trick
    indicator_matrix = samples[:, None] <= x
    return np.mean(indicator_matrix, axis=0)
    
def Kolmogorov_Smirnov_test(samples, N):
    uni_real = np.linspace(0,1, N)
    real_curve = st.uniform.cdf(uni_real)
    emp_curve = empiric_dist(samples, uni_real)
    
    absolute_differences = np.abs(emp_curve - real_curve)
    D_n = np.max(absolute_differences)
    adjusted_D_n = (math.sqrt(N)+0.12+0.11/math.sqrt(N))*D_n
    threshold = 1.358 # From the slides of testing 
    return adjusted_D_n, threshold

def run_test_1(samples):
    samples = np.asarray(samples)
    
    # Calculate the median of the dataset
    median_val = np.median(samples)
    
    # Binarize the data: +1 for strictly above median, -1 for strictly below median
    filtered_samples = samples[samples != median_val]
    binary_seq = np.where(filtered_samples > median_val, 1, -1)
    
    # Count sizes of both partitions
    n1 = np.sum(binary_seq == 1)   # Number of samples above
    n2 = np.sum(binary_seq == -1)  # Number of samples below

    n1 = np.float64(n1)
    n2 = np.float64(n2)
    
    # Calculate total observed runs (T)
    run_changes = np.diff(binary_seq) != 0
    T = np.sum(run_changes) + 1
    
    # Expected mean
    mean_T = (2 * n1 * n2) / (n1 + n2) + 1
    
    # Expected variance
    numerator_var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)
    denominator_var = ((n1 + n2) ** 2) * (n1 + n2 - 1)
    var_T = numerator_var / denominator_var
    
    # Calculate Z-score and Two-Sided p-value
    std_T = np.sqrt(var_T)
    Z = (T - mean_T) / std_T
    p_value = 2 * (1 - st.norm.cdf(np.abs(Z)))

    return float(p_value)

def run_test_2(samples):
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    
    if n <= 4000:
        print(f"Warning: n = {n}. The slide recommends n > 4000.")

    R = np.zeros(6)

    # logic is to increment the run_length while the trend doesn't change
    current_run_length = 1    
    
    for i in range(1, n):
        if samples[i] > samples[i-1]:
            current_run_length += 1
        else:
            if current_run_length >= 6:
                R[5] += 1
            else:
                R[current_run_length - 1] += 1
            current_run_length = 1
    if current_run_length >= 6:
        R[5] += 1
    else:
        R[current_run_length - 1] += 1
            
    # Matrix A from slide
    A = np.array([
        [4529.4, 9044.9, 13568, 18091, 22615, 27892],
        [9044.9, 18097,  27139, 36187, 45234, 55789],
        [13568,  27139,  40721, 54281, 67852, 83685],
        [18091,  36187,  54281, 72414, 90470, 111580],
        [22615,  45234,  67852, 90470, 113262, 139476],
        [27892,  55789,  83685, 111580, 139476, 172860]
    ])
    
    # Vector B from slide
    B = np.array([1/6, 5/24, 11/120, 19/720, 29/5040, 1/840])
    
    # Compute Test Statistic Z 
    diff_vector = R - (n * B)
    Z = (1 / (n - 6)) * (diff_vector.T @ A @ diff_vector)
    
    # p-value
    p_value = st.chi2.sf(Z, df=6)
    
    return float(p_value)
    

def correlation_test(samples, lag=1):
    samples = np.asarray(samples)
    n = len(samples)
    h = lag

    # Compute the estimated correlation coefficient c_h
    u_i = samples[:n-h]
    u_ih = samples[h:]
    c_h = np.sum(u_i * u_ih) / (n - h)
    
    # Get theoretical Normal distribution parameters from the slide
    mean_theoretical = 0.25
    var_theoretical = 7 / (144 * n)
    std_theoretical = np.sqrt(var_theoretical)
    
    # Calculate the Z-score
    Z = (c_h - mean_theoretical) / std_theoretical
    
    # Compute the two-sided p-value
    p_value = 2 * (1 - st.norm.cdf(np.abs(Z)))
    
    return float(p_value)

# Looking for the period length of the finite field (Implementation from Wikipedia: https://en.wikipedia.org/wiki/Cycle_detection)
def lcg_step(x, params):
    """
    Computes the next state in a Linear Congruential Generator.
    params is expected to be a tuple or list: (a, c, M)
    """
    a, c, M = params
    return (a * x + c) % M

def floyd_lcg(f, x0, params, max_iter=10000):
    """Floyd's cycle detection algorithm adapted for a parameterized function. For computation complexity we stop the algorithm after max_iter iterations"""
    
    # The hare moves twice as fast as the tortoise
    tortoise = f(x0, params)
    hare = f(f(x0, params), params)
    iterations = 1
    
    while tortoise != hare:
        if iterations >= max_iter:
            return f"greater than {max_iter}"
        
        tortoise = f(tortoise, params)
        hare = f(f(hare, params), params)
        iterations += 1
  
    # Find the position μ of the first repetition
    mu = 0
    tortoise = x0
    while tortoise != hare:
        tortoise = f(tortoise, params)
        hare = f(hare, params)   # Both move at the same speed
        mu += 1
 
    # Find the length λ of the shortest cycle
    lam = 1
    hare = f(tortoise, params)
    while tortoise != hare:
        hare = f(hare, params)
        lam += 1
 
    return lam

