import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.linalg import expm

# For Task 7
def simulate_ctmc(Q, start_state=0):
    current_state = start_state
    current_time = 0.0
    
    # Trackers
    history_states = [current_state]
    history_times = [current_time]
    
    while current_state != 4:
        exit_rate = -Q[current_state, current_state]
        
        # If exit rate is 0, we hit an absorbing state
        if np.isclose(exit_rate, 0.0):
            history_states.append(current_state)
            break
            
        # Sample holding time in current state
        holding_time = np.random.exponential(1.0 / exit_rate)
        current_time += holding_time
            
        # Next state
        row_rates = Q[current_state, :].copy()
        row_rates[current_state] = 0.0  
        probabilities = row_rates / exit_rate
        
        next_state = np.random.choice(len(Q), p=probabilities)
        
        # Append to history
        history_times.append(current_time)
        history_states.append(next_state)
        
        current_state = next_state
        
    return np.array(history_times), np.array(history_states)

# For Task 8:
def phase_type_cdf(t):
    """Theorical cdf of CTMC"""
    return np.array([1.0 - (p0 @ expm(Qs * time) @ ones)[0] for time in t])
    
def phase_type_pdf(t_vec):
    """Theorical pdf of CTMC"""
    return np.array([(p0 @ expm(Qs * t) @ -Qs @ ones)[0] for t in t_vec])

# For Task 9:
def Kaplan_Meier_est(t, times, N):
    # counting the number of died women at time t
    deaths_by_t = times <= t
    d_t = np.sum(deaths_by_t)
    return (N-d_t)/N

# For Task 10
def log_rank(lifetimes_control, lifetimes_treat):
    # Combine and find unique event times
    all_times = np.concatenate([lifetimes_control, lifetimes_treat])
    unique_times = np.unique(all_times)
    unique_times = np.sort(unique_times)
    
    # Initialize running totals for Observed and Expected events
    O_1 = 0.0  # Observed deaths in Control
    E_1 = 0.0  # Expected deaths in Control
    Var_V = 0.0 # Running variance
    
    for t in unique_times:
        # Count deaths occurring precisely at time t in each group
        d1 = np.sum(lifetimes_control == t)
        d2 = np.sum(lifetimes_treat == t)
        d_total = d1 + d2
        
        if d_total == 0:
            continue
      
        # Count how many individuals are still alive before time t
        n1 = np.sum(lifetimes_control >= t)
        n2 = np.sum(lifetimes_treat >= t)
        n_total = n1 + n2
        
        # If no one is left at risk, stop counting
        if n_total == 0:
            break
            
        # Calculate Expected deaths for Control at this moment
        e1 = n1 * (d_total / n_total)
        
        # Update
        O_1 += d1
        E_1 += e1
        
        # Compute Hypergeometric Variance contribution at this time point
        if n_total > 1:
            v = (n1 * n2 * d_total * (n_total - d_total)) / (n_total**2 * (n_total - 1))
            Var_V += v

    # We compute two different tests just to see the equlity of them aha: chi square and t-test
    Z = (O_1 - E_1) / np.sqrt(Var_V)
    chi2_stat = Z**2
    
    # 1 degree of freedom because we are comparing 2 groups
    p_value_chi = 1.0 - st.chi2.cdf(chi2_stat, df=1)
    p_value_ttest = 2 * st.norm.cdf(-np.abs(Z))
    return p_value_chi, p_value_ttest

if __name__=='__main__':
    # Parameters
    Q = np.array([
        [-0.0085,  0.005,   0.0025,  0.0,     0.001 ],
        [ 0.0,    -0.014,   0.005,   0.004,   0.005 ],
        [ 0.0,     0.0,    -0.008,   0.003,   0.005 ],
        [ 0.0,     0.0,     0.0,    -0.009,   0.009 ],
        [ 0.0,     0.0,     0.0,     0.0,     0.0   ]
    ])
    n_simu = 1000

    ## Task 7:
    lifetimes = []
    distant_recurrence_at_30_5 = 0
    
    for n in range(n_simu):
        times, trajectory = simulate_ctmc(Q, start_state=0)
        
        # Lifetimes
        lifetimes.append(times[-1])
        
        # Check state at t = 30.5 months
        idx = np.searchsorted(times, 30.5, side='right') - 1
        state_at_30_5 = trajectory[idx]
        
        # Distant = state 3 or state 4
        if state_at_30_5 in [2, 3]:
            distant_recurrence_at_30_5 += 1

    lifetimes = np.array(lifetimes)

    # Confidence Interval
    mean_lifetime = np.mean(lifetimes)
    sd_lifetime = np.std(lifetimes, ddof=1)
    
    # 95% CI for Mean
    mean_ci = st.t.interval(0.95, len(lifetimes)-1, loc=mean_lifetime, scale=st.sem(lifetimes))
    
    # 95% CI for Standard Deviation 
    df = n_simu - 1
    chi2_lower = st.chi2.ppf(0.025, df)
    chi2_upper = st.chi2.ppf(0.975, df)
    sd_ci_lower = np.sqrt((df * sd_lifetime**2) / chi2_upper)
    sd_ci_upper = np.sqrt((df * sd_lifetime**2) / chi2_lower)
    
    prop_distant = distant_recurrence_at_30_5 / n_simu

    # Summary
    print("Task 7: LIFETIME DISTRIBUTION SUMMARY -------------")
    print(f"Mean Lifetime: {mean_lifetime:.2f} months")
    print(f"95% CI for Mean: ({mean_ci[0]:.2f}, {mean_ci[1]:.2f})")
    print(f"Standard Deviation: {sd_lifetime:.2f} months")
    print(f"95% CI for SD: ({sd_ci_lower:.2f}, {sd_ci_upper:.2f})")
    print("-" * 37)
    print(f"Proportion of women with distant reappearance at 30.5 months: {prop_distant:.4f} ({prop_distant*100:.2f}%)")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(lifetimes, bins=30, edgecolor='black', color='skyblue', alpha=0.8, density=True)
    plt.axvline(mean_lifetime, color='red', linestyle='dashed', linewidth=2, label=f'{mean_lifetime=:.1f}')
    plt.title(f'Lifetime Distribution After Surgery ({n_simu=} Simulated Women)')
    plt.xlabel('Lifetime (Months)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    ## Task 8:
    # parameters
    Qs = Q[:4, :4]
    p0 = np.array([1.0, 0.0, 0.0, 0.0])
    ones = np.ones((4, 1))

    # Test 
    ks_stat, p_value = st.ks_1samp(lifetimes, phase_type_cdf)
    print("Task 8: KS Test----------------")
    print(f"KS Statistic: {ks_stat:.4f}")
    print(f"P-value:      {p_value:.4f}")
    print("-" * 31)
    
    t_max = np.max(lifetimes)
    t_grid = np.linspace(0, t_max, 500)

    # Theorical CDF
    cdf_theoretical = phase_type_cdf(t_grid)
    pdf_theoretical = phase_type_pdf(t_grid)

    # Empiric CDF
    sorted_lifetimes = np.sort(lifetimes)
    cdf_empirical = np.arange(1, n_simu + 1) / n_simu

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # CDF comparison
    ax1.step(sorted_lifetimes, cdf_empirical, where='post', label='Empirical CDF (Simulation)', color='royalblue', alpha=0.8, lw=2)
    ax1.plot(t_grid, cdf_theoretical, label='Theoretical CDF ($F_T(t)$)', color='crimson', linestyle='--', lw=2)
    ax1.set_title('CDF Comparison: Simulation vs. Theory', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (Months)', fontsize=10)
    ax1.set_ylabel('Cumulative Probability', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)

    # PDF comparison
    ax2.hist(lifetimes, bins=35, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Empirical Density (Histogram)')
    ax2.plot(t_grid, pdf_theoretical, label='Theoretical PDF ($f_T(t)$)', color='crimson', linestyle='--', lw=2)
    ax2.set_title('Density Comparison: Simulation vs. Theory', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (Months)', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

    ## Task 9:
    Q_treatment = np.array([
        [0.,  0.0025, 0.00125, 0., 0.001 ],
        [ 0., 0., 0., 0.002, 0.005 ],
        [ 0., 0., 0., 0.003, 0.005, ],
        [ 0.0,     0.0,     0.0,    0.,   0.009 ],
        [ 0.0,     0.0,     0.0,     0.0,     0.0   ]
    ])
    for i in range(Q_treatment.shape[0]):
        Q_treatment[i,i] = -sum(Q_treatment[i,i+1:])
    print(Q_treatment)

    lifetimes_treat = []
    distant_recurrence_at_30_5 = 0
    
    for n in range(n_simu):
        times_treat, traj_treat = simulate_ctmc(Q, start_state=0)
        
        # Lifetimes for treatment
        lifetimes_treat.append(times_treat[-1])

    lifetimes_treat = np.array(lifetimes_treat)

    # Getting the value of the estimator
    max_lifetime = np.max([np.max(lifetimes_treat), np.max(lifetimes)])
    time_support = np.linspace(0., max_lifetime, 1000)
    KM_treat = [Kaplan_Meier_est(t, lifetimes_treat, n_simu) for t in time_support]
    KM = [Kaplan_Meier_est(t, lifetimes, n_simu) for t in time_support]
    plt.figure(figsize=(10,7))
    plt.step(time_support, KM_treat, where='post', color='#E63946', lw=2.5, label="With Treatment")
    plt.step(time_support, KM, where='post', color='#1D3557', lw=2.5, label="Without Treatment")
    plt.title('Kaplan-Meier estimator of the survival function with and without the treatment')
    plt.xlabel("time")
    plt.ylabel('Estimator values')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()

    ## Task 10:
    p_chi, p_t = log_rank(lifetimes, lifetimes_treat)
    print("Task 10: Is the treatment significant ----------------")
    print(f"t-test pvalue: {p_t:.4f}")
    print(f"chi-test pvalue: {p_chi:.4f}")
    print('Should be >0.05 so the two survival are the same and the treatment has no effect significantly')
    print("-" * 31)



