import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

P = np.array([[0.9915, 0.005, 0.0025, 0, 0.001], 
             [0, 0.986, 0.005, 0.004, 0.005], 
             [0, 0, 0.992, 0.003, 0.005], 
             [0, 0, 0, 0.991, 0.009], 
             [0, 0, 0, 0, 1]])

samples = 1000
max_steps = 1500
num_states = P.shape[0]

def simulate_stages(P, samples, max_steps):
    num_states = P.shape[0]
    absorbing_state = num_states - 1

    trajectories = np.zeros((samples, max_steps), dtype=int)
    lifetimes = np.zeros(samples, dtype=int)

    for s in range(samples):
        state = 0
        for t in range(max_steps):
            trajectories[s, t] = state
            if state == absorbing_state:
                lifetimes[s] = t
                trajectories[s,t:] = 4
                break
            state = np.random.choice(num_states, p=P[state,:])
    return trajectories, lifetimes

def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    
    # t-distribution margin of error
    t_val = stats.t.ppf((1 + confidence) / 2, df=n-1)
    moe = t_val * sem
    
    return mean, mean - moe, mean + moe

# Task 3: Confirming the simulation follows the distribution
P_s = np.array([[0.9915, 0.005, 0.0025, 0], 
             [0, 0.986, 0.005, 0.004], 
             [0, 0, 0.992, 0.003], 
             [0, 0, 0, 0.991]])

# Taking the first four values of the last column of P
p_s = np.array([0.001, 0.005, 0.005, 0.009])

p_0 = np.array([1, 0, 0, 0])

# Discrete phase-type distribution
DPT = np.zeros(max_steps, dtype=float)
for t in range(max_steps):
    DPT[t] = (p_0 @ np.linalg.matrix_power(P_s,t) @ p_s)

ones = np.ones(num_states-1)
N = np.linalg.inv(np.identity(num_states-1) - P_s)
# Mean 
E = p_0 @ N @ ones

# Calculate CIs across 10 simulation data points
ci_obs = []
for _ in range(10):
    _, lifetimes = simulate_stages(P,samples,max_steps)
    ci_obs.append(np.mean(lifetimes))

mean_m, lower_m, upper_m = confidence_interval(ci_obs)

print(f'DPT Mean: {E:.1f} steps')
print(f'Confidence interval 95%: ({lower_m:.1f}, {upper_m:.1f})') 
print(f"Mean simulated: {lifetimes.mean():.1f} steps")
print(f"Median lifetime: {np.median(lifetimes):.1f} steps")

# Theoretical CDF:
cdf_theory = np.cumsum(DPT)

fig, (ax2, ax) = plt.subplots(1, 2, figsize=(14, 5))

sorted_obs = np.sort(lifetimes)
ax2.plot(range(max_steps)[:max(sorted_obs)], cdf_theory[:max(sorted_obs)], linestyle='--', linewidth=1.5, label='Theoretical CDF')

ecdf = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs)
ax2.step(sorted_obs, ecdf, alpha=0.6, linewidth=1, label='Empirical CDF (Simulation)')
ax2.set_xlabel("Time (Months)")
ax2.set_ylabel("Cumulative Probability")
ax2.set_title("CDF comparison: Simulation vs. Theory")
ax2.legend()

ax.hist(lifetimes, bins=50, edgecolor='black', density=True, label='Empirical Density')
ax.plot(DPT, color="#D83030", linestyle='-', linewidth=1.5, label='Discrete phase-type distribution')
ax.set_xlabel("Time (Months)")
ax.set_ylabel("Density")
ax.set_title("Density comparison: Simulation vs. Theory")
ax.legend()

plt.tight_layout()
plt.show()