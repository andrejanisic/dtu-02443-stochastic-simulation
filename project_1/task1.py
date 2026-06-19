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
absorbing_state = num_states - 1
local_recurrence_state = 1
p0 = [1,0,0,0,0]

# Task 1: Simulate the distribution
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

trajectories, lifetimes = simulate_stages(P, samples,max_steps)

# Proportion of women which cancer eventually locally
prop = np.zeros(10)
for o in range(10):
    trajectories, _ = simulate_stages(P, samples,max_steps)
    for i in range(samples):
        for j in range(max_steps):
            if trajectories[i, j] != 0:
                if trajectories[i,j] == local_recurrence_state:
                    prop[o] += 1
                    break

# Confidence intervals
ci_mean = []
ci_std = []
for _ in range(10):
    _, lifetimes = simulate_stages(P,samples,max_steps)
    ci_mean.append(np.mean(lifetimes))
    ci_std.append(np.std(lifetimes))

def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    
    # t-distribution margin of error
    t_val = stats.t.ppf((1 + confidence) / 2, df=n-1)
    moe = t_val * sem
    
    return mean, mean - moe, mean + moe

# Calculate CIs across 10 simulation data points
mean_m, lower_m, upper_m = confidence_interval(ci_mean)
mean_s, lower_s, upper_s = confidence_interval(ci_std)

print(f"Mean lifetime: {mean_m:.1f} months; 95% Confidence interval: [{lower_m:.1f}, {upper_m:.1f}]")
print(f"Standard deviation lifetime: {mean_s:.1f} months; 95% Confidence interval: [{lower_s:.1f}, {upper_s:.1f}]")
print(f'Cancer appeared locally in {np.mean(prop)} of 1000 women')

fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.hist(lifetimes, bins=50, edgecolor='black')
ax.axvline(lifetimes.mean(), color='#D85A30', linestyle='-', linewidth=1.5, label=f'Mean = {lifetimes.mean():.2f}')
ax.set_xlabel("Months until death (post-surgery)")
ax.set_ylabel("Number of patients")
ax.set_title("Lifetime distribution after surgery")
ax.legend()
plt.show()