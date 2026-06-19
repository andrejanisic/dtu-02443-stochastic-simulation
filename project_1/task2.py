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

trajectories, _ = simulate_stages(P, samples, max_steps)

# Task 2
# Expected distribution at t=120
p0 = np.zeros(num_states)
p0[0] = 1  # start in stage 0

exp_distribution = p0 @ np.linalg.matrix_power(P,119)

# Simulated distribution at t = 120
# index starting at 0
t_check = 119
counts = np.zeros(num_states)

for s in range(num_states):
    for j in range(samples):
        if trajectories[j, t_check] == s:
            counts[s] += 1

distribution = counts / samples

expected = exp_distribution * samples

print(f'Simulated distribution: {distribution}')
print(f'Expected distribution: {exp_distribution}')

# Chi-square test
chi2, pval = stats.chisquare(counts, expected)
print(f'Chi-squared: {chi2:.2f} p-value: {pval:.2}')

x = np.arange(5)
width = 0.4

plt.bar(x - width/2, counts,   width, color="#3062D8", label='Simulation')
plt.bar(x + width/2, expected, width, color="#D83030", label='Theory')

plt.xticks(x, ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5'])
plt.ylabel('Number of women')
plt.title('Distribution at t=120')
plt.grid()
plt.legend()
plt.show()