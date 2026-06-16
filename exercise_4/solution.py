"""
Write a discrete event simulation program for a blocking system,
i.e. a system with m service units and no waiting room.
The oﬀered traﬃc A is the product of the mean arrival rate and the mean service time.

Part 1:
The arrival process is modelled as a Poisson process. Report the fraction of blocked customers, and a confidence interval for this fraction.
Choose the service time distribution as exponential.
Parameters: m = 10, mean service time = 8 time units, mean
time between customers = 1 time unit (corresponding to an
oﬀered traﬃc of 8 Erlang), 10 x 10.000 customers.

This system is suﬃciently simple such that the analytical solution is known. See the last slide for the solution.
Verify your simulation program using this knowledge.

Part 2:
The arrival process is modelled as a renewal process using the same parameters as in Part 1 when possible. Report the
fraction of blocked customers, and a confidence interval for this fraction for at least the following two cases:
(a) Experiment with Erlang distributed inter arrival times The Erlang distribution should have a mean of 1
(b) hyper exponential inter arrival times. The parameters for
the hyper exponential distribution should be p1 = 0.8, λ1 = 0.8333, p2 = 0.2, λ2 = 5.0.

Part 3:
The arrival process is again a Poisson process like in Part 1.
Experiment with diﬀerent service time distributions with the same mean service time and m as in Part 1 and Part 2.
(a) Constant service time
(b) Pareto distributed service times with at least k = 1.05 and k = 2.05
(c) Choose one or two different distributions.

Part 4:
Compare confidence intervals for Parts 1, 2, and 3 then interpret and explain differences if any. 
"""

import numpy as np
from scipy import stats


# arrival time generators

def poisson_arrivals(n, rng, mean=1.0):
    return np.cumsum(rng.exponential(scale=mean, size=n))

def erlang_arrivals(n, rng, k, mean=1.0):
    # Gamma(shape=k, scale=mean/k) equivalent to Erlang(k) with given mean
    return np.cumsum(rng.gamma(shape=k, scale=mean/k, size=n))

def hyperexp_arrivals(n, rng, p1, lambda1, p2, lambda2):
    u = rng.uniform(low=0.0, high=1.0, size=n)
    samples = np.where(
        u < p1,
        rng.exponential(scale=1/lambda1, size=n),
        rng.exponential(scale=1/lambda2, size=n)
    )
    return np.cumsum(samples)


# service time generators

def exponential_service(n, rng, mean=8.0):
    return rng.exponential(scale=mean, size=n)

def constant_service(n, rng, mean=8.0):
    return np.full(n, mean)

def pareto_service(n, rng, k, mean=8.0):
    # Standard Pareto: X = x_m * (1 + Pareto_II(k))
    # E[X] = x_m * k/(k-1)  =>  x_m = mean*(k-1)/k
    x_m = mean * (k - 1) / k
    return x_m * (1.0 + rng.pareto(k, size=n))

def erlang_service(n, rng, k, mean=8.0):
    return rng.gamma(shape=k, scale=mean/k, size=n)


#Simulation

def simulate_batch(arrival_times, m_servers, service_gen, rng):
    n_customers = len(arrival_times)
    service_times = service_gen(n_customers, rng)
    departure_times = np.zeros(m_servers)
    n_blocked = 0

    for i, t in enumerate(arrival_times):
        earliest_free = np.min(departure_times)
        if earliest_free <= t:
            server_idx = np.argmin(departure_times)
            departure_times[server_idx] = t + service_times[i]
        else:
            n_blocked += 1

    return n_blocked / n_customers

def run_simulation(n_batches, n_per_batch, m_servers, service_gen, arrival_gen):
    rng = np.random.default_rng(seed=42)
    blocking_fractions = []

    for i in range(n_batches):
        arrival_times = arrival_gen(n_per_batch, rng)
        b = simulate_batch(arrival_times, m_servers, service_gen, rng)
        blocking_fractions.append(b)
        print(f"  Batch {i+1:2d}: blocking = {b:.4f}")

    return np.array(blocking_fractions)

def confidence_interval(estimates, alpha=0.05):
    n = len(estimates)
    mean = np.mean(estimates)
    std = np.std(estimates, ddof=1)
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    margin = t_crit * std / np.sqrt(n)
    return mean, mean - margin, mean + margin

def erlang_b(A, m):
    b = 1.0
    for i in range(1, m + 1):
        b = (A * b) / (i + A * b)
    return b


def main():
    n_customers = 10000
    m_servers = 10
    mean_service_time = 8
    mean_interarrival = 1
    n_batches = 10
    A = mean_service_time / mean_interarrival
    erlang = erlang_b(A=A, m=m_servers)

    poisson_gen = lambda n, rng: poisson_arrivals(n, rng, mean=mean_interarrival)
    exp_svc = lambda n, rng: exponential_service(n, rng, mean=mean_service_time)

    # results collected for Part 4 summary: (label, cv_arr, cv_svc, mean, lower, upper)
    summary = []

    # Part 1: Poisson arrivals, exponential service ---
    print("=" * 45)
    print("  Part 1: Poisson arrivals, Exp service")
    print("=" * 45)
    bf = run_simulation(n_batches, n_customers, m_servers, exp_svc, poisson_gen)
    mean_b, lower_b, upper_b = confidence_interval(bf)
    print("-" * 45)
    print(f"  Mean blocking fraction:  {mean_b:.4f}")
    print(f"  95% CI:                  [{lower_b:.4f}, {upper_b:.4f}]")
    print(f"  Erlang B (exact):        {erlang:.4f}")
    print("=" * 45)
    summary.append(("P1: Poisson arr, Exp svc",    "1.000", "1.000", mean_b, lower_b, upper_b))

    # Part 2a: Erlang inter-arrivals ---
    print("\n" + "=" * 45)
    print("  Part 2a: Erlang inter-arrivals")
    print("=" * 45)
    for k in [2, 5, 10]:
        cv = 1 / np.sqrt(k)
        print(f"\n  k={k}  (CV={cv:.3f})")
        print("-" * 45)
        gen = lambda n, rng, k=k: erlang_arrivals(n, rng, k=k, mean=mean_interarrival)
        bf = run_simulation(n_batches, n_customers, m_servers, exp_svc, gen)
        mean_b, lower_b, upper_b = confidence_interval(bf)
        print("-" * 45)
        print(f"  Mean blocking fraction:  {mean_b:.4f}")
        print(f"  95% CI:                  [{lower_b:.4f}, {upper_b:.4f}]")
        print(f"  Erlang B (Poisson ref):  {erlang:.4f}")
        summary.append((f"P2a: Erlang arr k={k}",  f"{cv:.3f}", "1.000", mean_b, lower_b, upper_b))
    print("=" * 45)

    # Part 2b: Hyper-exponential inter-arrivals ---
    p1, lam1, p2, lam2 = 0.8, 0.8333, 0.2, 5.0
    cv_hyperexp = np.sqrt(p1*(2/lam1**2) + p2*(2/lam2**2) - 1.0)
    print("\n" + "=" * 45)
    print("  Part 2b: Hyper-exponential inter-arrivals")
    print(f"  p1={p1}, λ1={lam1}, p2={p2}, λ2={lam2}  CV={cv_hyperexp:.3f}")
    print("=" * 45)
    gen = lambda n, rng: hyperexp_arrivals(n, rng, p1=p1, lambda1=lam1, p2=p2, lambda2=lam2)
    bf = run_simulation(n_batches, n_customers, m_servers, exp_svc, gen)
    mean_b, lower_b, upper_b = confidence_interval(bf)
    print("-" * 45)
    print(f"  Mean blocking fraction:  {mean_b:.4f}")
    print(f"  95% CI:                  [{lower_b:.4f}, {upper_b:.4f}]")
    print(f"  Erlang B (Poisson ref):  {erlang:.4f}")
    print("=" * 45)
    summary.append(("P2b: Hyper-exp arr",          f"{cv_hyperexp:.3f}", "1.000", mean_b, lower_b, upper_b))

    # Part 3: Poisson arrivals, varying service distribution ---
    print("\n" + "=" * 55)
    print("  Part 3: Poisson arrivals, varying service distribution")
    print(f"  Erlang B (reference): {erlang:.4f}")
    print("=" * 55)

    # Part 3a: Constant service time
    print("\n  (a) Constant service time")
    print("  CV_svc = 0.000")
    print("-" * 55)
    svc = lambda n, rng: constant_service(n, rng, mean=mean_service_time)
    bf = run_simulation(n_batches, n_customers, m_servers, svc, poisson_gen)
    mean_b, lower_b, upper_b = confidence_interval(bf)
    print("-" * 55)
    print(f"  Mean blocking fraction:  {mean_b:.4f}")
    print(f"  95% CI:                  [{lower_b:.4f}, {upper_b:.4f}]")
    print(f"  Erlang B (reference):    {erlang:.4f}")
    summary.append(("P3a: Constant svc",           "1.000", "0.000", mean_b, lower_b, upper_b))

    # Part 3b: Pareto service times
    for k in [2.05, 1.05]:
        if k > 2:
            # Pareto(shape k): CV = 1/sqrt(k*(k-2)) for k>2, infinite for k<=2
            cv_svc = f"{1/np.sqrt(k*(k-2)):.3f}"
        else:
            cv_svc = "inf"
        print(f"\n  (b) Pareto service, k={k}  CV_svc={cv_svc}")
        print("-" * 55)
        svc = lambda n, rng, k=k: pareto_service(n, rng, k=k, mean=mean_service_time)
        bf = run_simulation(n_batches, n_customers, m_servers, svc, poisson_gen)
        mean_b, lower_b, upper_b = confidence_interval(bf)
        print("-" * 55)
        print(f"  Mean blocking fraction:  {mean_b:.4f}")
        print(f"  95% CI:                  [{lower_b:.4f}, {upper_b:.4f}]")
        print(f"  Erlang B (reference):    {erlang:.4f}")
        summary.append((f"P3b: Pareto svc k={k}", "1.000", cv_svc, mean_b, lower_b, upper_b))

    # Part 3c: Erlang service times
    for k in [2, 5]:
        cv_svc = f"{1/np.sqrt(k):.3f}"
        print(f"\n  (c) Erlang service, k={k}  CV_svc={cv_svc}")
        print("-" * 55)
        svc = lambda n, rng, k=k: erlang_service(n, rng, k=k, mean=mean_service_time)
        bf = run_simulation(n_batches, n_customers, m_servers, svc, poisson_gen)
        mean_b, lower_b, upper_b = confidence_interval(bf)
        print("-" * 55)
        print(f"  Mean blocking fraction:  {mean_b:.4f}")
        print(f"  95% CI:                  [{lower_b:.4f}, {upper_b:.4f}]")
        print(f"  Erlang B (reference):    {erlang:.4f}")
        summary.append((f"P3c: Erlang svc k={k}", "1.000", cv_svc, mean_b, lower_b, upper_b))

    print("=" * 55)

    # Part 4: Comparison of confidence intervals
    print("\n" + "=" * 75)
    print("  Part 4: Comparison of 95% Confidence Intervals")
    print(f"  Erlang B (reference) = {erlang:.4f}    A = {A} Erlang,  m = {m_servers}")
    print("=" * 75)
    print(f"  {'Experiment':<28} {'CV_arr':>6} {'CV_svc':>6}  {'Mean':>6}  {'95% CI':^20}")
    print("-" * 75)
    for label, cv_arr, cv_svc, mean_b, lower_b, upper_b in summary:
        ci_str = f"[{lower_b:.4f}, {upper_b:.4f}]"
        print(f"  {label:<28} {cv_arr:>6} {cv_svc:>6}  {mean_b:.4f}  {ci_str:^20}")
    print("=" * 75)



if __name__ == "__main__":
    main()
