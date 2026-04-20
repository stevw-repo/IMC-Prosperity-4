import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product

# ------------------------------
# 1. Constants & Pillar functions
# ------------------------------
BUDGET = 50_000
MAX_PERCENT = 100

def research(x):
    """Research outcome for x% invested (0..100)."""
    return 200_000 * np.log(1 + x) / np.log(1 + 100)

def scale(x):
    """Scale outcome for x% invested (0..100)."""
    return 7 * x / 100

# ------------------------------
# 2. Speed multiplier simulation
# ------------------------------
def compute_speed_multiplier(own_speed, other_speeds):
    all_speeds = np.append(other_speeds, own_speed)
    # Sort descending, maintain ties
    sorted_unique, counts = np.unique(all_speeds, return_counts=True)
    sorted_unique = sorted_unique[::-1]  # descending
    # Build rank mapping
    rank = 1
    rank_dict = {}
    for val in sorted_unique:
        rank_dict[val] = rank
        rank += 1
    own_rank = rank_dict[own_speed]
    N = len(all_speeds)
    if N == 1:
        return 0.9
    multiplier = 0.9 - 0.8 * (own_rank - 1) / (N - 1)
    return multiplier

def simulate_speed_multipliers(n_other=20000, n_sims=10_000, dist='uniform',
                               low=0, high=100, seed=42):
    np.random.seed(seed)
    mean_mult = np.zeros(MAX_PERCENT + 1)
    std_mult = np.zeros(MAX_PERCENT + 1)

    for own in range(MAX_PERCENT + 1):
        mults = np.zeros(n_sims)
        for i in range(n_sims):
            if dist == 'uniform':
                others = np.random.randint(low, high+1, size=n_other)
            elif dist == 'normal':
                mu, sigma = 50, 25
                others = np.random.normal(mu, sigma, n_other).round().astype(int)
                others = np.clip(others, 0, 100)
            mults[i] = compute_speed_multiplier(own, others)
        mean_mult[own] = np.mean(mults)
        std_mult[own] = np.std(mults)
    return mean_mult, std_mult

# ------------------------------
# 3. Expected PnL and Risk
# ------------------------------
def expected_pnl(r, s, sp, mean_mult):
    return research(r) * scale(s) * mean_mult[sp] - 500 * (r + s + sp)

def risk_pnl(r, s, sp, std_mult):
    return research(r) * scale(s) * std_mult[sp]

# ------------------------------
# 4. Optimisation: min risk for given target expected PnL
# ------------------------------
def find_min_risk_allocation(target_pnl, mean_mult, std_mult, step=1):
    """
    Brute-force over all integer allocations (0..100) with sum ≤ 100.
    Returns (best_allocation, best_risk, best_expected_pnl)
    """
    best_alloc = None
    best_risk = np.inf
    best_exp_pnl = -np.inf

    # Precompute research and scale arrays
    res_vals = np.array([research(r) for r in range(101)])
    sc_vals = np.array([scale(s) for s in range(101)])

    # Loop over all combinations with sum ≤ 100
    for r in range(101):
        for s in range(101):
            if r + s > 100:
                break
            sp_max = 100 - r - s
            for sp in range(sp_max + 1):
                exp_pnl = res_vals[r] * sc_vals[s] * mean_mult[sp] - 500*(r+s+sp)
                if exp_pnl >= target_pnl:
                    risk = res_vals[r] * sc_vals[s] * std_mult[sp]
                    if risk < best_risk:
                        best_risk = risk
                        best_exp_pnl = exp_pnl
                        best_alloc = (r, s, sp)

    return best_alloc, best_risk, best_exp_pnl

# ------------------------------
# 5. Visualisation
# ------------------------------
def plot_pareto_frontier(mean_mult, std_mult):
    """Scatter plot of risk vs expected PnL for all feasible allocations."""
    res_vals = np.array([research(r) for r in range(101)])
    sc_vals = np.array([scale(s) for s in range(101)])

    points = []  # (risk, exp_pnl, r, s, sp)
    for r in range(0, 101, 2):      # step 2 for speed
        for s in range(0, 101, 2):
            if r + s > 100:
                break
            sp_max = 100 - r - s
            for sp in range(0, sp_max+1, 2):
                exp_pnl = res_vals[r] * sc_vals[s] * mean_mult[sp] - 500*(r+s+sp)
                risk = res_vals[r] * sc_vals[s] * std_mult[sp]
                points.append((risk, exp_pnl, r, s, sp))

    points = np.array(points)
    risks = points[:,0]
    pnls = points[:,1]

    # Find Pareto frontier (max pnl for given risk)
    sorted_idx = np.argsort(risks)
    risks_sorted = risks[sorted_idx]
    pnls_sorted = pnls[sorted_idx]
    pareto_front = []
    max_pnl = -np.inf
    for risk, pnl in zip(risks_sorted, pnls_sorted):
        if pnl > max_pnl:
            pareto_front.append((risk, pnl))
            max_pnl = pnl
    pareto_front = np.array(pareto_front)

    plt.figure(figsize=(10,6))
    plt.scatter(risks, pnls, c='blue', alpha=0.3, s=10, label='Allocations')
    plt.plot(pareto_front[:,0], pareto_front[:,1], 'r-', linewidth=2, label='Pareto frontier')
    plt.xlabel('Risk (Std Dev of PnL)')
    plt.ylabel('Expected PnL')
    plt.title('Risk-Return Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_multiplier_stats(mean_mult, std_mult):
    """Plot mean and std of speed multiplier vs investment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    ax1.plot(range(101), mean_mult)
    ax1.set_xlabel('Speed investment (%)')
    ax1.set_ylabel('Expected multiplier')
    ax1.set_title('Mean multiplier vs Speed %')
    ax1.grid(True)

    ax2.plot(range(101), std_mult)
    ax2.set_xlabel('Speed investment (%)')
    ax2.set_ylabel('Std Dev of multiplier')
    ax2.set_title('Multiplier volatility vs Speed %')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------
# 6. Main execution example
# ------------------------------
if __name__ == "__main__":
    # Simulate multiplier statistics
    print("Simulating speed multiplier distribution (this may take ~10 seconds)...")
    mean_mult, std_mult = simulate_speed_multipliers(
        n_other=1_000, n_sims=10_000, dist='uniform'
    )

    # Plot multiplier stats
    plot_multiplier_stats(mean_mult, std_mult)

    # Plot risk-return landscape
    plot_pareto_frontier(mean_mult, std_mult)

    # Find optimal allocation for a given target PnL
    target = 160_000   # Example target PnL (adjust as needed)
    best_alloc, best_risk, best_exp = find_min_risk_allocation(
        target, mean_mult, std_mult
    )

    if best_alloc is not None:
        print(f"\nTarget Expected PnL: {target:,.0f}")
        print(f"Optimal allocation (R%, S%, Sp%): {best_alloc}")
        print(f"Achieved Expected PnL: {best_exp:,.0f}")
        print(f"Minimised Risk (Std Dev): {best_risk:,.0f}")
        print(f"Budget used: {500*sum(best_alloc):,.0f} XIRECs")
    else:
        print(f"No allocation can achieve expected PnL >= {target:,.0f}")