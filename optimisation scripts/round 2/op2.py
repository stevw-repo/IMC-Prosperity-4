import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product

# ----------------------------
# Parameters
# ----------------------------
N_PARTICIPANTS = 15000          # total number of participants (including us)
BUDGET_TOTAL = 50000            # total budget in XIRECs
TARGET_PNL = 1600000            # change this to your desired "k"
MONTE_CARLO_SCENARIOS = 5000    # number of simulated opponent fields

# ----------------------------
# Pillar functions (percentages are integers 0..100)
# ----------------------------
def research(pct: int) -> float:
    """Research outcome from percentage invested."""
    if pct == 0:
        return 0.0
    return 200_000 * np.log(1 + pct) / np.log(101)

def scale(pct: int) -> float:
    """Scale outcome from percentage invested."""
    return 7.0 * pct / 100.0

# ----------------------------
# Speed multiplier simulation
# ----------------------------
# We model other participants' speed allocations as uniformly distributed
# over the integer percentages 0..100. In a real competition the distribution
# might be strategic, but uniform is a reasonable neutral assumption.
opponent_speeds = np.random.randint(0, 101, size=(MONTE_CARLO_SCENARIOS, N_PARTICIPANTS - 1))

# Precompute for each possible speed percentage (0..100) the multiplier in each scenario.
# Rank 1 gets 0.9, rank N gets 0.1, linear interpolation between.
# To handle ties correctly: we assign the same rank to equal speeds.
# Efficient vectorised computation:
multiplier_matrix = np.zeros((101, MONTE_CARLO_SCENARIOS))
for sp in range(101):
    # Combine our speed with opponents
    all_speeds = np.concatenate([opponent_speeds, np.full((MONTE_CARLO_SCENARIOS, 1), sp)], axis=1)
    # Sort each scenario row in descending order (higher speed = better rank)
    # We need the rank of our value. Since we appended our value at the end,
    # we can use argsort and find its position.
    # Alternatively: rank = (N - 1) - np.sum(all_speeds > sp, axis=1) + 1
    # Because: if there are `c` values strictly greater than sp, our rank is c+1 (1-indexed).
    # But with ties: if we have the same value as others, they are not "greater".
    # This formula gives the highest rank among tied values? Let's test:
    # For values [95,20,10] with sp=20, greater=1, rank=2 -> correct (tie with others?).
    # Actually for ties, they share the same rank. The multiplier formula uses the rank
    # as the position in the sorted list, and ties get the same rank. Our method
    # gives the smallest rank (best) among tied participants, which is what the challenge
    # specifies: "equal investments share the same rank" and "first three players get 0.9".
    # So our method is correct.
    count_greater = np.sum(all_speeds > sp, axis=1)
    rank = count_greater + 1  # 1-indexed
    # Multiplier: linearly interpolate between 0.9 (rank=1) and 0.1 (rank=N)
    multiplier = 0.9 - 0.8 * (rank - 1) / (N_PARTICIPANTS - 1)
    multiplier_matrix[sp] = multiplier

# ----------------------------
# Search for best allocation
# ----------------------------
# We'll search over integer percentages with step = 1% for high resolution,
# but restrict total allocation ≤ 100%.
# To keep runtime manageable, we use a coarse grid first, then refine.
print(f"Searching for allocation that maximises P(PnL >= {TARGET_PNL})...")

best_prob = -1.0
best_alloc = None

# Precompute Research and Scale for all percentages
R_vals = np.array([research(p) for p in range(101)])
S_vals = np.array([scale(p) for p in range(101)])

# For speed we will evaluate many allocations; we can vectorise over scenarios.
# We'll iterate over all (r, s, sp) with r+s+sp <= 100.
# Since 101^3 ~ 1e6, we can do a full search with vectorised probability calculation.
all_r = np.arange(101)
all_s = np.arange(101)
all_sp = np.arange(101)

# Create a meshgrid for r, s, sp
R_grid, S_grid, SP_grid = np.meshgrid(all_r, all_s, all_sp, indexing='ij')
mask = (R_grid + S_grid + SP_grid) <= 100
r_valid = R_grid[mask]
s_valid = S_grid[mask]
sp_valid = SP_grid[mask]

n_alloc = len(r_valid)
print(f"Evaluating {n_alloc} valid allocations...")

# Pre-allocate arrays for speed multipliers (scenarios x allocations)
# We'll compute PnL for each scenario and then probability.
# Use broadcasting to avoid loops.

# Get multiplier for each allocation's sp
multipliers = multiplier_matrix[sp_valid, :]  # shape (n_alloc, n_scenarios)

# Compute PnL per scenario
budget_used = (r_valid + s_valid + sp_valid)[:, None] * (BUDGET_TOTAL / 100.0)  # shape (n_alloc, 1)
pnl = (R_vals[r_valid][:, None] * S_vals[s_valid][:, None] * multipliers) - budget_used

# Probability of PnL >= TARGET_PNL per allocation
probs = np.mean(pnl >= TARGET_PNL, axis=1)  # shape (n_alloc,)

# Find best
best_idx = np.argmax(probs)
best_prob = probs[best_idx]
best_r = r_valid[best_idx]
best_s = s_valid[best_idx]
best_sp = sp_valid[best_idx]

print(f"\nOptimal allocation for target PnL >= {TARGET_PNL}:")
print(f"  Research: {best_r}%")
print(f"  Scale:    {best_s}%")
print(f"  Speed:    {best_sp}%")
print(f"  Total used: {best_r+best_s+best_sp}%")
print(f"  Confidence level: {best_prob*100:.2f}%")

# Expected PnL under this allocation (optional)
expected_pnl = np.mean(pnl[best_idx])
print(f"  Expected PnL: {expected_pnl:.2f}")

# ----------------------------
# 3D Visualisation: Confidence surface for optimal Speed %
# ----------------------------
# We fix sp = best_sp and plot probability as a function of (r,s)
r_plot = np.arange(0, 101, 2)  # step 2% for cleaner plot
s_plot = np.arange(0, 101, 2)
R_plot, S_plot = np.meshgrid(r_plot, s_plot)
sp_fixed = best_sp

Z = np.zeros_like(R_plot, dtype=float)
for i in range(R_plot.shape[0]):
    for j in range(R_plot.shape[1]):
        r = R_plot[i, j]
        s = S_plot[i, j]
        if r + s + sp_fixed > 100:
            Z[i, j] = np.nan
            continue
        # Compute probability using precomputed multiplier matrix
        mult = multiplier_matrix[sp_fixed]
        budg = (r + s + sp_fixed) * (BUDGET_TOTAL / 100.0)
        pnl_fixed = R_vals[r] * S_vals[s] * mult - budg
        Z[i, j] = np.mean(pnl_fixed >= TARGET_PNL)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(R_plot, S_plot, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('Research %')
ax.set_ylabel('Scale %')
ax.set_zlabel(f'Confidence P(PnL ≥ {TARGET_PNL})')
ax.set_title(f'Confidence Surface for Fixed Speed = {sp_fixed}%')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.show()

# Optional: 3D scatter of top allocations (e.g., top 100 by confidence)
# top_indices = np.argsort(probs)[-100:]
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')
# ax2.scatter(r_valid[top_indices], s_valid[top_indices], sp_valid[top_indices],
#             c=probs[top_indices], cmap='plasma')
# ax2.set_xlabel('Research')
# ax2.set_ylabel('Scale')
# ax2.set_zlabel('Speed')
# plt.title('Top 100 Allocations by Confidence')
# plt.show()