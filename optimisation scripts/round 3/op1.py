import numpy as np
from scipy.optimize import minimize, fsolve

# ------------------------------------------------------------
# Parameters of the manual challenge
# ------------------------------------------------------------
L = 670.0          # lowest reserve price
H = 920.0          # highest reserve price (also fair selling price)
step = 5.0         # reserve prices increment
r_values = np.arange(L, H + step, step)   # [670, 675, ..., 920]
n_r = len(r_values)                       # 51 possible reserves
prob_r = 1.0 / n_r                        # uniform probability

def expected_profit(b1, b2, A):
    """
    Expected profit per counterparty for a trader who submits bids (b1, b2),
    when the average of all players' second bids is A.
    The reserve price r is uniformly distributed over the discrete set.
    """
    profit = 0.0
    for r in r_values:
        if b1 > r:
            # first bid wins
            profit += (H - b1)
        elif b2 > r:
            # second bid considered
            if b2 > A:
                # no penalty
                profit += (H - b2)
            else:
                # penalty applies
                penalty = ((H - A) / (H - b2)) ** 3
                profit += (H - b2) * penalty
        # else: no trade
    return profit * prob_r

def best_response(A, b1_guess=750.0, b2_guess=830.0):
    """
    For a given average second bid A, find the trader's optimal (b1, b2)
    that maximises expected profit.
    Returns (b1_opt, b2_opt, max_profit).
    """
    # Use scipy.optimize.minimize to maximise negative profit
    def objective(x):
        b1, b2 = x
        # ensure b1 <= b2, bids within [L, H]
        if b1 < L or b2 < L or b1 > H or b2 > H or b1 > b2:
            return 1e9  # large penalty for infeasible
        return -expected_profit(b1, b2, A)
    
    # Bounds and initial guess
    bounds = [(L, H), (L, H)]
    # Use L-BFGS-B which handles bounds
    res = minimize(objective, [b1_guess, b2_guess], bounds=bounds,
                   method='L-BFGS-B')
    if res.success:
        b1_opt, b2_opt = res.x
        return b1_opt, b2_opt, -res.fun
    else:
        # fallback: grid search (coarse then fine)
        best_profit = -np.inf
        best_b1, best_b2 = b1_guess, b2_guess
        for b1 in np.linspace(L, H, 101):
            for b2 in np.linspace(b1, H, 101):
                prof = expected_profit(b1, b2, A)
                if prof > best_profit:
                    best_profit = prof
                    best_b1, best_b2 = b1, b2
        return best_b1, best_b2, best_profit

def fixed_point_equation(A):
    """
    For a given candidate average A, compute the best response second bid.
    Nash equilibrium requires that the best response second bid equals A.
    """
    _, b2_opt, _ = best_response(A)
    return b2_opt - A

# ------------------------------------------------------------
# Solve for Nash equilibrium using fixed point iteration
# ------------------------------------------------------------
# Initial guess: analytic solution from continuous uniform approximation
A_initial = (2*H + L) / 3.0   # 836.666...
print(f"Analytic equilibrium (continuous): b1 = {(H+2*L)/3:.3f}, b2 = {A_initial:.3f}")

# Use fsolve to find A such that b2_opt(A) = A
A_eq = fsolve(fixed_point_equation, A_initial)[0]
b1_eq, b2_eq, _ = best_response(A_eq)

print("\nNash equilibrium (discrete reserves, numerical):")
print(f"First bid  = {b1_eq:.4f}")
print(f"Second bid = {b2_eq:.4f}")
print(f"Expected profit per counterparty = {expected_profit(b1_eq, b2_eq, b2_eq):.4f}")

# ------------------------------------------------------------
# Optional: verify that deviation does not improve profit
# ------------------------------------------------------------
def check_deviation(A_eq, b1_eq, b2_eq):
    sym_profit = expected_profit(b1_eq, b2_eq, b2_eq)
    # Try a slightly different second bid
    for delta in [-10, -5, -1, 1, 5, 10]:
        b2_test = b2_eq + delta
        if b2_test < L or b2_test > H:
            continue
        # re-optimise b1 for this b2 (fix b2, choose best b1)
        def profit_b1(b1):
            return expected_profit(b1, b2_test, b2_eq)
        b1_candidates = np.linspace(L, b2_test, 101)
        profits = [profit_b1(b1) for b1 in b1_candidates]
        best_idx = np.argmax(profits)
        b1_best = b1_candidates[best_idx]
        dev_profit = profit_b1(b1_best)
        print(f"b2={b2_test:.2f}, best b1={b1_best:.2f}, profit={dev_profit:.4f} (vs sym {sym_profit:.4f})")
    return

print("\nDeviation check (should show lower profit):")
check_deviation(A_eq, b1_eq, b2_eq)