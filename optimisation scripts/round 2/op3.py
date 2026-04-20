#!/usr/bin/env python3

# UNIFORM

"""
IMC Prosperity 4 – Manual Challenge "Invest & Expand" Optimizer
===============================================================
Finds the budget allocation (Research, Scale, Speed) that maximises
the probability of reaching a target PnL, treating the competitive
Speed multiplier as a random variable.

Speed model
-----------
For N ≈ 15 000 players the rank‑based multiplier is essentially
deterministic once you know the population CDF F:

    speed_mult(sp) ≈ 0.1 + 0.8 · F(sp / 100)

We don't know F, so we place a hierarchical Beta prior on each
opponent's speed allocation:

    μ  ~ Uniform(0.10, 0.40)   mean fraction allocated to speed
    κ  ~ Uniform(3, 15)        concentration
    α  = μ·κ ,  β = (1−μ)·κ

Each Monte‑Carlo draw gives one plausible F ⟹ one plausible
speed multiplier ⟹ one PnL sample.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import beta as beta_rv
import time, textwrap

# ── constants ────────────────────────────────────────────────
BUDGET      = 50_000
LOG101      = np.log(101)          # ≈ 4.6151
N_PLAYERS   = 20_000
N_MC        = 20_000               # Monte‑Carlo samples

# ── pillar value functions ───────────────────────────────────

def R(x):
    """Research value (vectorised). 0→0, 100→200 000, log growth."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 200_000 * np.log(1.0 + x) / LOG101, 0.0)

def S(x):
    """Scale value (vectorised). 0→0, 100→7, linear."""
    return 7.0 * np.asarray(x, dtype=float) / 100.0

# ── speed multiplier sampler ─────────────────────────────────

def speed_samples(sp, n=N_MC, rng=None):
    """
    Return *n* plausible speed‑multiplier values for allocation *sp* %.
    Uses the hierarchical Beta prior described in the docstring.
    """
    if rng is None:
        rng = np.random.default_rng()
    mu    = rng.uniform(0.10, 0.40, n)          # mean opp. speed frac
    kappa = rng.uniform(3.0,  15.0, n)          # concentration
    a, b  = mu * kappa, (1.0 - mu) * kappa
    cdf   = beta_rv.cdf(sp / 100.0, a, b)      # frac. of opponents below you
    return 0.1 + 0.8 * cdf                      # rank → multiplier

# ── exhaustive optimisation ──────────────────────────────────

def optimise(targets, n_mc=N_MC):
    """
    Enumerate every integer allocation (r, s, sp) with r+s+sp = 100.
    For each, compute E[PnL], median, σ, and P(PnL ≥ k) for every k
    in *targets*.  Returns a results dict.
    """
    rng = np.random.default_rng(42)
    t0 = time.time()

    # pre‑compute speed‑mult samples for sp = 0 … 100
    print("  ⏳ Sampling speed multipliers for sp = 0 … 100 …")
    sp_m = {sp: speed_samples(sp, n_mc, rng) for sp in range(101)}

    N = 101 * 102 // 2                   # total allocations
    allocs = np.empty((N, 3), dtype=int)
    E, Med, Std = [np.empty(N) for _ in range(3)]
    P10, P90    = np.empty(N), np.empty(N)
    Pr = {k: np.empty(N) for k in targets}

    print("  ⏳ Evaluating allocations …")
    i = 0
    for sp in range(101):
        m   = sp_m[sp]                    # (n_mc,) speed mults
        rem = 100 - sp
        for r in range(rem + 1):
            s   = rem - r
            pnl = float(R(r)) * float(S(s)) * m - BUDGET  # vector
            allocs[i] = (r, s, sp)
            E[i], Med[i], Std[i] = pnl.mean(), np.median(pnl), pnl.std()
            P10[i], P90[i] = np.percentile(pnl, 10), np.percentile(pnl, 90)
            for k in targets:
                Pr[k][i] = (pnl >= k).mean()
            i += 1

    sl = slice(0, i)
    res = dict(allocs=allocs[sl], E=E[sl], Med=Med[sl], Std=Std[sl],
               P10=P10[sl], P90=P90[sl],
               Pr={k: v[sl] for k, v in Pr.items()},
               sp_m=sp_m, targets=targets)

    # best expected
    j = res['E'].argmax()
    res['opt_E'] = dict(alloc=tuple(res['allocs'][j]), val=res['E'][j], idx=j)

    # best median
    j = res['Med'].argmax()
    res['opt_Med'] = dict(alloc=tuple(res['allocs'][j]), val=res['Med'][j], idx=j)

    # best P(PnL ≥ k) for each target
    res['opt_Pr'] = {}
    for k in targets:
        j = res['Pr'][k].argmax()
        res['opt_Pr'][k] = dict(alloc=tuple(res['allocs'][j]),
                                prob=res['Pr'][k][j],
                                E_pnl=res['E'][j])

    print(f"  ✅ {i:,} allocations in {time.time()-t0:.1f}s\n")
    return res

# ── pretty‑print ─────────────────────────────────────────────

def show(res):
    line = "=" * 72
    print(f"\n{line}")
    print("  OPTIMISATION RESULTS")
    print(f"  Speed prior: hierarchical Beta  |  N = {N_PLAYERS:,}  |  MC = {N_MC:,}")
    print(line)

    # — max expected —
    r, s, sp = res['opt_E']['alloc']
    j = res['opt_E']['idx']
    rv, sv = float(R(r)), float(S(s))
    m = res['sp_m'][sp]
    print(f"\n  ► Maximum Expected PnL")
    print(f"    Research = {r}%   Scale = {s}%   Speed = {sp}%")
    print(f"    E[PnL]  = {res['E'][j]:>12,.0f}")
    print(f"    Median  = {res['Med'][j]:>12,.0f}")
    print(f"    σ       = {res['Std'][j]:>12,.0f}")
    print(f"    [P10, P90] = [{res['P10'][j]:>10,.0f} , {res['P90'][j]:>10,.0f}]")
    print(f"    R({r})={rv:,.0f}   S({s})={sv:.4f}")
    print(f"    Speed mult: mean={m.mean():.4f}  σ={m.std():.4f}"
          f"  [{np.percentile(m,10):.3f} – {np.percentile(m,90):.3f}]")
    print(f"    PnL @  m=0.1 → {rv*sv*0.1-BUDGET:>10,.0f}"
          f"    m=0.5 → {rv*sv*0.5-BUDGET:>10,.0f}"
          f"    m=0.9 → {rv*sv*0.9-BUDGET:>10,.0f}")

    # — per‑target confidence —
    print(f"\n  ► Best allocation per target PnL  (highest confidence level)")
    print(f"    {'target k':>10}   R%  S%  Sp%    P(PnL≥k)      E[PnL]")
    print(f"    {'-'*58}")
    for k in res['targets']:
        d = res['opt_Pr'][k]
        r2, s2, sp2 = d['alloc']
        print(f"    {k:>10,}   {r2:>3} {s2:>3}  {sp2:>3}"
              f"    {d['prob']:>8.4f}    {d['E_pnl']:>11,.0f}")
    print()

# ── grid helper ──────────────────────────────────────────────

def _grid(res, field, target=None):
    """Map flat results to a (101, 101) grid  Z[sp, r]."""
    Z = np.full((101, 101), np.nan)
    arr = res[field] if target is None else res['Pr'][target]
    for i, (r, _, sp) in enumerate(res['allocs']):
        Z[sp, r] = arr[i]
    return Z

# ── plots ────────────────────────────────────────────────────

def plot_main(res):
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("IMC Prosperity 4 — Manual Challenge Optimiser", fontsize=14, y=0.98)

    # ---- 1  3‑D expected PnL surface ----
    ax = fig.add_subplot(231, projection='3d')
    Z  = _grid(res, 'E')
    Rg, SPg = np.meshgrid(range(101), range(101))
    Zm = np.ma.masked_invalid(Z)
    ax.plot_surface(Rg, SPg, Zm, cmap='viridis', alpha=.85, lw=0, antialiased=True)
    r, s, sp = res['opt_E']['alloc']
    ax.scatter([r], [sp], [res['opt_E']['val']], c='red', s=250, zorder=10,
              edgecolors='black', linewidths=1)
    ax.set_xlabel('Research %'); ax.set_ylabel('Speed %'); ax.set_zlabel('E[PnL]')
    ax.set_title(f'E[PnL]  (opt R={r} S={s} Sp={sp})', fontsize=10)

    # ---- 2  heatmap expected PnL ----
    ax2 = fig.add_subplot(232)
    Zp = Z.copy(); Zp[np.isnan(Zp)] = np.nanmin(Z)
    im = ax2.imshow(Zp, origin='lower', extent=[0,100,0,100],
                    aspect='auto', cmap='viridis')
    ax2.plot(r, sp, 'r*', ms=20, label=f'opt ({r},{100-r-sp},{sp})')
    ax2.set_xlabel('Research %'); ax2.set_ylabel('Speed %')
    ax2.set_title('E[PnL] heatmap'); ax2.legend(fontsize=8)
    fig.colorbar(im, ax=ax2, shrink=.82)

    # ---- 3  speed CDF ----
    ax3 = fig.add_subplot(233)
    for v in [0, 5, 10, 15, 20, 30, 40, 50, 60, 80]:
        ms = np.sort(res['sp_m'][v])
        ax3.plot(ms, np.linspace(0, 1, len(ms)), lw=1.5, label=f'sp={v}')
    ax3.set_xlabel('Speed multiplier'); ax3.set_ylabel('CDF')
    ax3.set_title('Speed mult. CDF by allocation')
    ax3.legend(fontsize=7, ncol=2); ax3.grid(alpha=.3)

    # ---- 4‑6  confidence heat maps (pick 3 interesting targets) ----
    sel = [k for k in res['targets']
           if 0.03 < res['opt_Pr'][k]['prob'] < 0.99]
    # pick low / mid / high if available
    if len(sel) >= 3:
        pick = [sel[0], sel[len(sel)//2], sel[-1]]
    else:
        pick = sel[:3]

    for pi, k in enumerate(pick):
        ax4 = fig.add_subplot(234 + pi)
        Zk = _grid(res, None, target=k)
        Zk[np.isnan(Zk)] = 0
        im = ax4.imshow(Zk, origin='lower', extent=[0,100,0,100],
                        aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        d = res['opt_Pr'][k]; ri, si, spi = d['alloc']
        ax4.plot(ri, spi, 'k*', ms=16)
        ax4.set_xlabel('Research %'); ax4.set_ylabel('Speed %')
        ax4.set_title(f"P(PnL ≥ {k:,})\nR={ri} S={si} Sp={spi}  P={d['prob']:.3f}",
                      fontsize=9)
        fig.colorbar(im, ax=ax4, shrink=.82)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('imc_main.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → saved imc_main.png")

def plot_pnl_dist(res):
    """PnL histogram + CDF for the expected‑optimal allocation."""
    r, s, sp = res['opt_E']['alloc']
    pnl = float(R(r)) * float(S(s)) * res['sp_m'][sp] - BUDGET

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'PnL distribution  R={r}%  S={s}%  Sp={sp}%', fontsize=12)

    ax1.hist(pnl, bins=80, density=True, color='steelblue', edgecolor='white', lw=.4)
    for val, c, ls, lbl in [
        (pnl.mean(),            'red',    '--', f'Mean {pnl.mean():,.0f}'),
        (np.median(pnl),        'orange', '--', f'Median {np.median(pnl):,.0f}'),
        (np.percentile(pnl,10), 'green',  ':',  f'P10 {np.percentile(pnl,10):,.0f}'),
        (np.percentile(pnl,90), 'green',  ':',  f'P90 {np.percentile(pnl,90):,.0f}'),
    ]:
        ax1.axvline(val, color=c, ls=ls, lw=2, label=lbl)
    ax1.legend(fontsize=8); ax1.set_xlabel('PnL'); ax1.grid(alpha=.3)

    sp_pnl = np.sort(pnl)
    ax2.plot(sp_pnl, np.linspace(0, 1, len(sp_pnl)), 'steelblue', lw=2)
    for k in res['targets']:
        p = (pnl >= k).mean()
        if 0.02 < p < 0.98:
            ax2.axhline(1-p, color='grey', ls=':', alpha=.4)
            ax2.text(sp_pnl[0], 1-p+.015, f'P(≥{k/1e3:.0f}k)={p:.2f}', fontsize=7)
    ax2.set_xlabel('PnL'); ax2.set_ylabel('CDF'); ax2.grid(alpha=.3)
    ax2.set_title('CDF with target thresholds')

    plt.tight_layout()
    plt.savefig('imc_pnl_dist.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → saved imc_pnl_dist.png")

def plot_deterministic():
    """PnL vs Research % for sp = 0, at fixed multiplier values."""
    fig, ax = plt.subplots(figsize=(12, 6))
    r_arr = np.arange(101)
    for m in np.arange(0.1, 1.0, 0.1):
        pnl = R(r_arr) * S(100 - r_arr) * m - BUDGET
        ax.plot(r_arr, pnl, lw=2, label=f'm = {m:.1f}')
        j = pnl.argmax()
        ax.plot(j, pnl[j], 'o', ms=6, color='black')
    ax.axhline(0, color='k', lw=.5)
    ax.set_xlabel('Research %  (Scale = 100 − R,  Speed = 0)')
    ax.set_ylabel('PnL')
    ax.set_title('Deterministic PnL  (fixed speed multiplier, no speed investment)')
    ax.legend(ncol=3, fontsize=9); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig('imc_deterministic.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → saved imc_deterministic.png")

# ── sensitivity analysis ─────────────────────────────────────

def sensitivity(targets_short):
    """Re‑run under different priors on opponent speed."""
    configs = {
        'Cautious  μ∈[5,25]%':  (0.05, 0.25),
        'Baseline  μ∈[10,40]%': (0.10, 0.40),
        'Aggress.  μ∈[25,55]%': (0.25, 0.55),
        'Wide      μ∈[5,55]%':  (0.05, 0.55),
    }
    print("  ► Sensitivity to prior on opponent speed mean\n")
    print(f"    {'Prior':>24}   {'opt R':>4} {'S':>3} {'Sp':>3}"
          f"   {'E[PnL]':>10}  |  target 200k best P")
    print(f"    {'-'*70}")

    for label, (lo, hi) in configs.items():
        rng = np.random.default_rng(99)
        sp_m = {}
        for sp in range(101):
            mu = rng.uniform(lo, hi, N_MC)
            kp = rng.uniform(3, 15, N_MC)
            a, b = mu*kp, (1-mu)*kp
            sp_m[sp] = 0.1 + 0.8 * beta_rv.cdf(sp/100., a, b)

        best_E, best_a = -np.inf, (0,0,0)
        best_p200, best_a200 = -1, (0,0,0)
        for sp in range(101):
            m = sp_m[sp]; rem = 100-sp
            for r in range(rem+1):
                s = rem - r
                pnl = float(R(r))*float(S(s))*m - BUDGET
                ev = pnl.mean()
                if ev > best_E: best_E, best_a = ev, (r,s,sp)
                p2 = (pnl >= 200_000).mean()
                if p2 > best_p200: best_p200, best_a200 = p2, (r,s,sp)

        r,s,sp = best_a
        r2,s2,sp2 = best_a200
        print(f"    {label:>24}   {r:>4} {s:>3} {sp:>3}"
              f"   {best_E:>10,.0f}  |  R={r2} S={s2} Sp={sp2}  P={best_p200:.3f}")
    print()

# ── main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    np.set_printoptions(linewidth=120)

    print("=" * 72)
    print("  IMC Prosperity 4 – Manual Challenge Optimiser")
    print("  «Invest & Expand»")
    print("=" * 72)
    print(f"  Budget       = {BUDGET:>10,} XIRECs")
    print(f"  Players (≈)  = {N_PLAYERS:>10,}")
    print(f"  MC samples   = {N_MC:>10,}")
    print()

    targets = [150_000, 160_000, 170_000, 180_000, 190_000, 200_000]

    # ── optimise ──
    res = optimise(targets)
    show(res)

    # ── deterministic reference ──
    print("  ► Deterministic optimum  (sp = 0, fixed m)")
    for m in np.arange(0.1, 1.0, 0.1):
        r_a = np.arange(101)
        pv  = R(r_a) * S(100 - r_a) * m - BUDGET
        j   = pv.argmax()
        print(f"    m={m:.1f}:  R={j:>3}%  S={100-j:>3}%  Sp=0%   PnL = {pv[j]:>10,.0f}")
    print()

    # ── sensitivity ──
    print("=" * 72)
    print("  SENSITIVITY ANALYSIS")
    print("=" * 72)
    sensitivity([200_000])

    # ── plots ──
    print("  Generating plots …")
    plot_main(res)
    plot_pnl_dist(res)
    plot_deterministic()

    print("\n  All done ✅")