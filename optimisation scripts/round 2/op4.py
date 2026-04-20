#!/usr/bin/env python3

# TRUNICATED NORMAL

"""
IMC Prosperity 4 – Manual Challenge "Invest & Expand" Optimizer
===============================================================
Speed model (v2 – Truncated Normal)
------------------------------------
Opponents' speed allocation ~ TruncNormal(μ=50, σ², 0, 100).
σ is unknown → prior  σ ~ Uniform(σ_lo, σ_hi).

Each MC draw:
  1. sample σ
  2. compute CDF of TruncNormal(50, σ²) at your allocation sp
  3. speed_mult = 0.1 + 0.8 · CDF(sp)

With N ≈ 15 000 the empirical rank converges to the population CDF.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import truncnorm, norm
import time

# ── constants ────────────────────────────────────────────────
BUDGET      = 50_000
LOG101      = np.log(101)
N_PLAYERS   = 15_000
N_MC        = 20_000

# Prior on σ of opponents' speed allocation (%)
SIGMA_LO    = 10.0      # tight cluster around 50
SIGMA_HI    = 40.0     # broad spread

# ── pillar value functions ───────────────────────────────────

def R(x):
    """Research value. 0→0, 100→200 000, log growth."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 200_000 * np.log(1.0 + x) / LOG101, 0.0)

def S(x):
    """Scale value. 0→0, 100→7, linear."""
    return 7.0 * np.asarray(x, dtype=float) / 100.0

# ── speed multiplier via truncated normal CDF ────────────────

def trunc_normal_cdf(sp, mu, sigma):
    """
    CDF of TruncNormal(mu, sigma², 0, 100) evaluated at sp.
    Vectorised over sigma.
    """
    a = (0.0   - mu) / sigma          # lower bound in standard units
    b = (100.0 - mu) / sigma          # upper bound in standard units
    z = (sp    - mu) / sigma
    # CDF = (Φ(z) − Φ(a)) / (Φ(b) − Φ(a))
    Phi_a = norm.cdf(a)
    Phi_b = norm.cdf(b)
    Phi_z = norm.cdf(z)
    return (Phi_z - Phi_a) / (Phi_b - Phi_a)

def speed_samples(sp, n=N_MC, rng=None, sigma_lo=SIGMA_LO, sigma_hi=SIGMA_HI):
    """
    Return n plausible speed‑multiplier values for allocation sp %.
    Each draw samples σ ~ U(σ_lo, σ_hi), then computes the TruncNormal CDF.
    """
    if rng is None:
        rng = np.random.default_rng()
    sigmas = rng.uniform(sigma_lo, sigma_hi, n)
    cdf = trunc_normal_cdf(sp, 35.0, sigmas)
    return 0.1 + 0.8 * cdf

# ── exhaustive optimisation ──────────────────────────────────

def optimise(targets, n_mc=N_MC):
    rng = np.random.default_rng(42)
    t0 = time.time()

    print("  ⏳ Sampling speed multipliers for sp = 0 … 100 …")
    sp_m = {sp: speed_samples(sp, n_mc, rng) for sp in range(101)}

    N = 101 * 102 // 2
    allocs = np.empty((N, 3), dtype=int)
    E, Med, Std = [np.empty(N) for _ in range(3)]
    P10, P90    = np.empty(N), np.empty(N)
    Pr = {k: np.empty(N) for k in targets}

    print("  ⏳ Evaluating allocations …")
    i = 0
    for sp in range(101):
        m   = sp_m[sp]
        rem = 100 - sp
        for r in range(rem + 1):
            s   = rem - r
            pnl = float(R(r)) * float(S(s)) * m - BUDGET
            allocs[i] = (r, s, sp)
            E[i]   = pnl.mean()
            Med[i] = np.median(pnl)
            Std[i] = pnl.std()
            P10[i] = np.percentile(pnl, 10)
            P90[i] = np.percentile(pnl, 90)
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

    # best P(PnL ≥ k)
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
    print(f"  Speed prior: TruncNormal(μ=50%, σ~U[{SIGMA_LO},{SIGMA_HI}])  "
          f"|  N ≈ {N_PLAYERS:,}  |  MC = {N_MC:,}")
    print(line)

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
    Z = np.full((101, 101), np.nan)
    arr = res[field] if target is None else res['Pr'][target]
    for i, (r, _, sp) in enumerate(res['allocs']):
        Z[sp, r] = arr[i]
    return Z

# ── plots ────────────────────────────────────────────────────

def plot_main(res):
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("IMC Prosperity 4 — Manual Challenge Optimiser  "
                 f"[TruncNorm(50, σ~U[{SIGMA_LO},{SIGMA_HI}])]",
                 fontsize=13, y=0.98)

    # 1  3‑D expected PnL surface
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

    # 2  heatmap expected PnL
    ax2 = fig.add_subplot(232)
    Zp = Z.copy(); Zp[np.isnan(Zp)] = np.nanmin(Z)
    im = ax2.imshow(Zp, origin='lower', extent=[0,100,0,100],
                    aspect='auto', cmap='viridis')
    ax2.plot(r, sp, 'r*', ms=20, label=f'opt ({r},{100-r-sp},{sp})')
    ax2.set_xlabel('Research %'); ax2.set_ylabel('Speed %')
    ax2.set_title('E[PnL] heatmap'); ax2.legend(fontsize=8)
    fig.colorbar(im, ax=ax2, shrink=.82)

    # 3  speed multiplier distribution for selected sp values
    ax3 = fig.add_subplot(233)
    for v in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        ms = np.sort(res['sp_m'][v])
        ax3.plot(ms, np.linspace(0, 1, len(ms)), lw=1.5, label=f'sp={v}')
    ax3.set_xlabel('Speed multiplier'); ax3.set_ylabel('CDF')
    ax3.set_title('Speed mult. CDF by allocation')
    ax3.legend(fontsize=7, ncol=2); ax3.grid(alpha=.3)

    # 4–6  confidence heatmaps
    sel = [k for k in res['targets']
           if 0.03 < res['opt_Pr'][k]['prob'] < 0.99]
    if len(sel) >= 3:
        pick = [sel[0], sel[len(sel)//2], sel[-1]]
    else:
        pick = sel[:3] if sel else res['targets'][:3]

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

# ── speed multiplier vs allocation curve ─────────────────────

def plot_speed_curve(res):
    """Expected speed multiplier ± bands vs speed allocation."""
    sps = np.arange(101)
    means = np.array([res['sp_m'][sp].mean() for sp in sps])
    p10   = np.array([np.percentile(res['sp_m'][sp], 10) for sp in sps])
    p90   = np.array([np.percentile(res['sp_m'][sp], 90) for sp in sps])
    p25   = np.array([np.percentile(res['sp_m'][sp], 25) for sp in sps])
    p75   = np.array([np.percentile(res['sp_m'][sp], 75) for sp in sps])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(sps, p10, p90, alpha=.15, color='steelblue', label='10–90 %ile')
    ax.fill_between(sps, p25, p75, alpha=.3,  color='steelblue', label='25–75 %ile')
    ax.plot(sps, means, 'steelblue', lw=2.5, label='Mean')
    ax.axhline(0.5, color='grey', ls=':', lw=1)
    ax.set_xlabel('Your Speed allocation %')
    ax.set_ylabel('Speed multiplier')
    ax.set_title(f'Speed multiplier vs allocation  '
                 f'[TruncNorm(50, σ~U[{SIGMA_LO},{SIGMA_HI}])]')
    ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig('imc_speed_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → saved imc_speed_curve.png")

# ── sensitivity analysis ─────────────────────────────────────

def sensitivity():
    configs = {
        'Tight    σ∈[ 5,15]':  ( 5.0, 15.0),
        'Baseline σ∈[ 5,30]':  ( 5.0, 30.0),
        'Wide     σ∈[10,35]':  (10.0, 35.0),
        'VeryWide σ∈[ 3,40]':  ( 3.0, 40.0),
    }
    print("  ► Sensitivity to σ prior on opponent speed\n")
    print(f"    {'Prior':>24}   {'opt R':>4} {'S':>3} {'Sp':>3}"
          f"   {'E[PnL]':>10}  |  target 200k: best alloc & P")
    print(f"    {'-'*72}")

    for label, (slo, shi) in configs.items():
        rng = np.random.default_rng(99)
        sp_m = {}
        for sp in range(101):
            sigmas = rng.uniform(slo, shi, N_MC)
            sp_m[sp] = 0.1 + 0.8 * trunc_normal_cdf(sp, 50.0, sigmas)

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
    print("  «Invest & Expand»  –  TruncNormal speed model")
    print("=" * 72)
    print(f"  Budget       = {BUDGET:>10,} XIRECs")
    print(f"  Players (≈)  = {N_PLAYERS:>10,}")
    print(f"  MC samples   = {N_MC:>10,}")
    print(f"  Opp. speed ~ TruncNormal(μ=50%, σ ~ U[{SIGMA_LO}, {SIGMA_HI}])")
    print()

    targets = [50_000, 100_000, 150_000, 200_000, 250_000,
               300_000, 400_000, 500_000, 600_000]

    res = optimise(targets)
    show(res)

    # deterministic reference
    print("  ► Deterministic optimum  (sp = 0, fixed m)")
    for m in np.arange(0.1, 1.0, 0.1):
        r_a = np.arange(101)
        pv  = R(r_a) * S(100 - r_a) * m - BUDGET
        j   = pv.argmax()
        print(f"    m={m:.1f}:  R={j:>3}%  S={100-j:>3}%  Sp=0%   PnL = {pv[j]:>10,.0f}")
    print()

    # sensitivity
    print("=" * 72)
    print("  SENSITIVITY ANALYSIS")
    print("=" * 72)
    sensitivity()

    # plots
    print("  Generating plots …")
    plot_main(res)
    plot_pnl_dist(res)
    plot_speed_curve(res)
    plot_deterministic()

    print("\n  All done ✅")