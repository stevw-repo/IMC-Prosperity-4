#!/usr/bin/env python3

# BIMODAL 

"""
IMC Prosperity 4 – Manual Challenge "Invest & Expand" Optimizer
===============================================================
Bimodal opponent speed distribution model.

Speed model
-----------
Opponents' speed allocations follow a 4-component mixture:
  - ~15% cluster near 0%   (invest nothing in speed)
  - ~55% spread in the middle
  - ~15% cluster near 100% (go all-in on speed)
  - ~15% uniform catch-all

The CDF F of this mixture determines your speed multiplier:
    speed_mult(sp) = 0.1 + 0.8 · F(sp/100)

Uncertainty is modeled by jittering the mixture parameters on each
Monte Carlo draw.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import beta as beta_rv
import time

# ── constants ────────────────────────────────────────────────
BUDGET      = 50_000
LOG101      = np.log(101)
N_PLAYERS   = 20_000
N_MC        = 20_000

# ══════════════════════════════════════════════════════════════
#  BIMODAL SPEED DISTRIBUTION — EDIT THESE TO CHANGE THE MODEL
# ══════════════════════════════════════════════════════════════
#
#  The opponent speed allocation (as fraction 0–1) is drawn from
#  a 4-component mixture:
#
#  ┌──────────────┬────────┬────────┬─────────────────────────────────┐
#  │  Component    │ Weight │ Params │ Interpretation                  │
#  ├──────────────┼────────┼────────┼─────────────────────────────────┤
#  │ Low cluster  │ w_lo   │ a_lo,  │ Players investing ~0% in speed. │
#  │              │        │ b_lo   │ ↑b_lo or ↓a_lo = tighter spike  │
#  │              │        │        │ at 0.  mean ≈ a/(a+b).          │
#  ├──────────────┼────────┼────────┼─────────────────────────────────┤
#  │ Middle bulk  │ w_mid  │ a_mid, │ Moderate speed investors.       │
#  │              │        │ b_mid  │ mean ≈ a/(a+b).  ↑both = less   │
#  │              │        │        │ spread.                         │
#  ├──────────────┼────────┼────────┼─────────────────────────────────┤
#  │ High cluster │ w_hi   │ a_hi,  │ Players investing ~100% speed.  │
#  │              │        │ b_hi   │ ↑a_hi or ↓b_hi = tighter spike  │
#  │              │        │        │ at 100%.                        │
#  ├──────────────┼────────┼────────┼─────────────────────────────────┤
#  │ Uniform      │ 1-rest │  —     │ Catch-all / ignorance.          │
#  └──────────────┴────────┴────────┴─────────────────────────────────┘
#
#  jitter_frac : fractional noise (±%) applied to every parameter
#                on each MC draw, modelling our uncertainty about
#                the true population distribution.

SPEED_DIST = dict(
    # ── low cluster (near 0%) ──────────────────────────────
    w_lo  = 0.25,       # fraction of players (~15%)
    a_lo  = 2.0,        # Beta α  (keep small → spike near 0)
    b_lo  = 8.0,       # Beta β  (keep large → spike near 0)
                        # component mean ≈ 1.2/(1.2+12) ≈ 9%

    # ── middle bulk ────────────────────────────────────────
    w_mid = 0.70,       # fraction of players (~55%)
    a_mid = 2.5,        # Beta α
    b_mid = 3.0,        # Beta β
                        # component mean ≈ 2.0/(2.0+2.5) ≈ 44%

    # ── high cluster (near 100%) ───────────────────────────
    w_hi  = 0.05,       # fraction of players (~15%)
    a_hi  = 12.2,       # Beta α  (keep large → spike near 100%)
    b_hi  = 1.0,        # Beta β  (keep small → spike near 100%)
                        # component mean ≈ 12/(12+1.2) ≈ 91%

    # ── uniform remainder ──────────────────────────────────
    # weight = 1 - w_lo - w_mid - w_hi = 0.15  (automatic)

    # ── uncertainty ────────────────────────────────────────
    jitter_frac = 0.2, # ±15% random jitter on all params per MC draw
)

# ── pillar value functions ───────────────────────────────────

def R(x):
    """Research value. 0→0, 100→200 000, log growth."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 200_000 * np.log(1.0 + x) / LOG101, 0.0)

def S(x):
    """Scale value. 0→0, 100→7, linear."""
    return 7.0 * np.asarray(x, dtype=float) / 100.0

# ── speed multiplier sampler ─────────────────────────────────

def speed_samples(sp, n=N_MC, rng=None, cfg=None):
    """
    Return *n* plausible speed-multiplier values for allocation *sp* %.
    Uses the bimodal Beta mixture defined in cfg (default: SPEED_DIST).
    """
    if cfg is None:
        cfg = SPEED_DIST
    if rng is None:
        rng = np.random.default_rng()

    x  = sp / 100.0
    jf = cfg['jitter_frac']

    def jit(val):
        """Jitter a scalar parameter by ±jf, return (n,) array."""
        return np.maximum(val * rng.uniform(1 - jf, 1 + jf, n), 1e-6)

    # Jitter shape parameters
    a_lo  = jit(cfg['a_lo']);   b_lo  = jit(cfg['b_lo'])
    a_mid = jit(cfg['a_mid']); b_mid = jit(cfg['b_mid'])
    a_hi  = jit(cfg['a_hi']);  b_hi  = jit(cfg['b_hi'])

    # Jitter weights and renormalise so they sum to 1
    w_uni_base = max(1.0 - cfg['w_lo'] - cfg['w_mid'] - cfg['w_hi'], 0.0)
    w_lo_r  = jit(cfg['w_lo'])
    w_mid_r = jit(cfg['w_mid'])
    w_hi_r  = jit(cfg['w_hi'])
    w_uni_r = jit(w_uni_base) if w_uni_base > 0 else np.zeros(n)
    total   = w_lo_r + w_mid_r + w_hi_r + w_uni_r
    w_lo  = w_lo_r  / total
    w_mid = w_mid_r / total
    w_hi  = w_hi_r  / total
    w_uni = w_uni_r / total

    # Mixture CDF at x
    cdf = (w_lo  * beta_rv.cdf(x, a_lo, b_lo) +
           w_mid * beta_rv.cdf(x, a_mid, b_mid) +
           w_hi  * beta_rv.cdf(x, a_hi, b_hi) +
           w_uni * x)                              # Uniform CDF = x

    return 0.1 + 0.8 * cdf

# ── visualise the assumed population distribution ────────────

def plot_population_dist(cfg=None):
    """Show PDF and CDF of the assumed opponent speed distribution."""
    if cfg is None:
        cfg = SPEED_DIST

    rng = np.random.default_rng(0)
    N_SAMPLE = 200_000

    w_uni_base = max(1.0 - cfg['w_lo'] - cfg['w_mid'] - cfg['w_hi'], 0.0)
    weights = np.array([cfg['w_lo'], cfg['w_mid'], cfg['w_hi'], w_uni_base])
    weights /= weights.sum()

    comp = rng.choice(4, size=N_SAMPLE, p=weights)
    samples = np.empty(N_SAMPLE)
    m0, m1, m2, m3 = (comp == 0), (comp == 1), (comp == 2), (comp == 3)
    samples[m0] = rng.beta(cfg['a_lo'],  cfg['b_lo'],  m0.sum())
    samples[m1] = rng.beta(cfg['a_mid'], cfg['b_mid'], m1.sum())
    samples[m2] = rng.beta(cfg['a_hi'],  cfg['b_hi'],  m2.sum())
    samples[m3] = rng.uniform(0, 1, m3.sum())

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle('Assumed Opponent Speed Allocation Distribution (Bimodal)',
                 fontsize=13)

    # ── PDF ──
    ax1 = axes[0]
    ax1.hist(samples * 100, bins=100, density=True, color='steelblue',
             edgecolor='white', lw=0.3, alpha=0.85)
    ax1.axvline(np.median(samples)*100, color='red', ls='--', lw=2,
                label=f'Median = {np.median(samples)*100:.1f}%')
    ax1.axvline(np.mean(samples)*100, color='orange', ls='--', lw=2,
                label=f'Mean = {np.mean(samples)*100:.1f}%')
    frac_lo = (samples < 0.10).mean()
    frac_hi = (samples > 0.90).mean()
    ax1.text(0.02, 0.95, f'{frac_lo:.1%} below 10%\n{frac_hi:.1%} above 90%',
             transform=ax1.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))
    ax1.set_xlabel('Speed allocation %')
    ax1.set_ylabel('Density')
    ax1.set_title('PDF — opponent speed investments')
    ax1.legend(fontsize=9); ax1.grid(alpha=.3)

    # ── CDF ──
    ax2 = axes[1]
    ss = np.sort(samples)
    ax2.plot(ss * 100, np.linspace(0, 1, len(ss)), 'steelblue', lw=2)
    ax2.set_xlabel('Speed allocation %')
    ax2.set_ylabel('CDF (fraction of opponents below)')
    ax2.set_title('CDF — your percentile rank at each speed%')
    for sp_pct in [0, 5, 10, 20, 30, 50, 70, 90, 100]:
        cdf_val = (samples <= sp_pct / 100).mean()
        mult = 0.1 + 0.8 * cdf_val
        ax2.plot(sp_pct, cdf_val, 'ro', ms=5)
        ax2.annotate(f'{sp_pct}%→m={mult:.2f}',
                     (sp_pct, cdf_val), textcoords='offset points',
                     xytext=(8, -8 if sp_pct > 50 else 5), fontsize=7)
    ax2.grid(alpha=.3)

    # ── Speed multiplier vs allocation ──
    ax3 = axes[2]
    sp_arr = np.arange(101)
    cdf_arr = np.array([(samples <= sp/100).mean() for sp in sp_arr])
    mult_arr = 0.1 + 0.8 * cdf_arr
    ax3.plot(sp_arr, mult_arr, 'steelblue', lw=2.5)
    ax3.fill_between(sp_arr, 0.1, mult_arr, alpha=0.15, color='steelblue')
    ax3.set_xlabel('Your speed allocation %')
    ax3.set_ylabel('Expected speed multiplier')
    ax3.set_title('Speed multiplier vs your allocation')
    ax3.set_ylim(0.05, 0.95)
    ax3.grid(alpha=.3)
    # Highlight diminishing returns near extremes
    for sp_pct in [0, 20, 50, 80, 100]:
        ax3.plot(sp_pct, mult_arr[sp_pct], 'ro', ms=7)
        ax3.annotate(f'm={mult_arr[sp_pct]:.3f}', (sp_pct, mult_arr[sp_pct]),
                     textcoords='offset points', xytext=(5, 8), fontsize=8)

    plt.tight_layout()
    plt.savefig('imc_population_dist.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → saved imc_population_dist.png")

# ── exhaustive optimisation ──────────────────────────────────

def optimise(targets, n_mc=N_MC):
    """
    Enumerate every integer allocation (r, s, sp) with r+s+sp ≤ 100.
    We enforce r+s+sp = 100 since using the full budget is always optimal
    (Scale's marginal > 1 whenever Research > 0).
    """
    rng = np.random.default_rng(42)
    t0 = time.time()

    print("  ⏳ Sampling speed multipliers for sp = 0 … 100 …")
    sp_m = {sp: speed_samples(sp, n_mc, rng) for sp in range(101)}

    N = 101 * 102 // 2
    allocs = np.empty((N, 3), dtype=int)
    E, Med, Std = [np.empty(N) for _ in range(3)]
    P10, P90    = np.empty(N), np.empty(N)
    Pr = {k: np.empty(N) for k in targets}

    print("  ⏳ Evaluating all 5 151 allocations …")
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

    # Best by expected PnL
    j = res['E'].argmax()
    res['opt_E'] = dict(alloc=tuple(res['allocs'][j]), val=res['E'][j], idx=j)

    # Best by median PnL
    j = res['Med'].argmax()
    res['opt_Med'] = dict(alloc=tuple(res['allocs'][j]), val=res['Med'][j], idx=j)

    # Best P(PnL ≥ k) for each target
    res['opt_Pr'] = {}
    for k in targets:
        j = res['Pr'][k].argmax()
        res['opt_Pr'][k] = dict(alloc=tuple(res['allocs'][j]),
                                prob=res['Pr'][k][j],
                                E_pnl=res['E'][j])

    print(f"  ✅ {i:,} allocations in {time.time()-t0:.1f}s\n")
    return res

# ── pretty-print ─────────────────────────────────────────────

def show(res):
    line = "=" * 72
    print(f"\n{line}")
    print("  OPTIMISATION RESULTS  (Bimodal speed prior)")
    print(f"  N ≈ {N_PLAYERS:,}  |  MC = {N_MC:,}")
    sd = SPEED_DIST
    print(f"  Speed dist: lo={sd['w_lo']:.0%} Beta({sd['a_lo']},{sd['b_lo']})"
          f"  mid={sd['w_mid']:.0%} Beta({sd['a_mid']},{sd['b_mid']})"
          f"  hi={sd['w_hi']:.0%} Beta({sd['a_hi']},{sd['b_hi']})"
          f"  uni={1-sd['w_lo']-sd['w_mid']-sd['w_hi']:.0%}"
          f"  jitter=±{sd['jitter_frac']:.0%}")
    print(line)

    # ── Max expected ──
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

    # ── Max median ──
    r2, s2, sp2 = res['opt_Med']['alloc']
    print(f"\n  ► Maximum Median PnL")
    print(f"    Research = {r2}%   Scale = {s2}%   Speed = {sp2}%")
    print(f"    Median  = {res['opt_Med']['val']:>12,.0f}")

    # ── Per-target confidence ──
    print(f"\n  ► Best allocation per target PnL  (highest confidence level)")
    print(f"    {'target k':>10}   R%  S%  Sp%    P(PnL≥k)      E[PnL]")
    print(f"    {'-'*58}")
    for k in res['targets']:
        d = res['opt_Pr'][k]
        rk, sk, spk = d['alloc']
        print(f"    {k:>10,}   {rk:>3} {sk:>3}  {spk:>3}"
              f"    {d['prob']:>8.4f}    {d['E_pnl']:>11,.0f}")
    print()

# ── grid helper ──────────────────────────────────────────────

def _grid(res, field, target=None):
    """Map flat results → (101, 101) grid  Z[sp, r]."""
    Z = np.full((101, 101), np.nan)
    arr = res[field] if target is None else res['Pr'][target]
    for i, (r, _, sp) in enumerate(res['allocs']):
        Z[sp, r] = arr[i]
    return Z

# ── plots ────────────────────────────────────────────────────

def plot_main(res):
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("IMC Prosperity 4 — Optimiser (Bimodal Speed Model)",
                 fontsize=14, y=0.98)

    # 1: 3D expected PnL surface
    ax = fig.add_subplot(231, projection='3d')
    Z  = _grid(res, 'E')
    Rg, SPg = np.meshgrid(range(101), range(101))
    Zm = np.ma.masked_invalid(Z)
    ax.plot_surface(Rg, SPg, Zm, cmap='viridis', alpha=.85,
                    lw=0, antialiased=True)
    r, s, sp = res['opt_E']['alloc']
    ax.scatter([r], [sp], [res['opt_E']['val']], c='red', s=250,
              zorder=10, edgecolors='black', linewidths=1)
    ax.set_xlabel('Research %'); ax.set_ylabel('Speed %')
    ax.set_zlabel('E[PnL]')
    ax.set_title(f'E[PnL]  (opt R={r} S={s} Sp={sp})', fontsize=10)

    # 2: heatmap expected PnL
    ax2 = fig.add_subplot(232)
    Zp = Z.copy(); Zp[np.isnan(Zp)] = np.nanmin(Z)
    im = ax2.imshow(Zp, origin='lower', extent=[0,100,0,100],
                    aspect='auto', cmap='viridis')
    ax2.plot(r, sp, 'r*', ms=20, label=f'opt ({r},{100-r-sp},{sp})')
    ax2.set_xlabel('Research %'); ax2.set_ylabel('Speed %')
    ax2.set_title('E[PnL] heatmap'); ax2.legend(fontsize=8)
    fig.colorbar(im, ax=ax2, shrink=.82)

    # 3: speed multiplier CDF
    ax3 = fig.add_subplot(233)
    for v in [0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100]:
        ms = np.sort(res['sp_m'][v])
        ax3.plot(ms, np.linspace(0, 1, len(ms)), lw=1.5, label=f'sp={v}')
    ax3.set_xlabel('Speed multiplier'); ax3.set_ylabel('CDF')
    ax3.set_title('Speed mult. CDF by allocation')
    ax3.legend(fontsize=7, ncol=2); ax3.grid(alpha=.3)

    # 4-6: confidence heatmaps for selected targets
    sel = [k for k in res['targets']
           if 0.03 < res['opt_Pr'][k]['prob'] < 0.99]
    if len(sel) >= 3:
        pick = [sel[0], sel[len(sel)//2], sel[-1]]
    elif sel:
        pick = sel[:3]
    else:
        pick = res['targets'][:3]

    for pi, k in enumerate(pick):
        ax4 = fig.add_subplot(234 + pi)
        Zk = _grid(res, None, target=k)
        Zk[np.isnan(Zk)] = 0
        im = ax4.imshow(Zk, origin='lower', extent=[0,100,0,100],
                        aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        d = res['opt_Pr'][k]; ri, si, spi = d['alloc']
        ax4.plot(ri, spi, 'k*', ms=16)
        ax4.set_xlabel('Research %'); ax4.set_ylabel('Speed %')
        ax4.set_title(
            f"P(PnL ≥ {k:,})\nR={ri} S={si} Sp={spi}  P={d['prob']:.3f}",
            fontsize=9)
        fig.colorbar(im, ax=ax4, shrink=.82)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('imc_main.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → saved imc_main.png")


def plot_pnl_dist(res):
    """PnL histogram + CDF for the expected-optimal allocation."""
    r, s, sp = res['opt_E']['alloc']
    pnl = float(R(r)) * float(S(s)) * res['sp_m'][sp] - BUDGET

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'PnL distribution  R={r}%  S={s}%  Sp={sp}%', fontsize=12)

    ax1.hist(pnl, bins=80, density=True, color='steelblue',
             edgecolor='white', lw=.4)
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
            ax2.text(sp_pnl[0], 1-p+.015,
                     f'P(≥{k/1e3:.0f}k)={p:.2f}', fontsize=7)
    ax2.set_xlabel('PnL'); ax2.set_ylabel('CDF'); ax2.grid(alpha=.3)
    ax2.set_title('CDF with target thresholds')

    plt.tight_layout()
    plt.savefig('imc_pnl_dist.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → saved imc_pnl_dist.png")


def plot_deterministic():
    """PnL vs Research % for sp=0, at fixed multiplier values."""
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
    ax.set_title('Deterministic PnL  (fixed speed multiplier, Speed = 0)')
    ax.legend(ncol=3, fontsize=9); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig('imc_deterministic.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  → saved imc_deterministic.png")

# ── sensitivity analysis ─────────────────────────────────────

def sensitivity():
    """Re-run under different bimodal configurations."""
    configs = {
        'Strong bimodal (25/25)': dict(
            w_lo=0.25, a_lo=0.8, b_lo=15.0,
            w_mid=0.30, a_mid=2.0, b_mid=2.5,
            w_hi=0.25, a_hi=15.0, b_hi=0.8,
            jitter_frac=0.15),
        'Baseline bimodal (15/15)': dict(
            w_lo=0.15, a_lo=1.2, b_lo=12.0,
            w_mid=0.55, a_mid=2.0, b_mid=2.5,
            w_hi=0.15, a_hi=12.0, b_hi=1.2,
            jitter_frac=0.15),
        'Top-heavy (10/30)': dict(
            w_lo=0.10, a_lo=1.2, b_lo=12.0,
            w_mid=0.40, a_mid=2.5, b_mid=2.5,
            w_hi=0.30, a_hi=12.0, b_hi=1.2,
            jitter_frac=0.15),
        'Bottom-heavy (30/10)': dict(
            w_lo=0.30, a_lo=1.2, b_lo=12.0,
            w_mid=0.40, a_mid=2.0, b_mid=3.0,
            w_hi=0.10, a_hi=12.0, b_hi=1.2,
            jitter_frac=0.15),
        'Mostly middle (5/5)': dict(
            w_lo=0.05, a_lo=1.2, b_lo=12.0,
            w_mid=0.80, a_mid=2.0, b_mid=2.5,
            w_hi=0.05, a_hi=12.0, b_hi=1.2,
            jitter_frac=0.15),
        'High jitter (±30%)': dict(
            w_lo=0.15, a_lo=1.2, b_lo=12.0,
            w_mid=0.55, a_mid=2.0, b_mid=2.5,
            w_hi=0.15, a_hi=12.0, b_hi=1.2,
            jitter_frac=0.30),
    }
    print("  ► Sensitivity to opponent speed distribution shape\n")
    print(f"    {'Config':>30}   {'R':>3} {'S':>3} {'Sp':>3}"
          f"   {'E[PnL]':>10}  |  200k: R  S  Sp   P(≥200k)")
    print(f"    {'-'*78}")

    for label, cfg in configs.items():
        rng = np.random.default_rng(99)
        sp_m = {sp: speed_samples(sp, N_MC, rng, cfg) for sp in range(101)}

        best_E, best_a = -np.inf, (0, 0, 0)
        best_p200, best_a200 = -1, (0, 0, 0)
        for sp in range(101):
            m = sp_m[sp]; rem = 100 - sp
            for r in range(rem + 1):
                s   = rem - r
                pnl = float(R(r)) * float(S(s)) * m - BUDGET
                ev  = pnl.mean()
                if ev > best_E:
                    best_E, best_a = ev, (r, s, sp)
                p2 = (pnl >= 200_000).mean()
                if p2 > best_p200:
                    best_p200, best_a200 = p2, (r, s, sp)

        r, s, sp = best_a
        r2, s2, sp2 = best_a200
        print(f"    {label:>30}   {r:>3} {s:>3} {sp:>3}"
              f"   {best_E:>10,.0f}  |  {r2:>3} {s2:>3} {sp2:>3}   {best_p200:.3f}")
    print()

# ── main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    np.set_printoptions(linewidth=120)

    print("=" * 72)
    print("  IMC Prosperity 4 – Manual Challenge Optimiser")
    print("  «Invest & Expand»  — Bimodal Speed Model")
    print("=" * 72)
    print(f"  Budget       = {BUDGET:>10,} XIRECs")
    print(f"  Players (≈)  = {N_PLAYERS:>10,}")
    print(f"  MC samples   = {N_MC:>10,}")
    sd = SPEED_DIST
    print(f"  Speed dist   : lo={sd['w_lo']:.0%} mid={sd['w_mid']:.0%}"
          f" hi={sd['w_hi']:.0%}"
          f" uni={1-sd['w_lo']-sd['w_mid']-sd['w_hi']:.0%}")
    print()

    # ── Show assumed distribution ──
    print("  Plotting assumed opponent speed distribution …")
    plot_population_dist()

    targets = [150_000, 160_000, 170_000, 180_000, 190_000, 200_000]

    # ── Optimise ──
    res = optimise(targets)
    show(res)

    # ── Deterministic reference ──
    print("  ► Deterministic optimum  (sp = 0, fixed m)")
    for m in np.arange(0.1, 1.0, 0.1):
        r_a = np.arange(101)
        pv  = R(r_a) * S(100 - r_a) * m - BUDGET
        j   = pv.argmax()
        print(f"    m={m:.1f}:  R={j:>3}%  S={100-j:>3}%  Sp=0%"
              f"   PnL = {pv[j]:>10,.0f}")
    print()

    # ── Sensitivity ──
    print("=" * 72)
    print("  SENSITIVITY ANALYSIS")
    print("=" * 72)
    sensitivity()

    # ── Plots ──
    print("  Generating plots …")
    plot_main(res)
    plot_pnl_dist(res)
    plot_deterministic()

    print("\n  All done ✅")