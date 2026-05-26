#!/usr/bin/env python3
"""
Experiment 2: Exact LAP Cost vs. Spectral Distance for Head Matching.

Compares two strategies for inter-head alignment in the wreath product framework:
  - Exact LAP cost: enumerate all permutations to find optimal head alignment.
  - Spectral proxy: use Frobenius distance between singular value spectra.

Setup:
  d_m = 16, H = 4, d_k = 4
  Two random MHA weight sets W^A, W^B (randomly initialized with different seeds).
  100 trials.

For each trial:
  1. Compute the exact cost matrix (head‑pair) via brute‑force over S_{d_k} (24 perms).
  2. Compute the spectral distance matrix.
  3. Solve both assignment problems with the Hungarian algorithm.
  4. Score each matching by the optimal inner product.
  5. Record scores, equality of matchings, and the score gap.

Produces two figures:
  - Figure 1: exp2a_boxplot.pdf  (violin + scatter, Exact LAP vs Spectral Proxy)
  - Figure 2: exp2b_gap.pdf      (histogram of Score_exact - Score_spec)
"""

import os
import sys
from itertools import permutations

import numpy as np
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
d_m = 16          # model dimension
H = 4             # number of heads
d_k = 4           # head dimension
N_TRIALS = 100

FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures",
)

# Precompute all permutations of {0, …, d_k-1}  (4! = 24)
ALL_PERMS = list(permutations(range(d_k)))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mha_weights(seed: int) -> dict:
    """
    Return a dict {'Q': array(H, d_m, d_k), 'K': …, 'V': …}
    with random Gaussian entries.
    """
    rng = np.random.RandomState(seed)
    weights = {}
    for proj in ("Q", "K", "V"):
        weights[proj] = rng.randn(H, d_m, d_k).astype(np.float64)
    return weights


def _pairwise_summed_gram(WB_h: np.ndarray, WA_h: np.ndarray) -> np.ndarray:
    """
    For a single head pair (B-head i, A-head j), compute the d_k x d_k matrix

        M[a,b] = sum_{proj ∈ {Q,K,V}} sum_p  WB_h^proj[p,a] * WA_h^proj[p,b]

    This is the summed Gram matrix over the three projection types, so

        max_τ  sum_{proj} ⟨W_i^B, τ·W_j^A⟩  =  max_τ  sum_a M[a, τ(a)].
    """
    M = np.zeros((d_k, d_k), dtype=np.float64)
    for proj_key in ("Q", "K", "V"):
        Wi = WB_h[proj_key]   # (d_m, d_k)
        Wj = WA_h[proj_key]   # (d_m, d_k)
        M += Wi.T @ Wj        # (d_k, d_k)
    return M


def _optimal_inner(M: np.ndarray) -> float:
    """Return max_{τ ∈ S_{d_k}} Σ_a M[a, τ(a)] via brute‑force enumeration."""
    best = -np.inf
    for perm in ALL_PERMS:
        val = sum(M[a, perm[a]] for a in range(d_k))
        if val > best:
            best = val
    return float(best)


def _spectral_distance(WB_h: dict, WA_h: dict) -> float:
    """
    D_{ij} = Σ_{proj} || σ(W_i^B) - σ(W_j^A) ||_F
    where σ(·) are singular values.
    """
    d = 0.0
    for proj_key in ("Q", "K", "V"):
        Wi = WB_h[proj_key]
        Wj = WA_h[proj_key]
        # singular values in descending order, length min(d_m, d_k) = d_k
        si = np.linalg.svd(Wi, full_matrices=False, compute_uv=False)
        sj = np.linalg.svd(Wj, full_matrices=False, compute_uv=False)
        d += float(np.linalg.norm(si - sj))
    return d


# ---------------------------------------------------------------------------
# Core experiment logic
# ---------------------------------------------------------------------------

def exact_cost_matrix(WA: dict, WB: dict) -> np.ndarray:
    """
    c[i,j] = - max_{τ} Σ_{proj} ⟨W_i^B, τ·W_j^A⟩
    (Hungarian *minimises* total cost → maximises total inner product.)
    """
    cost = np.zeros((H, H), dtype=np.float64)
    for i in range(H):
        WB_i = {k: WB[k][i] for k in ("Q", "K", "V")}
        for j in range(H):
            WA_j = {k: WA[k][j] for k in ("Q", "K", "V")}
            M = _pairwise_summed_gram(WB_i, WA_j)
            cost[i, j] = -_optimal_inner(M)
    return cost


def spectral_distance_matrix(WA: dict, WB: dict) -> np.ndarray:
    """D[i,j] = Σ_{proj} ||σ(W_i^B) - σ(W_j^A)||_F."""
    dist = np.zeros((H, H), dtype=np.float64)
    for i in range(H):
        WB_i = {k: WB[k][i] for k in ("Q", "K", "V")}
        for j in range(H):
            WA_j = {k: WA[k][j] for k in ("Q", "K", "V")}
            dist[i, j] = _spectral_distance(WB_i, WA_j)
    return dist


def alignment_score(matching: np.ndarray, WA: dict, WB: dict) -> float:
    """
    Score(σ) = Σ_i  max_τ  Σ_{proj}  ⟨W_i^B, τ·W_{σ(i)}^A⟩.
    matching[i] = j  →  B-head i matched to A-head j.
    """
    total = 0.0
    for i in range(H):
        j = matching[i]
        WB_i = {k: WB[k][i] for k in ("Q", "K", "V")}
        WA_j = {k: WA[k][j] for k in ("Q", "K", "V")}
        M = _pairwise_summed_gram(WB_i, WA_j)
        total += _optimal_inner(M)
    return total


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)

    scores_exact = []
    scores_spec = []
    matchings_equal = []

    print("=" * 62)
    print(" Experiment 2: Exact LAP Cost vs. Spectral Distance ")
    print("=" * 62)
    print(f"  d_m = {d_m},  H = {H},  d_k = {d_k},  trials = {N_TRIALS}")
    print(f"  |S_{{d_k}}| = {len(ALL_PERMS)}  (brute-force enumeration)")
    print("-" * 62)

    for trial in range(N_TRIALS):
        seed_a = 1000 + trial * 3
        seed_b = 2000 + trial * 3

        WA = make_mha_weights(seed_a)
        WB = make_mha_weights(seed_b)

        # ---- exact matching ------------------------------------------------
        C = exact_cost_matrix(WA, WB)
        _, col_exact = linear_sum_assignment(C)         # minimises Σ c[i, σ(i)]
        matching_exact = col_exact

        # ---- spectral matching --------------------------------------------
        D = spectral_distance_matrix(WA, WB)
        _, col_spec = linear_sum_assignment(D)
        matching_spec = col_spec

        # ---- scores -------------------------------------------------------
        se = alignment_score(matching_exact, WA, WB)
        ss = alignment_score(matching_spec, WA, WB)

        scores_exact.append(se)
        scores_spec.append(ss)
        matchings_equal.append(bool(np.array_equal(matching_exact, matching_spec)))

    scores_exact = np.array(scores_exact)
    scores_spec  = np.array(scores_spec)
    matchings_equal = np.array(matchings_equal)

    # ---- summary statistics -----------------------------------------------
    gap = scores_exact - scores_spec
    frac_better = (gap > 0).mean()
    frac_equal  = np.isclose(gap, 0.0).mean()
    frac_match_id = matchings_equal.mean()
    mean_rel_impr = (
        100.0 * gap.mean() / abs(scores_spec.mean())
        if abs(scores_spec.mean()) > 1e-12
        else 0.0
    )

    print(f"\n  Score_exact  :  mean = {scores_exact.mean():.4f}, "
          f"std = {scores_exact.std():.4f}")
    print(f"  Score_spec   :  mean = {scores_spec.mean():.4f}, "
          f"std = {scores_spec.std():.4f}")
    print(f"  Score gap    :  mean = {gap.mean():.6f}, std = {gap.std():.6f}")
    print()
    print(f"  Trials where exact  > spectral : {100*frac_better:.1f} %")
    print(f"  Trials where exact == spectral : {100*frac_equal:.1f} %")
    print(f"  Trials with identical matchings: {100*frac_match_id:.1f} %")
    print(f"  Mean relative improvement       : {mean_rel_impr:.4f} %")
    print()

    # ---- Figure 1: boxplot / violin ---------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7.5, 5.5))
    positions = [1, 2]
    data = [scores_exact, scores_spec]
    labels = ["Exact LAP", "Spectral Proxy"]
    colors_violin = ["#2196F3", "#FF9800"]
    colors_scatter = ["#1565C0", "#E65100"]

    vp = ax1.violinplot(data, positions=positions, showmeans=True,
                        showmedians=True, widths=0.55)
    for body, fc in zip(vp["bodies"], colors_violin):
        body.set_facecolor(fc)
        body.set_alpha(0.55)
    for part in ("cmeans", "cmedians"):
        vp[part].set_color("#333333")
        vp[part].set_linewidth(1.2)

    rng_jitter = np.random.RandomState(2026)
    for idx, (d, pos) in enumerate(zip(data, positions)):
        jitter = rng_jitter.uniform(-0.08, 0.08, size=len(d))
        ax1.scatter(np.full_like(d, pos) + jitter, d,
                    alpha=0.35, s=18, color=colors_scatter[idx],
                    edgecolors="none", zorder=5, label="_nolegend_")

    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, fontsize=13)
    ax1.set_ylabel("Alignment Score (inner product)", fontsize=13)
    ax1.set_title("Head Matching: Exact LAP vs. Spectral Proxy",
                  fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.25)

    fig1.tight_layout()
    path1 = os.path.join(FIGURES_DIR, "exp2a_boxplot.pdf")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ---- Figure 2: gap histogram ------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.8))

    ax2.hist(gap, bins=22, color="#607D8B", edgecolor="white",
             alpha=0.85, linewidth=0.6)
    ax2.axvline(0.0, color="#D32F2F", linestyle="--", linewidth=2.2,
                label="Zero gap")
    ax2.legend(fontsize=11, loc="upper right")

    ax2.set_xlabel(r"Score$_{\rm exact}$ – Score$_{\rm spectral}$", fontsize=13)
    ax2.set_ylabel("Frequency", fontsize=13)
    ax2.set_title(
        f"Alignment Score Gap Distribution\n"
        f"Exact > Spectral in {100*frac_better:.1f} % of {N_TRIALS} trials",
        fontsize=14, fontweight="bold",
    )
    ax2.grid(axis="y", alpha=0.25)

    fig2.tight_layout()
    path2 = os.path.join(FIGURES_DIR, "exp2b_gap.pdf")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ---- report -----------------------------------------------------------
    print(f"  Figure 1 saved: {path1}  ({os.path.getsize(path1):,} bytes)")
    print(f"  Figure 2 saved: {path2}  ({os.path.getsize(path2):,} bytes)")
    print("=" * 62)
    print(" Done.\n")
