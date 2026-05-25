#!/usr/bin/env python3
"""
Experiment 1: Wreath Product Verification for Theorem 1
========================================================
Verifies that the wreath product group W(H, d_k) = S_H ≀ S_{d_k} EXACTLY
characterizes the set of permutations that preserve multi-head attention
functional equivalence.

Setup: d_m = 8, H = 2, d_k = 4, 40320 permutations of S_8.
"""

import numpy as np
from itertools import permutations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import time
import os

# ============================================================
# Configuration
# ============================================================
np.random.seed(42)

d_m = 8
H = 2
d_k = 4
n_tokens = 4

assert d_m == H * d_k, f"d_m ({d_m}) must equal H* d_k ({H * d_k})"

# Output paths
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
FIGURE_PATH = os.path.join(FIGURES_DIR, 'exp1_wreath_verification.pdf')

# ============================================================
# Generate random weights and input
# ============================================================
W_Q = np.random.randn(d_m, d_m).astype(np.float64)
W_K = np.random.randn(d_m, d_m).astype(np.float64)
W_V = np.random.randn(d_m, d_m).astype(np.float64)
W_O = np.eye(d_m, dtype=np.float64)  # Identity: no output projection

X = np.random.randn(n_tokens, d_m).astype(np.float64)

# ============================================================
# MHA Forward Pass
# ============================================================
def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def mha_forward(X_in, Wq, Wk, Wv, Wo, num_heads, head_dim):
    """
    Multi-head attention forward pass.

    Args:
        X_in:   (n, d_m) input tokens
        Wq,Wk,Wv: (d_m, d_m) projection matrices
        Wo:     (d_m, d_m) output projection
        num_heads: H
        head_dim: d_k

    Returns:
        (n, d_m) output
    """
    n, dm = X_in.shape
    # Linear projections: Q = X @ W^T
    Q = X_in @ Wq.T   # (n, d_m)
    K = X_in @ Wk.T
    V = X_in @ Wv.T

    # Reshape to (n, H, d_k) then transpose to (H, n, d_k)
    Q = Q.reshape(n, num_heads, head_dim).transpose(1, 0, 2)
    K = K.reshape(n, num_heads, head_dim).transpose(1, 0, 2)
    V = V.reshape(n, num_heads, head_dim).transpose(1, 0, 2)

    # Scaled dot-product attention
    scale = np.sqrt(head_dim)
    attn_scores = Q @ K.transpose(0, 2, 1) / scale   # (H, n, n)
    attn_weights = softmax(attn_scores, axis=-1)
    attn_out = attn_weights @ V                        # (H, n, d_k)

    # Concatenate heads: (H, n, d_k) -> (n, H, d_k) -> (n, d_m)
    concat = attn_out.transpose(1, 0, 2).reshape(n, dm)

    # Output projection
    out = concat @ Wo.T
    return out


# Compute reference output with original weights
O_orig = mha_forward(X, W_Q, W_K, W_V, W_O, H, d_k)
print(f"O_orig shape: {O_orig.shape}")
print(f"O_orig norm: {np.linalg.norm(O_orig, 'fro'):.6f}")

# ============================================================
# Generate all wreath product permutations W(H, d_k) = S_H ≀ S_{d_k}
# ============================================================
print("\nGenerating wreath product elements...")

S_H_list = list(permutations(range(H)))      # 2 elements
S_dk_list = list(permutations(range(d_k)))   # 24 elements

wreath_set = set()
for sigma in S_H_list:
    for tau_1 in S_dk_list:
        for tau_2 in S_dk_list:
            taus = [tau_1, tau_2]
            # Build permutation tuple p where p[old_idx] = new_idx
            p_list = [0] * d_m
            for h in range(H):
                target_head = sigma[h]
                for j in range(d_k):
                    old_idx = h * d_k + j
                    new_idx = target_head * d_k + taus[h][j]
                    p_list[old_idx] = new_idx
            wreath_set.add(tuple(p_list))

print(f"Wreath product elements: {len(wreath_set)}  (expected: 2 * 24^2 = 1152)")

# ============================================================
# Generate all S_{d_m} permutations and classify
# ============================================================
print("Generating all S_8 permutations...")
all_perms = list(permutations(range(d_m)))
n_total = len(all_perms)
print(f"Total S_{d_m} elements: {n_total}")

# Pre-classify: is_wreath[i] = True iff all_perms[i] is in wreath_set
is_wreath = np.zeros(n_total, dtype=bool)
for i, p in enumerate(all_perms):
    is_wreath[i] = (p in wreath_set)

n_wreath = np.sum(is_wreath)
n_non_wreath = n_total - n_wreath
print(f"Wreath: {n_wreath}, Non-wreath: {n_non_wreath}")

# ============================================================
# Precompute inverse permutations for all permutations
# (p_inv[j] = i such that p[i] = j, i.e., p^(-1)(j))
# ============================================================
print("Precomputing inverse permutations...")
inv_perms = np.zeros((n_total, d_m), dtype=np.int32)
for i, p in enumerate(all_perms):
    for j in range(d_m):
        inv_perms[i, p[j]] = j

# ============================================================
# Verify all permutations
# ============================================================
print("\nVerifying all permutations (this may take a minute)...")
start_time = time.time()

wreath_errors_list = []
non_wreath_errors_list = []

# We'll process in batches for progress reporting
batch_size = 5000
for start in range(0, n_total, batch_size):
    end = min(start + batch_size, n_total)
    batch_indices = range(start, end)
    n_batch = end - start

    # Build batch of permutation matrices P[b, :, :]
    # P[b, j, i] = 1 if p_b[i] = j, i.e., P_b is the permutation matrix for p_b
    P_batch = np.zeros((n_batch, d_m, d_m), dtype=np.float64)
    for k, idx in enumerate(batch_indices):
        p = all_perms[idx]
        for i in range(d_m):
            P_batch[k, p[i], i] = 1.0

    # Apply permutations to weight matrices: f_P(W) = P @ W
    # W_Q_perm[k] = P_batch[k] @ W_Q
    W_Q_perm = P_batch @ W_Q   # (n_batch, d_m, d_m)
    W_K_perm = P_batch @ W_K
    W_V_perm = P_batch @ W_V

    # Compute MHA outputs for each permuted weight set
    for k, idx in enumerate(batch_indices):
        O_new = mha_forward(X,
                            W_Q_perm[k],
                            W_K_perm[k],
                            W_V_perm[k],
                            W_O, H, d_k)

        # Reference: P @ O_orig (permuting output dimension)
        # O_ref[n, j] = O_orig[n, p^{-1}(j)]
        p_inv = inv_perms[idx]
        O_ref = O_orig[:, p_inv]  # (n, d_m)

        error = np.linalg.norm(O_new - O_ref, 'fro')

        if is_wreath[idx]:
            wreath_errors_list.append(error)
        else:
            non_wreath_errors_list.append(error)

    elapsed = time.time() - start_time
    print(f"  Processed {end}/{n_total} permutations ({elapsed:.1f}s elapsed)...")

total_time = time.time() - start_time
print(f"\nTotal verification time: {total_time:.1f}s")

# Convert to arrays
wreath_errors = np.array(wreath_errors_list, dtype=np.float64)
non_wreath_errors = np.array(non_wreath_errors_list, dtype=np.float64)

# ============================================================
# Summary Statistics
# ============================================================
eps_threshold = 1e-10
wreath_below = np.sum(wreath_errors < eps_threshold)
non_wreath_below = np.sum(non_wreath_errors < eps_threshold)

print(f"\n{'='*60}")
print(f"SUMMARY STATISTICS")
print(f"{'='*60}")
print(f"Wreath elements with epsilon < {eps_threshold:.0e}: "
      f"{wreath_below}/{len(wreath_errors)}")
print(f"Non-wreath elements with epsilon < {eps_threshold:.0e}: "
      f"{non_wreath_below}/{len(non_wreath_errors)}")
print(f"---")
print(f"Wreath errors:")
print(f"  Max:  {np.max(wreath_errors):.6e}")
print(f"  Mean: {np.mean(wreath_errors):.6e}")
print(f"  Median: {np.median(wreath_errors):.6e}")
print(f"---")
print(f"Non-wreath errors:")
if len(non_wreath_errors) > 0:
    print(f"  Min:    {np.min(non_wreath_errors):.6e}")
    print(f"  Median: {np.median(non_wreath_errors):.6e}")
    print(f"  Mean:   {np.mean(non_wreath_errors):.6e}")
    print(f"  Max:    {np.max(non_wreath_errors):.6e}")
print(f"{'='*60}")

# ============================================================
# Generate Figure
# ============================================================
print(f"\nGenerating figure: {FIGURE_PATH}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                gridspec_kw={'width_ratios': [1, 2]})

# --- Left panel: log-scale histogram of errors for wreath elements ---
# Wreath errors should all be near machine epsilon (~1e-15 to ~1e-13)
log_wreath = np.log10(np.maximum(wreath_errors, 1e-18))
bins_w = np.linspace(-18, -8, 51)
ax1.hist(log_wreath, bins=bins_w, color='steelblue', edgecolor='navy',
         alpha=0.85, linewidth=0.5)
ax1.set_xlabel('log10(error)')
ax1.set_ylabel('Count')
ax1.set_title(f'Wreath Elements (n={len(wreath_errors)})')
ax1.axvline(np.log10(eps_threshold), color='black', linestyle='--',
            linewidth=1, label=f'threshold = {eps_threshold:.0e}')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# --- Right panel: log-scale histogram comparing both classes ---
# Use log-spaced bins for the non-wreath errors (they span many orders)
log_non = np.log10(np.maximum(non_wreath_errors, 1e-18))

# Compute bin edges spanning the full range
all_log = np.concatenate([log_wreath, log_non])
bin_min = np.floor(np.min(all_log))
bin_max = np.ceil(np.max(all_log))
bins = np.linspace(bin_min, bin_max, 81)

ax2.hist(log_wreath, bins=bins, color='steelblue', edgecolor='navy',
         alpha=0.7, linewidth=0.3, label=f'Wreath (n={len(wreath_errors)})')
ax2.hist(log_non, bins=bins, color='crimson', edgecolor='darkred',
         alpha=0.6, linewidth=0.3, label=f'Non-Wreath (n={len(non_wreath_errors)})')

ax2.set_xlabel('log10(error)')
ax2.set_ylabel('Count (log scale)')
ax2.set_yscale('log')
ax2.set_title('MHA Permutation Equivalence: Wreath vs. Non-Wreath')
ax2.axvline(np.log10(eps_threshold), color='black', linestyle='--',
            linewidth=1, label=f'threshold = {eps_threshold:.0e}')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('Theorem 1 Verification: Wreath Product $W(2,4) = S_2 \\wr S_4$ '
             'Characterizes MHA Functional Equivalence',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
fig.savefig(FIGURE_PATH, dpi=150, bbox_inches='tight')
plt.close(fig)

# Verify file was created
file_size = os.path.getsize(FIGURE_PATH)
print(f"Figure saved: {FIGURE_PATH}")
print(f"File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

# ============================================================
# Final Verification
# ============================================================
print(f"\n{'='*60}")
print(f"THEOREM 1 VERIFICATION RESULT")
print(f"{'='*60}")
# Theorem predicts: all wreath elements have eps ~ 0, no non-wreath element does
if wreath_below == len(wreath_errors) and non_wreath_below == 0:
    print("SUCCESS: Theorem 1 confirmed!")
    print(f"  - ALL {wreath_below} wreath product elements produce epsilon ~ 0")
    print(f"  - ZERO non-wreath elements produce epsilon ~ 0")
else:
    print(f"PARTIAL RESULTS:")
    print(f"  - {wreath_below}/{len(wreath_errors)} wreath elements <= {eps_threshold:.0e}")
    print(f"  - {non_wreath_below}/{len(non_wreath_errors)} non-wreath elements <= {eps_threshold:.0e}")
    if wreath_below < len(wreath_errors):
        print(f"  WARNING: Some wreath elements have error > {eps_threshold:.0e}")
        print(f"  This may be due to floating point precision issues.")
print(f"{'='*60}")

print("\nDone.")
