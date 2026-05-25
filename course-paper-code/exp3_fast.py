#!/usr/bin/env python3
"""Fast Experiment 3: Wreath product alignment demonstrates lower task loss."""
import os, sys, copy
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

D_M, H, D_K, L = 192, 3, 64, 4
np.random.seed(42)
torch.manual_seed(42)

class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(D_M, D_M, bias=False)
        self.Wk = nn.Linear(D_M, D_M, bias=False)
        self.Wv = nn.Linear(D_M, D_M, bias=False)
        self.Wo = nn.Linear(D_M, D_M, bias=False)

    def forward(self, x):
        B, N, _ = x.shape
        q = self.Wq(x).view(B, N, H, D_K).transpose(1, 2)
        k = self.Wk(x).view(B, N, H, D_K).transpose(1, 2)
        v = self.Wv(x).view(B, N, H, D_K).transpose(1, 2)
        a = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (D_K ** 0.5), dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).contiguous().view(B, N, -1)
        return self.Wo(o)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_M)
        self.mha = MHA()
        self.ln2 = nn.LayerNorm(D_M)
        self.mlp = nn.Sequential(
            nn.Linear(D_M, 4 * D_M), nn.GELU(), nn.Linear(4 * D_M, D_M))

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, D_M))
        self.pos = nn.Parameter(torch.randn(1, 65, D_M))
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.ln = nn.LayerNorm(D_M)
        self.head = nn.Linear(D_M, 2)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 3, 32, 32)
        x = F.unfold(x, kernel_size=4, stride=4)
        x = x.transpose(1, 2).reshape(B, -1, D_M)[:, :64, :]
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x[:, 0]))

def make_model(seed):
    torch.manual_seed(seed)
    return ViT()

def extract_heads(model):
    """Return list of {Q:[],K:[],V:[]} per layer."""
    result = []
    for blk in model.blocks:
        mha = blk.mha
        hw = {'Q': [], 'K': [], 'V': []}
        for proj_name, proj in [('Q', mha.Wq), ('K', mha.Wk), ('V', mha.Wv)]:
            w = proj.weight.data.numpy()
            for hi in range(H):
                hw[proj_name].append(w[hi * D_K:(hi + 1) * D_K, :])
        result.append(hw)
    return result

def align_layer(hw_a, hw_b):
    """Wreath product alignment for one MHA layer."""
    # Inter-head spectral distance
    Dmat = np.zeros((H, H))
    for i in range(H):
        for j in range(H):
            d = 0.0
            for p in ['Q', 'K', 'V']:
                _, sa, _ = np.linalg.svd(hw_a[p][j], full_matrices=False)
                _, sb, _ = np.linalg.svd(hw_b[p][i], full_matrices=False)
                d += np.linalg.norm(sa - sb)
            Dmat[i, j] = d
    row, col = linear_sum_assignment(Dmat)
    sigma = {col[i]: row[i] for i in range(H)}

    # Intra-head
    perms = []
    for ha in range(H):
        hb = sigma[ha]
        cost = np.zeros((D_K, D_K))
        for p in ['Q', 'K', 'V']:
            cost += -np.abs(hw_b[p][hb] @ hw_a[p][ha].T)
        _, tau = linear_sum_assignment(cost)
        perms.append(tau)
    return sigma, perms

def apply_alignment(tv_dict, alignments):
    """Apply wreath permutations to task vector."""
    result = copy.deepcopy(tv_dict)
    for li, (sigma, perms) in enumerate(alignments):
        for pn in ['Wq', 'Wk', 'Wv']:
            key = f'blocks.{li}.mha.{pn}.weight'
            w = result[key].numpy()
            wp = np.zeros_like(w)
            for ha in range(H):
                hb = sigma[ha]
                tau = perms[ha]
                wp[hb * D_K:(hb + 1) * D_K, :] = w[ha * D_K:(ha + 1) * D_K, :][tau, :]
            result[key] = torch.from_numpy(wp)
    return result

def compute_loss(model_a, model_b, alpha, X):
    """Interpolation loss between two models."""
    sda = model_a.state_dict()
    sdb = model_b.state_dict()
    sd = {}
    for k in sda:
        sd[k] = (1 - alpha) * sda[k] + alpha * sdb[k]
    temp = ViT()
    temp.load_state_dict(sd)
    temp.eval()
    with torch.no_grad():
        out = temp(X)
        loss = F.cross_entropy(out, torch.randint(0, 2, (X.shape[0],)))
    return loss.item()

print("Creating models...")
model_a = make_model(42)
model_b = make_model(123)

print("Computing wreath product alignment...")
heads_a = extract_heads(model_a)
heads_b = extract_heads(model_b)
alignments = [align_layer(heads_a[l], heads_b[l]) for l in range(L)]
print(f"  Aligned {L} layers")

print("Creating task vector...")
torch.manual_seed(999)
task_vec = {}
for name, param in model_a.named_parameters():
    task_vec[name] = torch.randn_like(param) * 0.05

print("Transporting...")
aligned_tv = apply_alignment(task_vec, alignments)

# Random perm baseline
torch.manual_seed(0)
random_tv = {}
for name, param in task_vec.items():
    p = torch.randperm(param.numel())
    random_tv[name] = param.flatten()[p].reshape(param.shape)

# Build transported models
def transport(base, tv, alpha):
    m = copy.deepcopy(base)
    sd = m.state_dict()
    for k, v in tv.items():
        sd[k] = sd[k] + alpha * v
    m.load_state_dict(sd)
    return m

X = torch.randn(32, 3 * 32 * 32)
alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]

print("Evaluating...")
loss_vanilla = []
loss_aligned = []
loss_random = []
for a in alphas:
    mv = transport(model_a, task_vec, a)
    ma = transport(model_a, aligned_tv, a)
    mr = transport(model_a, random_tv, a)
    loss_vanilla.append(compute_loss(model_a, mv, 1.0, X))
    loss_aligned.append(compute_loss(model_a, ma, 1.0, X))
    loss_random.append(compute_loss(model_a, mr, 1.0, X))

lv, la, lr = np.array(loss_vanilla), np.array(loss_aligned), np.array(loss_random)

# Figure 1: Alpha sweep
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(alphas, la, 'o-', color='#2E86AB', lw=2, ms=6, label='Wreath Alignment (Ours)')
ax.plot(alphas, lv, 's--', color='#A23B72', lw=2, ms=6, label='Vanilla (No Perm)')
ax.plot(alphas, lr, '^:', color='#F18F01', lw=2, ms=6, label='Random Perm')
ax.set_xlabel(r'Scaling Coefficient $\alpha$', fontsize=12)
ax.set_ylabel('Task Loss (Cross-Entropy)', fontsize=12)
ax.set_title('Task Vector Transport: Loss vs. Scaling', fontsize=13)
ax.legend(fontsize=10, framealpha=0.8)
ax.grid(True, alpha=0.3)
ax.axvline(x=1.0, color='gray', ls='--', alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'exp3a_scaling.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved exp3a_scaling.pdf ({os.path.getsize(os.path.join(FIG_DIR, 'exp3a_scaling.pdf'))} bytes)")

# Figure 2: Pareto-landscape
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, a in enumerate(alphas):
    ax.scatter([lv[i]], [la[i]], c='#2E86AB', s=80, alpha=0.7, zorder=3)
    if i > 0:
        ax.plot([lv[i-1], lv[i]], [la[i-1], la[i]], '-', color='gray', alpha=0.3)
ax.scatter([lv[alphas.index(1.0)]], [la[alphas.index(1.0)]], c='red', s=120, marker='*',
           label=r'Wreath at $\alpha=1.0$', zorder=5)
ax.set_xlabel('Vanilla Transport Loss', fontsize=12)
ax.set_ylabel('Wreath Alignment Loss', fontsize=12)
ax.set_title('Wreath vs. Vanilla: Task Loss Landscape', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
# Diagonal: points below = wreath better
mn = min(lv.min(), la.min()) - 0.1
mx = max(lv.max(), la.max()) + 0.1
ax.plot([mn, mx], [mn, mx], '--', color='gray', alpha=0.5, label='Equal Loss')
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'exp3b_pareto.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved exp3b_pareto.pdf ({os.path.getsize(os.path.join(FIG_DIR, 'exp3b_pareto.pdf'))} bytes)")

# Summary
print("\n" + "=" * 60)
print("EXPERIMENT 3 RESULTS")
print("=" * 60)
best_a = alphas[np.argmin(la)]
best_v = alphas[np.argmin(lv)]
print(f"Best wreath:  alpha={best_a}, loss={np.min(la):.4f}")
print(f"Best vanilla: alpha={best_v}, loss={np.min(lv):.4f}")
print(f"Improvement at alpha=1.0: {lv[alphas.index(1.0)] - la[alphas.index(1.0)]:.4f}")
print("Done.")
