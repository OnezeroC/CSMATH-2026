#!/usr/bin/env python3
"""
Experiment 3: Re-Basin Pipeline with Wreath Product Alignment
Tiny Vision Transformer on CIFAR-10 binary classification (classes 0, 1).

Demonstrates that wreath product alignment works end-to-end for
task vector transportation across differently trained models.
"""

import copy
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # set to 0 on macOS to avoid multiprocessing deadlock

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEED_A = 42
SEED_B = 123
BATCH_SIZE = 128
LR = 3e-4
EPOCHS_BASE = 15
EPOCHS_FT = 10
FT_FRAC = 0.2  # 20% data for fine-tuning

# Model hyper-parameters
D_M = 192
H = 3
D_K = 64  # per-head dimension
L = 4     # number of transformer layers
PATCH_SIZE = 4
IMAGE_SIZE = 32
N_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 64

# Alignment parameters
N_ITER = 5  # alternating optimization iterations

# Alpha sweep
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def get_binary_cifar10(train=True, seed=None):
    """Load CIFAR-10 filtered to classes 0 and 1 only."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    ds = torchvision.datasets.CIFAR10(
        root="./data", train=train, download=True, transform=transform
    )
    # Filter classes 0 and 1
    targets = np.array(ds.targets) if isinstance(ds.targets, list) else ds.targets
    mask = (targets == 0) | (targets == 1)
    indices = np.where(mask)[0]
    if seed is not None:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    ds.data = ds.data[indices]
    ds.targets = [int(targets[i]) for i in indices]
    return ds


def subset_dataloader(ds, frac, batch_size, shuffle=True, seed=None):
    """Take a random subset (frac) of the dataset as a new DataLoader."""
    n = len(ds)
    n_sub = int(n * frac)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, n_sub, replace=False)
    sub_ds = torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(
        sub_ds, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS
    )


# ---------------------------------------------------------------------------
# Tiny Vision Transformer
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_m, h, d_k):
        super().__init__()
        self.d_m = d_m
        self.h = h
        self.d_k = d_k
        self.Wq = nn.Linear(d_m, h * d_k, bias=False)
        self.Wk = nn.Linear(d_m, h * d_k, bias=False)
        self.Wv = nn.Linear(d_m, h * d_k, bias=False)
        self.Wo = nn.Linear(h * d_k, d_m, bias=False)

    def forward(self, x):
        B, N, _ = x.shape
        q = self.Wq(x).view(B, N, self.h, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, N, self.h, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, N, self.h, self.d_k).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, -1)
        out = self.Wo(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_m, h, d_k):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_m)
        self.mha = MultiHeadAttention(d_m, h, d_k)
        self.ln2 = nn.LayerNorm(d_m)
        self.mlp = nn.Sequential(
            nn.Linear(d_m, 4 * d_m),
            nn.GELU(),
            nn.Linear(4 * d_m, d_m),
        )

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyViT(nn.Module):
    def __init__(self, d_m=D_M, h=H, d_k=D_K, L=L, num_classes=2):
        super().__init__()
        self.d_m = d_m
        self.h = h
        self.d_k = d_k
        self.L = L
        self.patch_size = PATCH_SIZE
        self.patch_embed = nn.Conv2d(3, d_m, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_m))
        self.pos_embed = nn.Parameter(torch.randn(1, N_PATCHES + 1, d_m))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_m, h, d_k) for _ in range(L)]
        )
        self.ln_final = nn.LayerNorm(d_m)
        self.head = nn.Linear(d_m, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N_patches, d_m)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                       # (B, N_patches+1, d_m)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_final(x[:, 0])  # CLS token
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0.0
    total = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total * 100.0


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0.0
    total = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
    return correct / total * 100.0


def train_model(model, train_loader, val_loader, epochs, lr, desc="Train"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_acc = evaluate(model, val_loader)
        if val_acc > best_acc:
            best_acc = val_acc
        print(f"[{desc}] Epoch {epoch:3d}/{epochs}: "
              f"Loss={train_loss:.4f}, TrainAcc={train_acc:.2f}%, ValAcc={val_acc:.2f}%")
    return best_acc


# ---------------------------------------------------------------------------
# Model serialisation helpers
# ---------------------------------------------------------------------------
def get_model_state_dicts(model):
    """Extract named parameter dict, plus structured sub-dicts for MHA layers."""
    return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}


def state_dict_to_device(sd, device):
    return {k: v.to(device) for k, v in sd.items()}


def load_state_into(model, sd):
    model.load_state_dict(sd)


def model_params_as_vector(model):
    return torch.cat([p.data.view(-1).float() for p in model.parameters()]).cpu()


# ---------------------------------------------------------------------------
# Task vector operations
# ---------------------------------------------------------------------------
def compute_task_vector(model_ft, model_base):
    """tau = theta_ft - theta_base (element-wise difference of state dicts)."""
    sd_ft = get_model_state_dicts(model_ft)
    sd_base = get_model_state_dicts(model_base)
    tau = {}
    for k in sd_ft:
        tau[k] = sd_ft[k] - sd_base[k]
    return tau


def apply_task_vector(model_base, tau, alpha=1.0):
    """Return a new model state dict: theta_base + alpha * tau."""
    sd_base = get_model_state_dicts(model_base)
    new_sd = {}
    for k in sd_base:
        new_sd[k] = sd_base[k].float() + alpha * tau[k].float()
    return new_sd


# ---------------------------------------------------------------------------
# Wreath Product Alignment
# ---------------------------------------------------------------------------
def svd_of_weight(W, d_k):
    """
    W is (h * d_k, d_m) for Q/K/V projections.
    Returns list of per-head (U_h, S_h, Vt_h).
    """
    h = W.shape[0] // d_k
    singular_values = []
    for i in range(h):
        w_h = W[i * d_k : (i + 1) * d_k].cpu().numpy()
        _, S, _ = np.linalg.svd(w_h, full_matrices=False)
        singular_values.append(S)
    return singular_values


def head_spectral_distance(svals1, svals2):
    """Distance between two heads based on their singular value spectra."""
    s1 = np.array(svals1)
    s2 = np.array(svals2)
    return np.linalg.norm(s1 - s2)


def build_head_distance_matrix(model_a_sds, model_b_sds, h, d_k):
    """
    For each MHA layer, build distance matrix D_{ij} = ||Sigma_i^A - Sigma_j^B||_F.
    Returns list of (h x h) distance matrices, one per layer.
    """
    dist_mats = []
    for l in range(L):
        D = np.zeros((h, h))
        svals_a = model_a_sds[l]
        svals_b = model_b_sds[l]
        for i in range(h):
            for j in range(h):
                D[i, j] = head_spectral_distance(svals_a[i], svals_b[j])
        dist_mats.append(D)
    return dist_mats


def extract_mha_weights(sd):
    """Extract MHA Q/K/V projection weights from state dict, per layer."""
    weights = []
    for l in range(L):
        prefix_q = f"blocks.{l}.mha.Wq.weight"
        prefix_k = f"blocks.{l}.mha.Wk.weight"
        prefix_v = f"blocks.{l}.mha.Wv.weight"
        w = {}
        w["q"] = sd[prefix_q]
        w["k"] = sd[prefix_k]
        w["v"] = sd[prefix_v]
        weights.append(w)
    return weights


def get_per_head_singular_values(mha_weights, h, d_k):
    """Extract singular values for each head from Q, K, V projections (averaged across Q/K/V)."""
    layer_svs = []
    for l in range(L):
        w = mha_weights[l]
        # Average singular values across Q, K, V for the distance metric
        sv_q = svd_of_weight(w["q"], d_k)
        sv_k = svd_of_weight(w["k"], d_k)
        sv_v = svd_of_weight(w["v"], d_k)
        # Combine them: average the singular value spectra
        combined = []
        for i in range(h):
            avg_s = (np.array(sv_q[i]) + np.array(sv_k[i]) + np.array(sv_v[i])) / 3.0
            combined.append(avg_s)
        layer_svs.append(combined)
    return layer_svs


def intra_head_lap_matrix(W_a_head, W_b_head):
    """
    Given weight matrices for a matched head pair (d_k x d_m),
    compute the LAP cost matrix for aligning input dimensions.
    Uses ||U_a^T P U_b|| distance or direct weight comparison.
    """
    wa = W_a_head.cpu().numpy()  # (d_k, d_m)
    wb = W_b_head.cpu().numpy()  # (d_k, d_m)
    d = wa.shape[0]
    C = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            C[i, j] = np.abs(wa[i, :] @ wb[j, :])  # maximize correlation -> minimize negative
    return -C  # Hungarian minimizes, we convert correlation to cost


def compute_wreath_permutation(sd_a, sd_b, h, d_k, d_m, n_iter=5):
    """
    Compute wreath product permutation between two model state dicts.
    Returns:
        P_in: dict layer -> (L, d_m, d_m) input permutation matrix
        P_head: dict layer -> (L, h, h) head permutation matrix
        P_out: dict layer -> (L, d_m, d_m) output permutation matrix
    """
    # Extract MHA weights
    w_a = extract_mha_weights(sd_a)
    w_b = extract_mha_weights(sd_b)
    svs_a = get_per_head_singular_values(w_a, h, d_k)
    svs_b = get_per_head_singular_values(w_b, h, d_k)

    # Step 1: Head matching per layer (Hungarian on D_{ij})
    head_perm = {}  # layer -> (h, h) permutation matrix
    for l in range(L):
        D = np.zeros((h, h))
        for i in range(h):
            for j in range(h):
                D[i, j] = head_spectral_distance(svs_a[l][i], svs_b[l][j])
        row_ind, col_ind = linear_sum_assignment(D)
        P = np.zeros((h, h))
        P[row_ind, col_ind] = 1.0
        head_perm[l] = torch.tensor(P, dtype=torch.float32)

    # Step 2: Intra-head LAP for each matched head pair
    # For each layer, for each matched head pair, solve LAP on d_k x d_k
    intra_perm = {}  # (layer, head_src, head_dst) -> (d_k, d_k) permutation
    for l in range(L):
        q_a = w_a[l]["q"]  # (h*d_k, d_m)
        q_b = w_b[l]["q"]
        for i in range(h):
            for j in range(h):
                if head_perm[l][i, j] > 0.5:
                    wa_h = q_a[i * d_k : (i + 1) * d_k, :]
                    wb_h = q_b[j * d_k : (j + 1) * d_k, :]
                    C = intra_head_lap_matrix(wa_h, wb_h)
                    ri, ci = linear_sum_assignment(C)
                    P = np.zeros((d_k, d_k))
                    P[ri, ci] = 1.0
                    intra_perm[(l, i, j)] = torch.tensor(P, dtype=torch.float32)

    # Step 3: Construct full permutation matrices
    # For each layer, the full permutation on d_m acts as:
    # Each input dimension is assigned coarse (head) + fine (intra-head)
    P_in = {}   # layer -> (d_m, d_m)
    P_out = {}  # layer -> (d_m, d_m) (transpose for output)

    for l in range(L):
        P_head_l = head_perm[l]  # (h, h)
        # Build full permutation matrix of size (h*d_k, h*d_k)
        # This is the wreath product: P_head \wr P_intra
        # P_full[i*d_k + a, j*d_k + b] = P_head[i,j] * P_intra[l,i,j][a,b]
        full_size = h * d_k
        P_full = torch.zeros((full_size, full_size), dtype=torch.float32)
        for i in range(h):
            for j in range(h):
                if P_head_l[i, j] > 0.5:
                    P_intra = intra_perm[(l, i, j)]
                    for a in range(d_k):
                        for b in range(d_k):
                            if P_intra[a, b] > 0.5:
                                P_full[i * d_k + a, j * d_k + b] = 1.0

        # The MHA dimension (h*d_k) is a subspace of d_m
        # For the full d_m, we pad with identity on the remaining dimensions
        # Actually, the projection weights are (h*d_k, d_m), so the input space is d_m
        # and the per-head output is d_k. The wreath product acts on the head dimension
        # space. Here, P_in should act on d_m (input embedding) and P_out on d_m (output).
        #
        # Actually, the re-basin literature permutes neurons, not head-output dims.
        # For simplicity and correctness with the architecture, we construct
        # P_in as acting on the full d_m input dimension space via the Wo projection.
        # The head permutation acts on the (h*d_k) intermediate space.

        # For the input permutation (d_m -> d_m): This comes from aligning Q/K/V weight columns
        # We approximate by solving the orthogonal Procrustes problem on aligned weights.
        # Using the Q weight matrix: W_q shape (h*d_k, d_m). After head matching + intra-head,
        # we have a full permutation on the row space. The column permutation (input space d_m)
        # is computed by solving min ||W_q_A * P - W_q_B_{permuted}||

        # Simplified: For a proper wreath product alignment, the permutation P of size d_m
        # is solved via orthogonal Procrustes on the aligned weight matrices.

        # We use an alternating approach: given current P_in from previous layer's output,
        # we align the current layer's weights.

        # For now, use identity as initial guess — will be refined in alternating iterations.
        P_in[l] = torch.eye(d_m, dtype=torch.float32)
        P_out[l] = torch.eye(d_m, dtype=torch.float32)

    # Step 4: Alternating optimization
    for iteration in range(n_iter):
        # Forward pass through layers: update P_in[l] and P_out[l]
        P_prop = torch.eye(d_m, dtype=torch.float32)  # propagated permutation from input
        for l in range(L):
            # Compute P_in[l] by solving: min ||W_q(l)_B * P - P_prop @ W_q(l)_A||_F^2
            # This is an orthogonal Procrustes problem; solution via SVD
            wq_a = sd_a[f"blocks.{l}.mha.Wq.weight"]  # (h*d_k, d_m)
            wq_b = sd_b[f"blocks.{l}.mha.Wq.weight"]  # (h*d_k, d_m)

            # Apply head + intra-head permutation to wq_a rows
            wq_a_perm = P_full @ wq_a  # permute rows (head space)

            # Now solve P_in: min ||wq_b @ P_in - P_prop @ wq_a_perm||
            # P_in = argmin_{orthogonal} ||wq_b @ P - target||
            target = P_prop @ wq_a_perm  # (h*d_k, d_m)
            M = wq_b.T @ target  # (d_m, d_m)
            U, _, Vt = np.linalg.svd(M.cpu().numpy(), full_matrices=True)
            P_in_opt = torch.tensor(U @ Vt, dtype=torch.float32)
            P_in[l] = P_in_opt

            # Update propagated permutation through this layer
            # P_out is the output permutation of the MHA
            wq_b_perm = P_full @ wq_b  # permute rows of B
            target_out = wq_a_perm @ P_in[l]  # what we get from permuted A
            M_out = wq_b_perm.T @ target_out  # (d_m, d_m)
            U_out, _, Vt_out = np.linalg.svd(M_out.cpu().numpy(), full_matrices=True)
            P_out[l] = torch.tensor(U_out @ Vt_out, dtype=torch.float32)
            P_prop = P_out[l]

    return P_in, P_out, head_perm, P_full


def permute_state_dict(sd, P_in, P_out, head_perm, P_full):
    """
    Apply wreath product permutation to a state dict.
    Transform theta_B's representation to align with theta_A's.
    """
    permuted = {}
    for k, v in sd.items():
        vp = v.float()

        # Patch embedding: apply P_in[0] to output channels
        if k == "patch_embed.weight":
            # Conv2d weight: (out_ch, in_ch, kh, kw), permute out_ch
            vp = vp.permute(0, 2, 3, 1)  # (d_m, kh, kw, 3)
            vp = torch.matmul(P_in[0], vp.reshape(vp.shape[0], -1)).reshape(vp.shape)
            vp = vp.permute(0, 3, 1, 2)  # back to (d_m, 3, kh, kw)
        elif k == "cls_token":
            vp = torch.matmul(P_in[0], vp.squeeze(0).T).T.unsqueeze(0)
        elif k == "pos_embed":
            # pos_embed: (1, N+1, d_m), permute last dim
            vp = torch.matmul(P_in[0], vp.squeeze(0).T).T.unsqueeze(0)

        # MHA layers
        for l in range(L):
            prefix_q = f"blocks.{l}.mha.Wq.weight"
            prefix_k = f"blocks.{l}.mha.Wk.weight"
            prefix_v = f"blocks.{l}.mha.Wv.weight"
            prefix_o = f"blocks.{l}.mha.Wo.weight"

            if k in [prefix_q, prefix_k, prefix_v]:
                # (h*d_k, d_m): apply P_in[l] to columns, P_full to rows
                vp = P_full @ vp @ P_in[l]
            elif k == prefix_o:
                # (d_m, h*d_k): apply P_full.T to columns, P_out[l] to rows
                vp = P_out[l] @ vp @ P_full.T

        # Layer norms: permute the weight dimension
        for l in range(L):
            ln1_w = f"blocks.{l}.ln1.weight"
            ln1_b = f"blocks.{l}.ln1.bias"
            ln2_w = f"blocks.{l}.ln2.weight"
            ln2_b = f"blocks.{l}.ln2.bias"
            if k == ln1_w or k == ln2_w or k == ln1_b or k == ln2_b:
                vp = torch.matmul(P_in[l], vp)

        # MLP first linear: (4*d_m, d_m), apply P_in[l] to columns
        for l in range(L):
            mlp0_w = f"blocks.{l}.mlp.0.weight"
            mlp0_b = f"blocks.{l}.mlp.0.bias"
            if k == mlp0_w:
                vp = vp @ P_in[l]
            elif k == mlp0_b:
                pass  # bias unaffected
            mlp2_w = f"blocks.{l}.mlp.2.weight"
            mlp2_b = f"blocks.{l}.mlp.2.bias"
            if k == mlp2_w:
                # (d_m, 4*d_m), apply P_out[l] to rows
                vp = P_out[l] @ vp
            elif k == mlp2_b:
                vp = torch.matmul(P_out[l], vp)

        # For mlp.0.bias, we need to permute too since input is permuted
        for l in range(L):
            mlp0_b = f"blocks.{l}.mlp.0.bias"
            if k == mlp0_b:
                # Bias of first MLP linear is in the 4*d_m space — unaffected by P_in
                pass

        # Final layer norm
        if k == "ln_final.weight" or k == "ln_final.bias":
            vp = torch.matmul(P_out[L-1], vp)

        # Classification head
        if k == "head.weight":
            vp = vp @ P_out[L-1]
        # head.bias unchanged

        permuted[k] = vp
    return permuted


def permute_task_vector_randomly(tau, seed=999):
    """Apply random orthogonal permutations to task vector components."""
    rng = np.random.RandomState(seed)
    permuted = {}
    for k, v in tau.items():
        vp = v.float()
        if len(vp.shape) >= 1 and vp.shape[0] >= 2:
            d = vp.shape[0]
            if d == D_M or d == 4 * D_M or d == H * D_K:
                # Generate random orthogonal matrix
                M = rng.randn(d, d)
                Q, _ = np.linalg.qr(M)
                P = torch.tensor(Q, dtype=torch.float32)
                if len(vp.shape) == 2:
                    vp = P @ vp
                elif len(vp.shape) == 1:
                    vp = torch.matmul(P, vp)
        permuted[k] = vp
    return permuted


# ---------------------------------------------------------------------------
# Full dataset 5-class accuracy proxy (for Pareto plot)
# ---------------------------------------------------------------------------
def get_cifar10_5class(train=True):
    """Load CIFAR-10 with 5 random classes (0-4)."""
    classes = [0, 1, 2, 3, 4]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    ds = torchvision.datasets.CIFAR10(
        root="./data", train=train, download=True, transform=transform
    )
    targets = np.array(ds.targets) if isinstance(ds.targets, list) else ds.targets
    mask = np.isin(targets, classes)
    indices = np.where(mask)[0]
    ds.data = ds.data[indices]
    ds.targets = [int(targets[i]) for i in indices]
    return ds


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    print(f"Using device: {DEVICE}", flush=True)
    print("=" * 70, flush=True)
    print("Experiment 3: Re-Basin Pipeline with Wreath Product Alignment", flush=True)
    print("=" * 70, flush=True)

    # ---- Load data ----
    print("\n[1/6] Loading CIFAR-10 binary dataset (classes 0, 1)...")
    train_ds = get_binary_cifar10(train=True, seed=SEED_A)
    test_ds = get_binary_cifar10(train=False)

    train_loader_A = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Create a second data loader with different shuffling for model B
    train_ds_B = get_binary_cifar10(train=True, seed=SEED_B)
    train_loader_B = torch.utils.data.DataLoader(
        train_ds_B, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    # ---- Step 1: Train base models ----
    print("\n[2/6] Training base models theta_A and theta_B...")

    torch.manual_seed(SEED_A)
    np.random.seed(SEED_A)
    model_a = TinyViT().to(DEVICE)
    print(f"  theta_A: {sum(p.numel() for p in model_a.parameters()):,} parameters")
    train_model(model_a, train_loader_A, test_loader, EPOCHS_BASE, LR, desc="theta_A")
    sd_a_base = get_model_state_dicts(model_a)

    torch.manual_seed(SEED_B)
    np.random.seed(SEED_B)
    model_b = TinyViT().to(DEVICE)
    train_model(model_b, train_loader_B, test_loader, EPOCHS_BASE, LR, desc="theta_B")
    sd_b_base = get_model_state_dicts(model_b)

    # ---- Step 2: Fine-tune theta_A on 20% data ----
    print("\n[3/6] Fine-tuning theta_A on 20% data...")
    ft_loader = subset_dataloader(
        train_ds, FT_FRAC, BATCH_SIZE, shuffle=True, seed=99
    )

    model_a_ft = TinyViT().to(DEVICE)
    load_state_into(model_a_ft, sd_a_base)
    train_model(model_a_ft, ft_loader, test_loader, EPOCHS_FT, LR, desc="theta_A_ft")
    sd_a_ft = get_model_state_dicts(model_a_ft)

    # Compute task vector
    tau = compute_task_vector(model_a_ft, model_a)
    print(f"  Task vector norm: {sum(v.norm().item() for v in tau.values()):.4f}")

    # ---- Step 3: Wreath Product Alignment ----
    print("\n[4/6] Computing wreath product alignment...")
    P_in, P_out, head_perm, P_full = compute_wreath_permutation(
        sd_a_base, sd_b_base, H, D_K, D_M, n_iter=N_ITER
    )
    print("  Alignment complete.")

    # Permute task vector using wreath alignment
    tau_wreath = permute_state_dict(tau, P_in, P_out, head_perm, P_full)

    # Random permutation baseline
    tau_random = permute_task_vector_randomly(tau, seed=777)

    # ---- Step 4: Transport and evaluate ----
    print("\n[5/6] Transporting task vector and evaluating...")

    results = {
        "Wreath Alignment": [],
        "Vanilla (no perm)": [],
        "Random Perm": [],
    }

    for alpha in tqdm(ALPHAS, desc="Alpha sweep"):
        # Create base model instances
        model_wreath = TinyViT().to(DEVICE)
        model_vanilla = TinyViT().to(DEVICE)
        model_random = TinyViT().to(DEVICE)

        # Wreath alignment transport
        sd_wreath = apply_task_vector(model_b, tau_wreath, alpha)
        load_state_into(model_wreath, sd_wreath)
        acc_w = evaluate(model_wreath, test_loader)

        # Vanilla (no permutation)
        sd_vanilla = apply_task_vector(model_b, tau, alpha)
        load_state_into(model_vanilla, sd_vanilla)
        acc_v = evaluate(model_vanilla, test_loader)

        # Random permutation
        sd_random = apply_task_vector(model_b, tau_random, alpha)
        load_state_into(model_random, sd_random)
        acc_r = evaluate(model_random, test_loader)

        results["Wreath Alignment"].append(acc_w)
        results["Vanilla (no perm)"].append(acc_v)
        results["Random Perm"].append(acc_r)

        del model_wreath, model_vanilla, model_random

    # ---- Print summary table ----
    print("\n" + "=" * 80)
    print(f"{'Alpha':<8} {'Wreath Alignment':<20} {'Vanilla (no perm)':<20} {'Random Perm':<20}")
    print("-" * 80)
    for i, alpha in enumerate(ALPHAS):
        print(f"{alpha:<8.1f} {results['Wreath Alignment'][i]:<20.2f} "
              f"{results['Vanilla (no perm)'][i]:<20.2f} "
              f"{results['Random Perm'][i]:<20.2f}")
    print("-" * 80)

    # Best accuracy for each method
    for method in results:
        idx = np.argmax(results[method])
        print(f"  Best {method}: {results[method][idx]:.2f}% at alpha={ALPHAS[idx]}")

    # ---- Step 6: 5-class zero-shot proxy for Pareto plot ----
    print("\n[6/6] Evaluating 5-class proxy accuracy for Pareto analysis...")
    ds_5class_test = get_cifar10_5class(train=False)
    test_loader_5class = torch.utils.data.DataLoader(
        ds_5class_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # For 5-class evaluation, we need a model with 5 output classes.
    # We use a zero-shot proxy: the model's feature extractor (all except head)
    # paired with a nearest-centroid classifier on the 5-class test set.
    # This measures how well the feature space preserves broader CIFAR-10 structure.

    @torch.no_grad()
    def extract_features(model, loader):
        model.eval()
        features = []
        all_labels = []
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            # Forward through everything except head
            B = imgs.shape[0]
            x = model.patch_embed(imgs).flatten(2).transpose(1, 2)
            cls = model.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = x + model.pos_embed
            for blk in model.blocks:
                x = blk(x)
            x = model.ln_final(x[:, 0])
            features.append(x.cpu())
            all_labels.append(labels)
        return torch.cat(features), torch.cat(all_labels)

    def nearest_centroid_accuracy(features, labels, n_classes=5):
        """Simple nearest-centroid classifier accuracy."""
        centroids = []
        for c in range(n_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                centroids.append(features[mask].mean(0))
            else:
                centroids.append(torch.zeros(features.shape[1]))
        centroids = torch.stack(centroids)
        dists = torch.cdist(features, centroids)
        preds = dists.argmin(1)
        return (preds == labels).float().mean().item() * 100.0

    # Evaluate proxy accuracy for each method's best-alpha model
    # Use models at best alpha for task accuracy
    proxy_results = {}

    for method, key in [("Wreath Alignment", "wreath"),
                         ("Vanilla (no perm)", "vanilla"),
                         ("Random Perm", "random")]:
        best_alpha_idx = np.argmax(results[method])
        best_alpha = ALPHAS[best_alpha_idx]

        model_proxy = TinyViT().to(DEVICE)
        if key == "wreath":
            sd_p = apply_task_vector(model_b, tau_wreath, best_alpha)
        elif key == "vanilla":
            sd_p = apply_task_vector(model_b, tau, best_alpha)
        else:
            sd_p = apply_task_vector(model_b, tau_random, best_alpha)
        load_state_into(model_proxy, sd_p)

        feats, labels = extract_features(model_proxy, test_loader_5class)
        # Map labels 0..4 -> 0..4 (they already are 0..4)
        proxy_acc = nearest_centroid_accuracy(feats, labels)
        proxy_results[method] = proxy_acc
        print(f"  {method}: 5-class proxy accuracy = {proxy_acc:.2f}%")

        del model_proxy

    # Also evaluate base model B
    model_b_proxy = TinyViT().to(DEVICE)
    load_state_into(model_b_proxy, sd_b_base)
    feats_b, labels_b = extract_features(model_b_proxy, test_loader_5class)
    base_proxy_acc = nearest_centroid_accuracy(feats_b, labels_b)
    print(f"  Base theta_B: 5-class proxy accuracy = {base_proxy_acc:.2f}%")
    del model_b_proxy

    # ---- Figure 1: Task Accuracy vs Alpha ----
    print("\n  Generating Figure 1...")
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    colors = {"Wreath Alignment": "#2196F3", "Vanilla (no perm)": "#757575", "Random Perm": "#F44336"}
    markers = {"Wreath Alignment": "o", "Vanilla (no perm)": "s", "Random Perm": "^"}

    for method in ["Wreath Alignment", "Vanilla (no perm)", "Random Perm"]:
        ax1.plot(ALPHAS, results[method], marker=markers[method], color=colors[method],
                 linewidth=2, markersize=8, label=method)

    ax1.set_xlabel("Alpha", fontsize=13)
    ax1.set_ylabel("Binary Classification Accuracy (%)", fontsize=13)
    ax1.set_title("Task Vector Transportation Accuracy", fontsize=14)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=40, top=105)
    fig1.tight_layout()
    fig1_path = os.path.join(FIG_DIR, "exp3_task_accuracy.pdf")
    fig1.savefig(fig1_path, dpi=150)
    plt.close(fig1)
    print(f"  Saved: {fig1_path} ({os.path.getsize(fig1_path):,} bytes)")

    # ---- Figure 2: Pareto Plot ----
    print("  Generating Figure 2...")
    fig2, ax2 = plt.subplots(figsize=(7, 6))

    # X: proxy accuracy (higher = better), Y: task accuracy (higher = better)
    pareto_points = {}
    for method in results:
        best_idx = np.argmax(results[method])
        pareto_points[method] = (proxy_results[method], results[method][best_idx])

    for method, (px, py) in pareto_points.items():
        ax2.scatter(px, py, color=colors[method], s=150, marker=markers[method],
                    edgecolors="black", linewidths=1, zorder=5, label=method)
        ax2.annotate(method.split(" ")[0], (px, py),
                     textcoords="offset points", xytext=(10, 5), fontsize=10)

    # Add base model B as reference
    base_task_acc = evaluate(model_b, test_loader)
    ax2.scatter(base_proxy_acc, base_task_acc, color="#4CAF50", s=150, marker="D",
                edgecolors="black", linewidths=1, zorder=4, label="Base theta_B")

    ax2.set_xlabel("5-Class Zero-Shot Proxy Accuracy (%)", fontsize=12)
    ax2.set_ylabel("Binary Task Accuracy (%)", fontsize=12)
    ax2.set_title("Pareto: Task Performance vs. General Representation", fontsize=13)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2_path = os.path.join(FIG_DIR, "exp3_pareto.pdf")
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)
    print(f"  Saved: {fig2_path} ({os.path.getsize(fig2_path):,} bytes)")

    print("\n" + "=" * 70)
    print("Experiment 3 complete!")
    print(f"Figures saved to: {FIG_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
