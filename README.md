# Math in Computer (Part 1) — Course Assignments

School of Computer Science & Technology, Zhejiang University · CSMATH 2026 · Spring/Summer 2026

This repository contains five programming assignments and a course paper for the *Math in Computer (Part 1)* course, covering dimensionality reduction, optimization, curve fitting, clustering, and their applications in computer science.

---

## Directory Structure

```
.
├── PCA/                # Assignment 1: PCA & Dimensionality Reduction
├── LMA/                # Assignment 2: Levenberg-Marquardt Algorithm
├── curveFitting/       # Assignment 3: Curve Fitting (Polynomial & Bezier)
├── Clustering/         # Assignment 4: Clustering Algorithms Compared
├── CoursePaper/        # Course Paper (CVPR 2026 LaTeX Template)
└── README.md
```

---

## Assignments

### 1. PCA — Dimensionality Reduction & Visualization

**Goal**: Reduce and visualize the UCI handwritten digits dataset.

| File | Description |
|------|-------------|
| `main.py` | **Basic task**: Filter digit "3", project to 2D via PCA, plot scatter + sample image grid |
| `allDigits.py` | **Comparison**: Apply **PCA / t-SNE / UMAP** to all digits (0–9) side by side |
| `bonus.py` | **Advanced task**: Four methods on digit "3" (PCA, t-SNE, UMAP, **VAE latent space**). VAE is built and trained from scratch with PyTorch |

**Key techniques**:
- PCA via `sklearn.decomposition.PCA`
- Non-linear dimensionality reduction: t-SNE vs. UMAP
- Variational Autoencoder (VAE): encoder-decoder architecture, reparameterization trick, ELBO loss

**Dependencies**: `numpy, matplotlib, scikit-learn, umap-learn, torch`

---

### 2. LMA — Levenberg-Marquardt Optimization

**Goal**: Implement the Levenberg-Marquardt algorithm from scratch and visualize convergence on five classic test functions.

**Key techniques**:
- Automatic Jacobian computation via `torch.func.jacrev`
- Adaptive damping parameter λ: decrease λ on loss improvement (→ Gauss-Newton), increase λ on loss increase (→ gradient descent)
- Iteration trajectories plotted on 2D loss landscapes

**Test functions**:

| Function | Characteristics |
|----------|----------------|
| Rosenbrock | Narrow, curved valley — classic hard optimization case |
| Booth | Smooth quadratic |
| Beale | Multiple extrema, non-convex |
| Himmelblau | Four local minima |
| Circle Optimization | Custom geometric optimization example |

**Run**: `python main.py` generates a 2×3 grid of convergence paths.

**Dependencies**: `torch, numpy, matplotlib`

---

### 3. curveFitting — Curve Fitting

**Goal**: Compare two basis-function approaches to curve fitting.

| File | Description |
|------|-------------|
| `main.py` | **Polynomial fitting**: Vandermonde matrix + pseudoinverse / ridge regression. Compares different sample sizes N and polynomial degrees M |
| `Bezier.py` | **Bezier curve fitting**: Bernstein basis polynomials, showing control points and their relationship to the fitted curve |

**Key techniques**:
- Ill-conditioning of the Vandermonde matrix and the need for regularization
- Ridge regression closed-form solution: $\mathbf{w} = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$
- Bernstein basis functions and the convex hull property of Bezier curves
- Overfitting in high-degree polynomials vs. regularization

**Experiments** (`main.py`):
- `(N=10, M=3)` — low-degree underfitting
- `(N=10, M=9)` — high-degree overfitting
- `(N=15, M=9)` — moderate data, reduced overfitting
- `(N=100, M=9)` — large-sample high-degree fit
- `(N=10, M=9, ln λ=-18)` — regularized, oscillation suppressed

**Dependencies**: `numpy, matplotlib, scipy, torch` (Bezier)

---

### 4. Clustering — Algorithm Comparison

**Goal**: Implement three clustering algorithms from scratch and compare them on the same synthetic MoG dataset.

| Algorithm | Implementation | Core Idea |
|-----------|---------------|-----------|
| Mean-Shift | Custom `MeanShift` class | Mode-seeking via kernel density estimation; auto-determines cluster count |
| EM-MoG | Custom `em_mog` function | Expectation-Maximization for Gaussian mixtures; soft clustering + parameter estimation |
| Spectral Clustering | Custom `spectral_clustering` function | Graph Laplacian eigendecomposition + K-means; connectivity-based |

**Key techniques**:
- **Mean-Shift**: bandwidth parameter sensitivity, centroid merging
- **EM-MoG**: E-step (posterior), M-step (parameter update), K-means init, covariance regularization
- **Spectral Clustering**: RBF similarity matrix → normalized Laplacian $L_{sym}$ → eigenvector embedding → K-means

**Visualization**: Three-column comparison with distinct markers for cluster centers, estimated means, and ground truth means.

**Dependencies**: `numpy, matplotlib, scipy, scikit-learn`

---

### 5. CoursePaper — Course Paper

A course paper framework built on the official **CVPR 2026 LaTeX template**:

```
CoursePaper/
├── rq-2026.pdf                          # Assignment requirements document
└── author-kit-CVPR2026-v1-latex-/       # CVPR 2026 LaTeX template
    ├── main.tex                          # Entry point (paper metadata, structure)
    ├── preamble.tex                      # Package imports
    ├── cvpr.sty                          # CVPR style file
    ├── main.bib                          # Bibliography (BibTeX)
    ├── ieeenat_fullname.bst              # Bibliography style
    ├── sec/
    │   ├── 0_abstract.tex                # Abstract
    │   ├── 1_intro.tex                   # Introduction (background, related work, contributions)
    │   ├── 2_formalization.tex           # Problem Formalization
    │   ├── 3_intuition.tex               # Mathematical Background & Intuition
    │   ├── 4_method.tex                  # Method Analysis
    │   ├── 5_experiments.tex             # Implementation & Experiments
    │   ├── 6_insights.tex                # Personal Insights
    │   ├── 7_acknowledgments.tex         # Acknowledgments
    │   └── X_suppl.tex                   # Supplementary Material
    └── .github/workflows/latex-build.yml # CI auto-build
```

**Core competency pipeline**: Formalization → Intuition → Mathematical Derivation → Computation

**Build**: `latexmk -pdf main.tex` or use the included GitHub Actions workflow.

---

## Browsing Guide

- **Quick start**: Follow the assignment order; `main.py` is the entry point for each assignment
- **Highlights**: VAE from scratch in `PCA/bonus.py`; three clustering algorithms implemented from scratch in `Clustering/main.py`
- **Course paper**: The template is self-contained — fill in the sections, compile, and it's ready to submit

## Environment

```bash
pip install numpy matplotlib scipy scikit-learn umap-learn torch
```

`umap-learn` is required for `PCA/allDigits.py` and `PCA/bonus.py`. `torch` is required for `LMA` and the VAE portion of `bonus.py`.
