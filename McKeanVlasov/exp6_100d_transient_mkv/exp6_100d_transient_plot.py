"""
McKean-Vlasov WANPM  —  Experiment 6: 100D Transient  (plotting only)
======================================================================
Loads the saved model and regenerates plots:
  - Histograms for dim 1, dim 2, dim 100 at each time slot
  - Mean ± σ evolution for those dimensions
  - The heatmap (exp6_100d_transient_heatmap.png) is NOT regenerated here.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Problem parameters  (must match training)
# ============================================================================
DIM    = 100
THETA  = 1.0
SIGMA  = 1.0
LAM    = THETA + 1.0
T_END  = 1.0
EPS_T  = 1e-3

np.random.seed(42)
MU0_NP  = np.random.randn(DIM) * 2.0
SIG0    = 0.5
VAR_EQ  = SIGMA**2 / (2.0 * LAM)

K      = 5000
D_BASE = 2 * DIM

def exact_mean_np(t):
    return MU0_NP * np.exp(-THETA * t)

def exact_var(t):
    return (SIG0**2 - VAR_EQ) * np.exp(-2.0 * LAM * t) + VAR_EQ

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

os.makedirs('results', exist_ok=True)

MU0 = torch.tensor(MU0_NP, dtype=torch.float32, device=device)

# ============================================================================
# Network definition  (must match training)
# ============================================================================

class PushforwardNet(nn.Module):
    def __init__(self, d_base, d_out, hidden=(128, 128, 128)):
        super().__init__()
        layers, in_dim = [], 1 + d_base
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, t, x0, r):
        return x0 + torch.sqrt(t) * self.net(torch.cat([t, r], dim=1))


class TestFunctions(nn.Module):
    def __init__(self, n_spatial, K):
        super().__init__()
        self.w     = nn.Parameter(torch.rand(n_spatial, K) * 0.1)
        self.kappa = nn.Parameter(torch.rand(K) * 0.1)
        self.b     = nn.Parameter(torch.rand(K) * 2 * np.pi)


def sample_ic(n):
    return MU0 + SIG0 * torch.randn(n, DIM, device=device)


# ============================================================================
# Load model
# ============================================================================
checkpoint = torch.load('results/exp6_model.pt', map_location=device)

pf = PushforwardNet(D_BASE, DIM).to(device)
pf.load_state_dict(checkpoint['pf_state_dict'])
pf.eval()

loss_log = checkpoint['loss_log']
print(f"Loaded model — final loss={loss_log[-1]:.4e}  ({len(loss_log)} epochs)")

# ============================================================================
# Plots
# ============================================================================

# (1) Training loss — single plot, 3:1 width:height ratio
fig, ax = plt.subplots(figsize=(12, 4))
ax.semilogy(loss_log, color='steelblue', lw=1.2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training loss — 100D Transient McKean-Vlasov')
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('results/exp6_100d_loss.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/exp6_100d_loss.png")

# (2) Histograms: dim 1, dim 2, dim 100 at each time slot
DIMS_HIST  = [0, 1, DIM - 1]        # 0-indexed → dim 1, 2, 100
HIST_TIMES = [0.1, 0.5, 1.0]
n_rows, n_cols = len(DIMS_HIST), len(HIST_TIMES)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

with torch.no_grad():
    for j, tv in enumerate(HIST_TIMES):
        tb = tv * torch.ones(10000, 1, device=device)
        xs = pf(tb, sample_ic(10000),
                torch.rand(10000, D_BASE, device=device)).cpu().numpy()
        for i, d in enumerate(DIMS_HIST):
            ax = axes[i, j]
            ax.hist(xs[:, d], bins=50, density=True, alpha=0.7,
                    color='skyblue', edgecolor='black', linewidth=0.4,
                    label='WANPM')
            m_t = exact_mean_np(tv)[d]
            v_t = exact_var(tv)
            x_r = np.linspace(xs[:, d].min() - 0.5, xs[:, d].max() + 0.5, 500)
            ax.plot(x_r, norm.pdf(x_r, m_t, np.sqrt(v_t)), 'r-', lw=2,
                    label=f'True N({m_t:.2f},{v_t:.2f})')
            ax.set_xlabel(f'$x_{{{d+1}}}$')
            ax.set_title(f'Dim {d+1},  $t={tv}$')
            if j == 0:
                ax.set_ylabel('Density')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.suptitle('100D Transient McKean-Vlasov — Histograms', fontsize=13)
plt.tight_layout()
plt.savefig('results/exp6_100d_histograms.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/exp6_100d_histograms.png")

# (3) Mean ± σ evolution: dim 1, dim 2, dim 100
DIMS_PLOT = [0, 1, DIM - 1]
t_ev = np.linspace(EPS_T, T_END, 100)
m_lrn = {d: [] for d in DIMS_PLOT}
s_lrn = {d: [] for d in DIMS_PLOT}

print("Computing mean/std evolution over time...")
with torch.no_grad():
    for tv in t_ev:
        tb = tv * torch.ones(5000, 1, device=device)
        xs = pf(tb, sample_ic(5000),
                torch.rand(5000, D_BASE, device=device)).cpu().numpy()
        for d in DIMS_PLOT:
            m_lrn[d].append(xs[:, d].mean())
            s_lrn[d].append(xs[:, d].std())

m_true_all = np.array([exact_mean_np(tv) for tv in t_ev])   # (n_t, DIM)
s_true_all = np.array([np.sqrt(exact_var(tv)) for tv in t_ev])  # (n_t,) scalar

fig, axes = plt.subplots(1, len(DIMS_PLOT), figsize=(6 * len(DIMS_PLOT), 5))
for col, d in enumerate(DIMS_PLOT):
    ax = axes[col]
    ml = np.array(m_lrn[d]); sl = np.array(s_lrn[d])
    mt = m_true_all[:, d];   st = s_true_all
    ax.plot(t_ev, ml, color='royalblue', lw=2, label='WANPM mean', zorder=3)
    ax.fill_between(t_ev, ml - sl, ml + sl, alpha=0.3, color='royalblue',
                    label='WANPM ±σ', zorder=1)
    ax.plot(t_ev, mt, 'r--', lw=2, label='Exact mean', zorder=2)
    ax.fill_between(t_ev, mt - st, mt + st, alpha=0.2, color='red',
                    label='Exact ±σ', zorder=0)
    ax.set_xlabel('$t$'); ax.set_ylabel('Value')
    ax.set_title(f'Dim {d+1} — Mean ± σ')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('100D Transient McKean-Vlasov — Mean ± σ Evolution', fontsize=13)
plt.tight_layout()
plt.savefig('results/exp6_100d_transient_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/exp6_100d_transient_evolution.png")

print("\nAll plots done.")
