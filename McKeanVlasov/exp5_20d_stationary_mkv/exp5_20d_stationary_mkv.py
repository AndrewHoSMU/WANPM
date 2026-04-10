"""
McKean-Vlasov WANPM  —  Experiment 5: 20D Stationary
======================================================
Problem
    V(x) = (theta/2) ||x||^2,   W(z) = (1/2) ||z||^2,   x in R^20
    b(x, rho) = -lambda*x  (m* = 0 by symmetry)

Exact:  rho* = N(0, sigma^2/(2*lambda) * I_20)

Metric: per-dimension mean absolute error and variance error.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import time as pytime

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Problem parameters
# ============================================================================
DIM      = 20
THETA    = 1.0
SIGMA    = 1.0
LAM      = THETA + 1.0
VAR_STAR = SIGMA**2 / (2.0 * LAM)
STD_STAR = np.sqrt(VAR_STAR)

# ============================================================================
# Hyperparameters
# ============================================================================
K        = 5000
M        = 5000
M_W      = 2 * M
D_BASE   = 30
N_EPOCHS = 10000
LR_GEN   = 1e-3
LR_TEST  = 1e-2
ADV_FREQ = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
os.makedirs('results', exist_ok=True)

# ============================================================================
# Networks
# ============================================================================

class PushforwardNet(nn.Module):
    def __init__(self, d_in, d_out, hidden=(64, 64)):
        super().__init__()
        layers, in_dim = [], d_in
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, r):
        return self.net(r)


class TestFunctions(nn.Module):
    def __init__(self, n_spatial, K):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_spatial, K) * 0.3)
        self.b = nn.Parameter(torch.rand(K) * 2 * np.pi)

    def _arg(self, x):
        return x @ self.w + self.b

    def grad_dot(self, x, v):
        return (v @ self.w) * torch.cos(self._arg(x))

    def laplacian(self, x):
        return -(self.w ** 2).sum(0) * torch.sin(self._arg(x))


# ============================================================================
# Loss
# ============================================================================

def compute_loss(pf, tf):
    xi  = pf(torch.rand(M, D_BASE, device=device))
    EV  = tf.grad_dot(xi, THETA * xi).mean(0)
    ED  = (SIGMA**2 / 2.0) * tf.laplacian(xi).mean(0)

    xi2 = pf(torch.rand(M_W, D_BASE, device=device))
    eta = pf(torch.rand(M_W, D_BASE, device=device))
    EW  = tf.grad_dot(xi2, xi2 - eta).mean(0)

    R = -EV - EW + ED
    return (R ** 2).mean()


# ============================================================================
# Training
# ============================================================================
pf  = PushforwardNet(D_BASE, DIM).to(device)
tf  = TestFunctions(DIM, K).to(device)

gen_opt  = optim.Adam(pf.parameters(), lr=LR_GEN)
test_opt = optim.SGD(tf.parameters(),  lr=LR_TEST)

loss_log = []
t0 = pytime.time()

print(f"\n20D Stationary MKV  |  lambda={LAM}  |  exact std={STD_STAR:.4f}")
print(f"K={K}, M={M}, M_W={M_W}, D_BASE={D_BASE}, "
      f"N_EPOCHS={N_EPOCHS}, ADV_FREQ={ADV_FREQ}")
print(f"PF params: {sum(p.numel() for p in pf.parameters()):,}")

for epoch in range(N_EPOCHS):
    if epoch > 0 and epoch % ADV_FREQ == 0:
        loss_adv = compute_loss(pf, tf)
        test_opt.zero_grad(); (-loss_adv).backward(); test_opt.step()

    loss = compute_loss(pf, tf)
    gen_opt.zero_grad(); loss.backward(); gen_opt.step()
    loss_log.append(loss.item())

    if epoch % 2000 == 0 or epoch == N_EPOCHS - 1:
        with torch.no_grad():
            xs = pf(torch.rand(5000, D_BASE, device=device)).cpu().numpy()
        me = np.abs(xs.mean(0)).mean()
        ve = np.abs(xs.var(0) - VAR_STAR).mean()
        print(f"  epoch {epoch:5d}  loss={loss.item():.4e}  "
              f"|mean_err|={me:.5f}  |var_err|={ve:.5f}  "
              f"t={pytime.time()-t0:.1f}s")

# ============================================================================
# Final evaluation
# ============================================================================
with torch.no_grad():
    xs = pf(torch.rand(50000, D_BASE, device=device)).cpu().numpy()

mean_err = np.abs(xs.mean(0))
var_err  = np.abs(xs.var(0) - VAR_STAR)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.bar(range(DIM), mean_err, color='royalblue', alpha=0.8)
ax.axhline(mean_err.mean(), color='r', ls='--', lw=1.5,
           label=f'avg={mean_err.mean():.4f}')
ax.set_xlabel('Dimension'); ax.set_ylabel('|mean error|')
ax.set_title('Per-dim mean error'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.bar(range(DIM), var_err, color='darkorange', alpha=0.8)
ax.axhline(var_err.mean(), color='r', ls='--', lw=1.5,
           label=f'avg={var_err.mean():.4f}')
ax.set_xlabel('Dimension'); ax.set_ylabel('|var error|')
ax.set_title('Per-dim variance error'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.semilogy(loss_log, color='steelblue', lw=1.2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training loss'); ax.grid(True, which='both', alpha=0.3)

plt.suptitle('20D Stationary McKean-Vlasov', fontsize=13)
plt.tight_layout()
plt.savefig('results/exp5_20d_stationary.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 20 marginal histograms — 4 rows × 5 columns
# ============================================================================
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
x_range_base = np.linspace(-4 * STD_STAR, 4 * STD_STAR, 500)
pdf_true = norm.pdf(x_range_base, 0.0, STD_STAR)

for d in range(DIM):
    ax = axes[d // 5, d % 5]
    ax.hist(xs[:, d], bins=60, density=True, alpha=0.7,
            color='skyblue', edgecolor='black', linewidth=0.4,
            label='WANPM')
    x_r = np.linspace(xs[:, d].min() - 0.2, xs[:, d].max() + 0.2, 500)
    ax.plot(x_r, norm.pdf(x_r, 0.0, STD_STAR), 'r-', lw=2,
            label=f'True N(0,{VAR_STAR:.3f})')
    ax.set_title(f'Dim {d+1}', fontsize=10)
    ax.set_xlabel(f'$x_{{{d+1}}}$', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

plt.suptitle('20D Stationary McKean-Vlasov — Marginal Distributions', fontsize=14)
plt.tight_layout()
plt.savefig('results/exp5_20d_marginals.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# Save model
# ============================================================================
torch.save({
    'pf_state_dict':  pf.state_dict(),
    'tf_state_dict':  tf.state_dict(),
    'loss_log':       loss_log,
    'N_EPOCHS':       N_EPOCHS,
}, 'results/exp5_model.pt')
print("Model saved to results/exp5_model.pt")

print(f"\nMean |mean err| = {mean_err.mean():.5f}")
print(f"Mean |var err|  = {var_err.mean():.5f}")
print(f"Final loss={loss_log[-1]:.4e}  time={pytime.time()-t0:.1f}s")
