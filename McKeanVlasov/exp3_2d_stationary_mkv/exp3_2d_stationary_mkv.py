"""
McKean-Vlasov WANPM  —  Experiment 3: 2D Stationary
=====================================================
Problem
    V(x) = (theta/2) ||x||^2,   W(z) = (1/2) ||z||^2,   x in R^2
    b(x, rho) = -lambda*x  (m* = 0 by symmetry of IC and V)

Exact:  rho* = N(0, sigma^2/(2*lambda) * I_2)

Loss (stationary):  -E_V - E_W + E_D = 0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm
import os
import time as pytime

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Problem parameters
# ============================================================================
DIM      = 2
THETA    = 1.0
SIGMA    = 1.0
LAM      = THETA + 1.0
VAR_STAR = SIGMA**2 / (2.0 * LAM)
STD_STAR = np.sqrt(VAR_STAR)

# ============================================================================
# Hyperparameters
# ============================================================================
K        = 2000
M        = 2000
M_W      = 2 * M
D_BASE   = 8
N_EPOCHS = 5000
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
    """F_vth : R^D_BASE -> R^DIM  (stationary)"""
    def __init__(self, d_in, d_out, hidden=(128, 128, 128)):
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
    """psi^(k)(x) = sin(w^(k).x + b^(k)),  w in R^n, b in R"""
    def __init__(self, n_spatial, K):
        super().__init__()
        self.w = nn.Parameter(torch.rand(n_spatial, K) * 2.0)
        self.b = nn.Parameter(torch.rand(K) * 2 * np.pi)

    def _arg(self, x):
        return x @ self.w + self.b                              # (B, K)

    def grad_dot(self, x, v):
        # v: (B, n),  grad_psi = w * cos(arg)
        return (v @ self.w) * torch.cos(self._arg(x))          # (B, K)

    def laplacian(self, x):
        # Delta_psi = -||w||^2 sin(arg)
        return -(self.w ** 2).sum(0) * torch.sin(self._arg(x)) # (B, K)


# ============================================================================
# Loss
# ============================================================================

def compute_loss(pf, tf):
    # E_V, E_D
    xi  = pf(torch.rand(M, D_BASE, device=device))            # (M, 2)
    EV  = tf.grad_dot(xi, THETA * xi).mean(0)                  # grad_V = theta*x
    ED  = (SIGMA**2 / 2.0) * tf.laplacian(xi).mean(0)

    # E_W  (M_W iid pairs)
    xi2 = pf(torch.rand(M_W, D_BASE, device=device))          # (M_W, 2)
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

print(f"\n2D Stationary MKV  |  lambda={LAM}  |  exact std={STD_STAR:.4f}")
print(f"K={K}, M={M}, M_W={M_W}, D_BASE={D_BASE}, "
      f"N_EPOCHS={N_EPOCHS}, ADV_FREQ={ADV_FREQ}")

for epoch in range(N_EPOCHS):
    if epoch > 0 and epoch % ADV_FREQ == 0:
        loss_adv = compute_loss(pf, tf)
        test_opt.zero_grad(); (-loss_adv).backward(); test_opt.step()

    loss = compute_loss(pf, tf)
    gen_opt.zero_grad(); loss.backward(); gen_opt.step()
    loss_log.append(loss.item())

    if epoch % 1000 == 0 or epoch == N_EPOCHS - 1:
        with torch.no_grad():
            xs = pf(torch.rand(10000, D_BASE, device=device)).cpu().numpy()
        print(f"  epoch {epoch:5d}  loss={loss.item():.4e}  "
              f"mean=({xs[:,0].mean():.4f},{xs[:,1].mean():.4f})  "
              f"std=({xs[:,0].std():.4f},{xs[:,1].std():.4f})  "
              f"(exact {STD_STAR:.4f})  t={pytime.time()-t0:.1f}s")

# ============================================================================
# Plots
# ============================================================================
with torch.no_grad():
    xs = pf(torch.rand(20000, D_BASE, device=device)).cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.scatter(xs[:3000, 0], xs[:3000, 1], s=2, alpha=0.3, color='royalblue')
theta_c = np.linspace(0, 2*np.pi, 300)
ax.plot(2*STD_STAR*np.cos(theta_c), 2*STD_STAR*np.sin(theta_c),
        'r--', lw=1.5, label='$2\\sigma$ exact')
ax.set_aspect('equal'); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('2D scatter'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
xg = np.linspace(-2, 2, 400)
ax.hist(xs[:, 0], bins=60, density=True, alpha=0.6, color='royalblue',
        edgecolor='none', label='WANPM ($x_1$)')
ax.plot(xg, scipy_norm.pdf(xg, 0, STD_STAR), 'r--', lw=2, label='Exact marginal')
ax.set_xlabel('$x_1$'); ax.set_ylabel('Density')
ax.set_title('Marginal $x_1$'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.semilogy(loss_log, color='steelblue', lw=1.2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training loss'); ax.grid(True, which='both', alpha=0.3)

plt.suptitle('2D Stationary McKean-Vlasov', fontsize=13)
plt.tight_layout()
plt.savefig('results/exp3_2d_stationary.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPer-dim std: {xs.std(axis=0).tolist()}  (exact {STD_STAR:.4f})")
print(f"Final loss={loss_log[-1]:.4e}  time={pytime.time()-t0:.1f}s")
