"""
McKean-Vlasov WANPM  —  Experiment 1: 1D Stationary
=====================================================
Problem
    V(x) = (theta/2) x^2,   W(z) = (1/2) z^2
    -d/dx[ b(x,rho) rho ] + (sigma^2/2) d^2rho/dx^2 = 0
    b(x,rho) = -theta*x - (x - m*)  = -lambda*x,  lambda = theta+1,  m*=0

Exact stationary solution:  rho* = N(0, sigma^2 / (2*lambda))

Weak form (stationary, eq. (stationary_compact) in notes):
    -E_V - E_W + E_D = 0

    E_V = E_xi[ grad_psi(xi) . grad_V(xi) ]
        = E_xi[ w cos(w xi + b) * theta*xi ]

    E_W = (1/M_W) sum_m grad_psi(xi^m) . (xi^m - eta^m)
        xi^m, eta^m  iid ~ rho_vth   (quadratic kernel: grad_W = xi - eta)

    E_D = (sigma^2/2) E_xi[ Delta_psi(xi) ]
        = (sigma^2/2) E_xi[ -w^2 sin(w xi + b) ]

Loss  =  mean_k ( -E_V^k - E_W^k + E_D^k )^2
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
DIM    = 1
THETA  = 1.0          # confinement strength
SIGMA  = 1.0          # noise
LAM    = THETA + 1.0  # effective rate  (theta from V  +  1 from W)
VAR_STAR = SIGMA**2 / (2.0 * LAM)
STD_STAR = np.sqrt(VAR_STAR)

# ============================================================================
# Hyperparameters
# ============================================================================
K          = 1000    # test functions
M          = 1000    # interior batch  (E_V, E_D)
M_W        = 2 * M   # interaction batch  (E_W): 2M pairs (xi, eta)
D_BASE     = 4       # base distribution dimension
N_EPOCHS   = 5000
LR_GEN     = 1e-3
LR_TEST    = 1e-2
ADV_FREQ   = 2       # adversary update every ADV_FREQ generator steps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
os.makedirs('results', exist_ok=True)

# ============================================================================
# Networks
# ============================================================================

class PushforwardNet(nn.Module):
    """F_vth : R^D_BASE -> R  (stationary, no t or x0 dependence)"""
    def __init__(self, d_in, d_out, hidden=(32, 32)):
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
    """psi^(k)(x) = sin(w^(k) x + b^(k)),  x in R"""
    def __init__(self, K):
        super().__init__()
        # w: (1, K),  b: (K,)
        self.w = nn.Parameter(torch.rand(1, K) * 2.0)
        self.b = nn.Parameter(torch.rand(K) * 2 * np.pi)

    def _arg(self, x):          # x: (B, 1)  -> (B, K)
        return x @ self.w + self.b

    def grad_dot(self, x, v):   # v: (B, 1) — the vector to dot with grad_psi
        # grad_psi = w * cos(arg),  dot with v -> (v @ w) * cos(arg)
        return (v @ self.w) * torch.cos(self._arg(x))   # (B, K)

    def laplacian(self, x):
        # Delta_psi = -w^2 * sin(arg),  summed over spatial dims (here 1)
        return -(self.w ** 2).sum(0) * torch.sin(self._arg(x))  # (B, K)


# ============================================================================
# Loss
# ============================================================================

def compute_loss(pf, tf):
    # ---- E_V and E_D  (M samples from rho_vth) ----
    r   = torch.rand(M, D_BASE, device=device)
    xi  = pf(r)                                           # (M, 1)
    EV  = tf.grad_dot(xi, THETA * xi).mean(0)             # grad_V = theta*x
    ED  = (SIGMA**2 / 2.0) * tf.laplacian(xi).mean(0)

    # ---- E_W  (M_W iid pairs (xi2, eta)) ----
    xi2  = pf(torch.rand(M_W, D_BASE, device=device))    # (M_W, 1)
    eta  = pf(torch.rand(M_W, D_BASE, device=device))    # (M_W, 1)
    EW   = tf.grad_dot(xi2, xi2 - eta).mean(0)            # grad_W(xi-eta) = xi-eta

    # Residual  -E_V - E_W + E_D = 0
    R = -EV - EW + ED                                     # (K,)
    return (R ** 2).mean()


# ============================================================================
# Training
# ============================================================================
pf  = PushforwardNet(D_BASE, DIM).to(device)
tf  = TestFunctions(K).to(device)

gen_opt  = optim.Adam(pf.parameters(), lr=LR_GEN)
test_opt = optim.SGD(tf.parameters(),  lr=LR_TEST)

loss_log = []
t0 = pytime.time()

print(f"\n1D Stationary MKV  |  lambda={LAM}  |  exact std={STD_STAR:.4f}")
print(f"K={K}, M={M}, M_W={M_W}, D_BASE={D_BASE}, "
      f"N_EPOCHS={N_EPOCHS}, ADV_FREQ={ADV_FREQ}")
print(f"PF params: {sum(p.numel() for p in pf.parameters()):,}")

for epoch in range(N_EPOCHS):
    # adversary step (ascent)
    if epoch > 0 and epoch % ADV_FREQ == 0:
        loss_adv = compute_loss(pf, tf)
        test_opt.zero_grad()
        (-loss_adv).backward()
        test_opt.step()

    # generator step (descent)
    loss = compute_loss(pf, tf)
    gen_opt.zero_grad()
    loss.backward()
    gen_opt.step()
    loss_log.append(loss.item())

    if epoch % 1000 == 0 or epoch == N_EPOCHS - 1:
        with torch.no_grad():
            xs = pf(torch.rand(10000, D_BASE, device=device)).cpu().numpy().ravel()
        print(f"  epoch {epoch:5d}  loss={loss.item():.4e}  "
              f"mean={xs.mean():.4f}(0.0)  std={xs.std():.4f}({STD_STAR:.4f})  "
              f"t={pytime.time()-t0:.1f}s")

# ============================================================================
# Plots
# ============================================================================
with torch.no_grad():
    xs = pf(torch.rand(20000, D_BASE, device=device)).cpu().numpy().ravel()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.hist(xs, bins=80, density=True, alpha=0.6, color='royalblue',
        edgecolor='none', label='WANPM samples')
xg = np.linspace(-3, 3, 500)
ax.plot(xg, scipy_norm.pdf(xg, 0, STD_STAR), 'r--', lw=2,
        label=f'Exact $\\mathcal{{N}}(0,{VAR_STAR:.3f})$')
ax.set_xlabel('$x$', fontsize=13); ax.set_ylabel('Density', fontsize=13)
ax.set_title('1D Stationary MKV — distribution', fontsize=12)
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(loss_log, color='steelblue', lw=1.2)
ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training loss', fontsize=12)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('results/exp1_1d_stationary.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nFinal: mean={xs.mean():.5f}  std={xs.std():.5f}  "
      f"loss={loss_log[-1]:.4e}  time={pytime.time()-t0:.1f}s")
