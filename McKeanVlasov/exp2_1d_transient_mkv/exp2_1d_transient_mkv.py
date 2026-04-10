"""
McKean-Vlasov WANPM  —  Experiment 2: 1D Transient
====================================================
Problem
    V(x) = (theta/2) x^2,   W(z) = (1/2) z^2,   x in R
    partial_t rho = -d/dx[ b(x,rho_t) rho ] + (sigma^2/2) d^2rho/dx^2
    b(x,rho_t) = -theta*x - (x - m(t)) = -lambda*x + m(t)

Exact solution  rho_t = N(m(t), Sigma(t)):
    m(t)     = m0 * exp(-lambda*t)
    Sigma(t) = (Sigma0 - sigma^2/(2*lambda)) * exp(-2*lambda*t)
               + sigma^2/(2*lambda)

Pushforward:  F(t, x0, r) = x0 + sqrt(t) * F_tilde(t, r)
  (following reference code: x0 added outside the network;
   network input is (t, r) only, not (t, x0, r))

Weak form (eq. weak_timedep in notes):
    E_T - E_0 - E_t + E_V + E_W - E_D = 0

    E_W (time-dep) = T/M_W * sum_m  grad_psi(t^m, xi^m) . (xi^m - eta^m)
    xi^m, eta^m  iid from rho_{t^m}  via separate base samples at same t^m
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
THETA  = 1.0
SIGMA  = 1.0
LAM    = THETA + 1.0
T_END  = 1.0
EPS_T  = 1e-3    # avoid t=0 in interior sampling

MU0    = 2.0     # initial mean (displaced from 0)
SIG0   = 0.5     # initial std

VAR_EQ  = SIGMA**2 / (2.0 * LAM)
STD_EQ  = np.sqrt(VAR_EQ)

def exact_mean(t):
    return MU0 * np.exp(-THETA * t)

def exact_var(t):
    return (SIG0**2 - VAR_EQ) * np.exp(-2.0 * LAM * t) + VAR_EQ

# ============================================================================
# Hyperparameters
# ============================================================================
K        = 2000
M        = 2000
M_0      = 1000
M_T      = 1000
M_W      = 2 * M
D_BASE   = 4
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
    """F(t, x0, r) = x0 + sqrt(t) * F_tilde(t, r)
    Network input: (t, r)  — x0 added externally (matches reference style)."""
    def __init__(self, d_base, d_out, hidden=(128, 128, 128)):
        super().__init__()
        layers, in_dim = [], 1 + d_base    # t | r
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, t, x0, r):
        return x0 + torch.sqrt(t) * self.net(torch.cat([t, r], dim=1))


class TestFunctions(nn.Module):
    """psi^(k)(t,x) = sin(w^(k) x + kappa^(k) t + b^(k))"""
    def __init__(self, K):
        super().__init__()
        self.w     = nn.Parameter(torch.rand(1, K) * 0.1)
        self.kappa = nn.Parameter(torch.rand(K)    * 0.1)
        self.b     = nn.Parameter(torch.rand(K)     * 2 * np.pi)

    def _arg(self, x, t):                     # (B, K)
        return x @ self.w + self.kappa * t + self.b

    def eval(self, x, t):
        return torch.sin(self._arg(x, t))

    def dt(self, x, t):
        return self.kappa * torch.cos(self._arg(x, t))

    def grad_dot(self, x, t, v):              # v: (B, 1)
        return (v @ self.w) * torch.cos(self._arg(x, t))

    def laplacian(self, x, t):
        return -(self.w ** 2).sum(0) * torch.sin(self._arg(x, t))


# ============================================================================
# Helpers
# ============================================================================

def sample_ic(n):
    return MU0 + SIG0 * torch.randn(n, DIM, device=device)


# ============================================================================
# Loss
# ============================================================================

def compute_loss(pf, tf):
    # ---- Terminal E_T ----
    t_T  = T_END * torch.ones(M_T, 1, device=device)
    x0_T = sample_ic(M_T)
    r_T  = torch.rand(M_T, D_BASE, device=device)
    ET   = tf.eval(pf(t_T, x0_T, r_T), t_T).mean(0)          # (K,)

    # ---- Initial E_0 ----
    t_0  = torch.zeros(M_0, 1, device=device)
    E0   = tf.eval(sample_ic(M_0), t_0).mean(0)               # (K,)

    # ---- Interior: E_t, E_V, E_D ----
    t_in = EPS_T + (T_END - EPS_T) * torch.rand(M, 1, device=device)
    xi   = pf(t_in, sample_ic(M), torch.rand(M, D_BASE, device=device))

    Et = T_END * tf.dt(xi, t_in).mean(0)
    EV = T_END * tf.grad_dot(xi, t_in, THETA * xi).mean(0)     # grad_V = theta*x
    ED = T_END * (SIGMA**2 / 2.0) * tf.laplacian(xi, t_in).mean(0)

    # ---- Interaction E_W  (M_W pairs at fresh times) ----
    t_W  = EPS_T + (T_END - EPS_T) * torch.rand(M_W, 1, device=device)
    xi2  = pf(t_W, sample_ic(M_W), torch.rand(M_W, D_BASE, device=device))
    eta  = pf(t_W, sample_ic(M_W), torch.rand(M_W, D_BASE, device=device))
    EW   = T_END * tf.grad_dot(xi2, t_W, xi2 - eta).mean(0)   # (K,)

    # Residual:  E_T - E_0 - E_t + E_V + E_W - E_D = 0
    R = ET - E0 - Et + EV + EW - ED
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

print(f"\n1D Transient MKV  |  T={T_END}  |  mu0={MU0}  |  sig0={SIG0}  "
      f"|  lambda={LAM}  |  eq_std={STD_EQ:.4f}")
print(f"K={K}, M={M}, M_W={M_W}, M_0={M_0}, M_T={M_T}, "
      f"N_EPOCHS={N_EPOCHS}, ADV_FREQ={ADV_FREQ}")

for epoch in range(N_EPOCHS):
    if epoch > 0 and epoch % ADV_FREQ == 0:
        loss_adv = compute_loss(pf, tf)
        test_opt.zero_grad(); (-loss_adv).backward(); test_opt.step()

    loss = compute_loss(pf, tf)
    gen_opt.zero_grad(); loss.backward(); gen_opt.step()
    loss_log.append(loss.item())

    if epoch % 1000 == 0 or epoch == N_EPOCHS - 1:
        print(f"  epoch {epoch:5d}  loss={loss.item():.4e}  "
              f"t={pytime.time()-t0:.1f}s")

# ============================================================================
# Evaluation: mean and std vs time
# ============================================================================
t_vals = np.linspace(EPS_T, T_END, 200)
m_lrn, s_lrn = [], []
with torch.no_grad():
    for tv in t_vals:
        tb  = tv * torch.ones(8000, 1, device=device)
        xs  = pf(tb, sample_ic(8000),
                 torch.rand(8000, D_BASE, device=device)).cpu().numpy().ravel()
        m_lrn.append(xs.mean()); s_lrn.append(xs.std())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.plot(t_vals, m_lrn,               color='royalblue', lw=2, label='WANPM')
ax.plot(t_vals, exact_mean(t_vals),  'r--', lw=2,             label='Exact')
ax.axhline(0, color='seagreen', lw=1.2, ls=':', label='Equilibrium (0)')
ax.set_xlabel('$t$', fontsize=13); ax.set_ylabel('$m(t)$', fontsize=13)
ax.set_title('Mean vs time'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t_vals, s_lrn,                               color='royalblue', lw=2, label='WANPM')
ax.plot(t_vals, np.sqrt([exact_var(v) for v in t_vals]), 'r--', lw=2, label='Exact')
ax.axhline(STD_EQ, color='seagreen', lw=1.2, ls=':',
           label=f'Eq. std ({STD_EQ:.3f})')
ax.set_xlabel('$t$', fontsize=13); ax.set_ylabel('$\\sigma(t)$', fontsize=13)
ax.set_title('Std vs time'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.semilogy(loss_log, color='steelblue', lw=1.2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training loss'); ax.grid(True, which='both', alpha=0.3)

plt.suptitle('1D Transient McKean-Vlasov', fontsize=13)
plt.tight_layout()
plt.savefig('results/exp2_1d_transient.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nFinal loss={loss_log[-1]:.4e}  time={pytime.time()-t0:.1f}s")