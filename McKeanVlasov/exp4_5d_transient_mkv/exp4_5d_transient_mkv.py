"""
McKean-Vlasov WANPM  —  Experiment 4: 5D Transient
====================================================
Problem
    V(x) = (theta/2) ||x||^2,   W(z) = (1/2) ||z||^2,   x in R^5
    b(x, rho_t) = -lambda*x + m(t),  lambda = theta + 1

Exact:  rho_t = N(m(t), Sigma(t) I_5)
    m(t)    = m0 * exp(-lambda*t)                      (componentwise)
    Sigma(t) = (Sigma0 - sigma^2/(2*lambda)) exp(-2*lambda*t) + sigma^2/(2*lambda)

Pushforward:  F(t, x0, r) = x0 + sqrt(t) * F_tilde(t, r)
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
DIM    = 5
THETA  = 1.0
SIGMA  = 1.0
LAM    = THETA + 1.0
T_END  = 1.0
EPS_T  = 1e-3

np.random.seed(0)
MU0_NP  = np.random.randn(DIM) * 2.0    # diverse initial means
SIG0    = 0.5
VAR_EQ  = SIGMA**2 / (2.0 * LAM)

def exact_mean_np(t):
    return MU0_NP * np.exp(-THETA * t)

def exact_var(t):
    return (SIG0**2 - VAR_EQ) * np.exp(-2.0 * LAM * t) + VAR_EQ

# ============================================================================
# Hyperparameters
# ============================================================================
K        = 3000
M        = 3000
M_0      = 1000
M_T      = 1000
M_W      = 2 * M
D_BASE   = 16
N_EPOCHS = 10000
LR_GEN   = 1e-3
LR_TEST  = 1e-2
ADV_FREQ = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
os.makedirs('results', exist_ok=True)

MU0 = torch.tensor(MU0_NP, dtype=torch.float32, device=device)

# ============================================================================
# Networks
# ============================================================================

class PushforwardNet(nn.Module):
    """F(t, x0, r) = x0 + sqrt(t) * net(t, r)"""
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
    """psi^(k)(t,x) = sin(w^(k).x + kappa^(k) t + b^(k))"""
    def __init__(self, n_spatial, K):
        super().__init__()
        self.w     = nn.Parameter(torch.rand(n_spatial, K) * 0.1)
        self.kappa = nn.Parameter(torch.rand(K) * 0.1)
        self.b     = nn.Parameter(torch.rand(K) * 2 * np.pi)

    def _arg(self, x, t):
        return x @ self.w + self.kappa * t + self.b

    def eval(self, x, t):        return torch.sin(self._arg(x, t))
    def dt(self, x, t):          return self.kappa * torch.cos(self._arg(x, t))
    def grad_dot(self, x, t, v): return (v @ self.w) * torch.cos(self._arg(x, t))
    def laplacian(self, x, t):   return -(self.w**2).sum(0) * torch.sin(self._arg(x, t))


# ============================================================================
# Helpers
# ============================================================================

def sample_ic(n):
    return MU0 + SIG0 * torch.randn(n, DIM, device=device)


# ============================================================================
# Loss
# ============================================================================

def compute_loss(pf, tf):
    # Terminal
    t_T = T_END * torch.ones(M_T, 1, device=device)
    ET  = tf.eval(pf(t_T, sample_ic(M_T), torch.rand(M_T, D_BASE, device=device)),
                  t_T).mean(0)

    # Initial
    t_0 = torch.zeros(M_0, 1, device=device)
    E0  = tf.eval(sample_ic(M_0), t_0).mean(0)

    # Interior
    t_in = EPS_T + (T_END - EPS_T) * torch.rand(M, 1, device=device)
    xi   = pf(t_in, sample_ic(M), torch.rand(M, D_BASE, device=device))
    Et   = T_END * tf.dt(xi, t_in).mean(0)
    EV   = T_END * tf.grad_dot(xi, t_in, THETA * xi).mean(0)
    ED   = T_END * (SIGMA**2 / 2.0) * tf.laplacian(xi, t_in).mean(0)

    # Interaction E_W
    t_W  = EPS_T + (T_END - EPS_T) * torch.rand(M_W, 1, device=device)
    xi2  = pf(t_W, sample_ic(M_W), torch.rand(M_W, D_BASE, device=device))
    eta  = pf(t_W, sample_ic(M_W), torch.rand(M_W, D_BASE, device=device))
    EW   = T_END * tf.grad_dot(xi2, t_W, xi2 - eta).mean(0)

    R = ET - E0 - Et + EV + EW - ED
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

print(f"\n5D Transient MKV  |  T={T_END}  |  lambda={LAM}  |  eq_var={VAR_EQ:.4f}")
print(f"MU0[:3]={MU0_NP[:3].round(3)}")
print(f"K={K}, M={M}, M_W={M_W}, D_BASE={D_BASE}, "
      f"N_EPOCHS={N_EPOCHS}, ADV_FREQ={ADV_FREQ}")

for epoch in range(N_EPOCHS):
    if epoch > 0 and epoch % ADV_FREQ == 0:
        loss_adv = compute_loss(pf, tf)
        test_opt.zero_grad(); (-loss_adv).backward(); test_opt.step()

    loss = compute_loss(pf, tf)
    gen_opt.zero_grad(); loss.backward(); gen_opt.step()
    loss_log.append(loss.item())

    if epoch % 2000 == 0 or epoch == N_EPOCHS - 1:
        print(f"  epoch {epoch:5d}  loss={loss.item():.4e}  "
              f"t={pytime.time()-t0:.1f}s")

# ============================================================================
# Validation table
# ============================================================================
val_times = [0.1, 0.5, 1.0]
print(f"\n{'t':>5}  {'mean_err (avg)':>15}  {'var_err (avg)':>15}")
print("-" * 40)
with torch.no_grad():
    for tv in val_times:
        tb  = tv * torch.ones(10000, 1, device=device)
        xs  = pf(tb, sample_ic(10000),
                 torch.rand(10000, D_BASE, device=device)).cpu().numpy()
        me  = np.abs(xs.mean(0) - exact_mean_np(tv)).mean()
        ve  = np.abs(xs.var(0)  - exact_var(tv)).mean()
        print(f"  {tv:.1f}  {me:15.6f}  {ve:15.6f}")

# ============================================================================
# Plots
# ============================================================================

# (1) Training loss — single plot, 3:1 width:height ratio
fig, ax = plt.subplots(figsize=(12, 4))
ax.semilogy(loss_log, color='steelblue', lw=1.2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training loss — 5D Transient McKean-Vlasov')
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('results/exp4_5d_loss.png', dpi=150, bbox_inches='tight')
plt.show()

# (2) Histograms: dim 1, dim 2, dim DIM at each time slot
DIMS_HIST  = [0, 1, DIM - 1]        # 0-indexed → dim 1, 2, 5
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

plt.suptitle('5D Transient McKean-Vlasov — Histograms', fontsize=13)
plt.tight_layout()
plt.savefig('results/exp4_5d_histograms.png', dpi=150, bbox_inches='tight')
plt.show()

# (3) Mean ± σ evolution: dim 1, dim 2, dim DIM
DIMS_PLOT = [0, 1, DIM - 1]
t_ev = np.linspace(EPS_T, T_END, 150)
m_lrn = {d: [] for d in DIMS_PLOT}
s_lrn = {d: [] for d in DIMS_PLOT}

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

plt.suptitle('5D Transient McKean-Vlasov — Mean ± σ Evolution', fontsize=13)
plt.tight_layout()
plt.savefig('results/exp4_5d_transient.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFinal loss={loss_log[-1]:.4e}  time={pytime.time()-t0:.1f}s")