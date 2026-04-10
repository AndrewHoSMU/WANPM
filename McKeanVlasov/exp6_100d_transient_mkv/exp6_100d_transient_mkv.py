"""
McKean-Vlasov WANPM  —  Experiment 6: 100D Transient
======================================================
Problem
    V(x) = (theta/2) ||x||^2,   W(z) = (1/2) ||z||^2,   x in R^100
    b(x, rho_t) = -lambda*x + m(t),  lambda = theta + 1

Exact:  rho_t = N(m(t), Sigma(t) I_100)
    m(t)     = m0 * exp(-lambda*t)
    Sigma(t) = (Sigma0 - sigma^2/(2*lambda)) exp(-2*lambda*t) + sigma^2/(2*lambda)

Pushforward:  F(t, x0, r) = x0 + sqrt(t) * net(t, r)
  (network input: (t, r), D_BASE=50 following reference)

Metric: per-dimension mean absolute error and variance error,
        reported as averages across all 100 dimensions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time as pytime

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Problem parameters
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

def exact_mean_np(t):
    return MU0_NP * np.exp(-THETA * t)

def exact_var(t):
    return (SIG0**2 - VAR_EQ) * np.exp(-2.0 * LAM * t) + VAR_EQ

# ============================================================================
# Hyperparameters
# ============================================================================
K        = 5000
M        = 10000
M_0      = 2000
M_T      = 2000
M_W      = 2 * M
D_BASE   = 2 * DIM
N_EPOCHS = 10000
LR_GEN   = 1e-3
LR_TEST  = 1e-2
ADV_FREQ = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
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

    # Interaction E_W  (M_W iid pairs at fresh times)
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

print(f"\n100D Transient MKV  |  T={T_END}  |  lambda={LAM}  |  eq_var={VAR_EQ:.4f}")
print(f"MU0[:5]={MU0_NP[:5].round(3)}")
print(f"K={K}, M={M}, M_W={M_W}, M_0={M_0}, M_T={M_T}, D_BASE={D_BASE}")
print(f"N_EPOCHS={N_EPOCHS}, ADV_FREQ={ADV_FREQ}")
print(f"PF params: {sum(p.numel() for p in pf.parameters()):,}")

for epoch in range(N_EPOCHS):
    if epoch > 0 and epoch % ADV_FREQ == 0:
        loss_adv = compute_loss(pf, tf)
        test_opt.zero_grad(); (-loss_adv).backward(); test_opt.step()

    loss = compute_loss(pf, tf)
    gen_opt.zero_grad(); loss.backward(); gen_opt.step()
    loss_log.append(loss.item())

    if epoch % 500 == 0 or epoch == N_EPOCHS - 1:
        elapsed = pytime.time() - t0
        avg_ep  = elapsed / (epoch + 1)
        eta     = avg_ep * (N_EPOCHS - epoch - 1)
        print(f"  epoch {epoch:5d}  loss={loss.item():.4e}  "
              f"t={elapsed:.1f}s  ETA={eta/60:.1f}min")

# ============================================================================
# Validation table
# ============================================================================
val_times = [0.1, 0.5, 1.0]
print(f"\n{'t':>5}  {'mean_err (avg)':>15}  {'var_err (avg)':>15}  "
      f"{'mean_err (L2)':>15}  {'var_err (L2)':>15}")
print("-" * 65)
with torch.no_grad():
    for tv in val_times:
        tb  = tv * torch.ones(10000, 1, device=device)
        xs  = pf(tb, sample_ic(10000),
                 torch.rand(10000, D_BASE, device=device)).cpu().numpy()
        me_abs = np.abs(xs.mean(0) - exact_mean_np(tv)).mean()
        ve_abs = np.abs(xs.var(0)  - exact_var(tv)).mean()
        me_l2  = np.sqrt(np.sum((xs.mean(0) - exact_mean_np(tv))**2))
        ve_l2  = np.sqrt(np.sum((xs.var(0)  - exact_var(tv))**2))
        print(f"  {tv:.1f}  {me_abs:15.6f}  {ve_abs:15.6f}  "
              f"{me_l2:15.6f}  {ve_l2:15.6f}")

# ============================================================================
# Plots
# ============================================================================
# (a) Mean and variance evolution for dims 0,1,2
t_ev   = np.linspace(EPS_T, T_END, 100)
DIMS_PLOT = [0, 1, 2]
m_lrn = {d: [] for d in DIMS_PLOT}
v_lrn = {d: [] for d in DIMS_PLOT}

with torch.no_grad():
    for tv in t_ev:
        tb  = tv * torch.ones(3000, 1, device=device)
        xs  = pf(tb, sample_ic(3000),
                 torch.rand(3000, D_BASE, device=device)).cpu().numpy()
        for d in DIMS_PLOT:
            m_lrn[d].append(xs[:, d].mean())
            v_lrn[d].append(xs[:, d].var())

fig, axes = plt.subplots(2, len(DIMS_PLOT), figsize=(14, 8))
for col, d in enumerate(DIMS_PLOT):
    ax = axes[0, col]
    ax.plot(t_ev, m_lrn[d], color='royalblue', lw=2, label='WANPM')
    ax.plot(t_ev, MU0_NP[d] * np.exp(-THETA * t_ev), 'r--', lw=2, label='Exact')
    ax.set_xlabel('$t$'); ax.set_ylabel(f'Mean (dim {d})')
    ax.set_title(f'Mean — dim {d}'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, col]
    ax.plot(t_ev, v_lrn[d], color='royalblue', lw=2, label='WANPM')
    ax.plot(t_ev, [exact_var(tv) for tv in t_ev], 'r--', lw=2, label='Exact')
    ax.axhline(VAR_EQ, color='seagreen', lw=1.2, ls=':',
               label=f'Eq. var ({VAR_EQ:.3f})')
    ax.set_xlabel('$t$'); ax.set_ylabel(f'Var (dim {d})')
    ax.set_title(f'Variance — dim {d}'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('100D Transient McKean-Vlasov — evolution', fontsize=13)
plt.tight_layout()
plt.savefig('results/exp6_100d_transient_evolution.png', dpi=150, bbox_inches='tight')
plt.show()

# (b) Error heatmap + loss
with torch.no_grad():
    hmap_times = np.linspace(0.05, T_END, 20)
    all_me = []
    for tv in hmap_times:
        tb  = tv * torch.ones(5000, 1, device=device)
        xs  = pf(tb, sample_ic(5000),
                 torch.rand(5000, D_BASE, device=device)).cpu().numpy()
        all_me.append(np.abs(xs.mean(0) - exact_mean_np(tv)))
all_me = np.array(all_me)   # (n_times, DIM)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
im = ax.imshow(all_me.T, aspect='auto', cmap='viridis',
               extent=[hmap_times[0], hmap_times[-1], 0, DIM])
ax.set_xlabel('Time'); ax.set_ylabel('Dimension')
ax.set_title('|Mean error| across dimensions')
plt.colorbar(im, ax=ax, label='|error|')

ax = axes[1]
ax.semilogy(loss_log, color='steelblue', lw=1.2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('Training loss'); ax.grid(True, which='both', alpha=0.3)

plt.suptitle('100D Transient McKean-Vlasov', fontsize=13)
plt.tight_layout()
plt.savefig('results/exp6_100d_transient_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

total_time = pytime.time() - t0
print(f"\nTotal time: {total_time/60:.2f} min  |  Final loss={loss_log[-1]:.4e}")

# ============================================================================
# Save model
# ============================================================================
torch.save({
    'pf_state_dict': pf.state_dict(),
    'tf_state_dict': tf.state_dict(),
    'loss_log':      loss_log,
    'N_EPOCHS':      N_EPOCHS,
}, 'results/exp6_model.pt')
print("Model saved to results/exp6_model.pt")