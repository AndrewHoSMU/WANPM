"""
1D Stationary McKean-Vlasov (Exact Gaussian Benchmark)
Weak Adversarial Neural Pushforward Method (WANPM)

Stationary McKean-Vlasov equation:
    -∇·(b(x, ρ) ρ(x)) + (σ²/2) Δρ(x) = 0,   ∫ρ dx = 1

with confinement   V(x)      = (θ/2) x²
and interaction    W(x - y)  = (1/2)|x - y|²  (quadratic/granular-media kernel)

Mean-field drift (Proposition 1 of the notes):
    b(x, ρ) = -∇V(x) - (x - m*)  =  -(θ+1) x + m*
where m* = E_ρ[X] is the mean of the stationary distribution.

By symmetry of V and W, m* = 0, so b(x, ρ*) = -(θ+1) x = -λ x,  λ = θ+1.

Exact stationary solution (self-consistent Gaussian):
    ρ* = N(0, σ²/(2λ))

Weak formulation (equation (8)-(9) of the notes), with plane-wave test functions
f^(k)(x) = sin(w^(k) x + b^(k)):

    E_ρ[L[ρ] f^(k)(x)] = 0   ∀ k

where  L[ρ] f(x) = b(x, ρ)·∇f(x) + (σ²/2) Δf(x)

For plane waves:
    ∇f  =  w cos(wx + b)
    Δf  = -w² sin(wx + b)

The pushforward is F_θ: R^D_BASE → R^DIM  (no t, no x_0).
The loss is the adversarial squared weak residual:

    L[θ, {η^(k)}] = (1/K) Σ_k [ (1/M) Σ_m L[ρ]f^(k)(F_θ(r^(m))) ]²

The mean m̂ = (1/M) Σ_m F_θ(r^(m)) is computed from the SAME primary batch,
with gradient flowing through it. This is essential: detaching m̂ breaks the
self-consistency coupling m* = E_ρ[X], allowing spurious non-Gaussian solutions
to satisfy the weak residual with correct mean and variance but wrong shape.

Author: Andrew Qing He, Wei Cai
        Department of Mathematics, Southern Methodist University
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time as pytime
from scipy.stats import norm as scipy_norm

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Problem Parameters
# ============================================================================

DIM    = 1      # Spatial dimension
THETA  = 1.0    # Confinement strength  (V(x) = θ/2 · x²)
SIGMA  = 1.0    # Diffusion coefficient

# Exact stationary solution
LAM       = THETA + 1.0          # Effective rate  λ = θ + 1
MU_STAR   = 0.0                  # Stationary mean  (m* = 0 by symmetry)
SIG_STAR  = np.sqrt(SIGMA**2 / (2.0 * LAM))   # Stationary std dev

# ============================================================================
# Training Hyper-parameters
# ============================================================================

K       = 1000   # Number of adversarial test functions
M       = 1000   # Pushforward sample batch size
D_BASE  = 3      # Base distribution dimension (= DIM for 1D target)
N_EPOCHS = 5000
LR_GEN  = 1e-3
LR_TEST = 1e-2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs('results_mkv1d_stationary', exist_ok=True)

# ============================================================================
# Neural Networks
# ============================================================================

class PushforwardNetwork(nn.Module):
    """
    Stationary pushforward:  F_θ: R^D_BASE → R^DIM
    No time, no x_0 — just maps base noise r to samples from ρ*.
    """
    def __init__(self, d_base, d_output, hidden_dims=(128, 128, 128)):
        super().__init__()
        layers = []
        input_dim = d_base
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), nn.Tanh()]
            input_dim = h
        layers.append(nn.Linear(input_dim, d_output))
        self.net = nn.Sequential(*layers)

    def forward(self, r):
        """r: (B, D_BASE) → x: (B, DIM)"""
        return self.net(r)


class TestFunctions(nn.Module):
    """
    Stationary plane-wave test functions:
        f^(k)(x) = sin(w^(k) x + b^(k))

    Derivatives:
        ∇f  =  w cos(wx + b)
        Δf  = -w² sin(wx + b)

    No κt term — stationary problem has no time dependence.
    """
    def __init__(self, n_spatial, n_funcs):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_spatial, n_funcs) * 2.0)
        self.b = nn.Parameter(torch.randn(n_funcs) * 2 * np.pi)

    def _arg(self, x):
        return x @ self.w + self.b                        # (B, K)

    def eval(self, x):
        return torch.sin(self._arg(x))

    def generator(self, x, m_batch):
        """
        L[ρ] f(x) = b(x, ρ)·∇f + (σ²/2) Δf

        b(x, ρ) = -(θ+1)x + m*  =  -λ x + m̂
        ∇f      =  w cos(arg)
        Δf      = -w² sin(arg)

        m_batch: (B, DIM) — batch mean estimate of m* = E_ρ[X]
        """
        arg   = self._arg(x)                              # (B, K)
        cos_  = torch.cos(arg)                            # (B, K)
        sin_  = torch.sin(arg)                            # (B, K)
        w2    = (self.w**2).sum(dim=0, keepdim=True)      # (1, K)

        drift      = -LAM * x + m_batch                   # (B, DIM)
        drift_grad = (drift @ self.w) * cos_              # (B, K)
        lap        = -(SIGMA**2 / 2.0) * w2 * sin_        # (B, K)

        return drift_grad + lap                            # (B, K)

# ============================================================================
# Loss Function
# ============================================================================

def compute_loss(pf_net, tf):
    """
    Stationary weak residual:
        R^(k) = (1/M) Σ_m  L[ρ] f^(k)(F_θ(r^(m)))
    Loss = (1/K) Σ_k (R^(k))²

    m̂ is computed from the SAME batch as the residual, with gradient flowing
    through it. This enforces the self-consistency condition m* = E_ρ[X]
    through the gradient — detaching m̂ breaks this coupling and allows
    spurious non-Gaussian solutions with correct mean/variance but wrong shape.
    """
    r = torch.rand(M, D_BASE, device=device)
    x = pf_net(r)                                         # (M, DIM)

    # Same-batch mean — gradient flows through m_hat
    m_hat = x.mean(dim=0, keepdim=True).expand(M, DIM)   # (M, DIM)

    Lf = tf.generator(x, m_hat)                          # (M, K)
    R  = Lf.mean(dim=0)                                   # (K,)
    return (R**2).mean()

# ============================================================================
# Training
# ============================================================================

print("=" * 60)
print("1D Stationary McKean-Vlasov — WANPM")
print("=" * 60)
print(f"θ = {THETA},  σ = {SIGMA},  λ = θ+1 = {LAM}")
print(f"Exact solution: ρ* = N({MU_STAR}, {SIG_STAR**2:.4f})")
print(f"K={K},  M={M},  D_BASE={D_BASE},  epochs={N_EPOCHS}")
print("=" * 60)

pf_net = PushforwardNetwork(D_BASE, DIM, hidden_dims=(10, 10)).to(device)
tf     = TestFunctions(DIM, K).to(device)

print(f"Pushforward parameters:   {sum(p.numel() for p in pf_net.parameters()):,}")
print(f"Test-function parameters: {sum(p.numel() for p in tf.parameters()):,}")

gen_opt  = optim.Adam(pf_net.parameters(), lr=LR_GEN)
test_opt = optim.SGD(tf.parameters(),      lr=LR_TEST)

loss_log = []
t_start  = pytime.time()

for epoch in range(N_EPOCHS):

    # Adversary update every 2 steps
    if epoch % 2 == 0 and epoch > 0:
        adv_loss = compute_loss(pf_net, tf)
        test_opt.zero_grad()
        (-adv_loss).backward()
        test_opt.step()

    # Generator update
    g_loss = compute_loss(pf_net, tf)
    gen_opt.zero_grad()
    g_loss.backward()
    gen_opt.step()

    loss_log.append(g_loss.item())

    if epoch % 500 == 0 or epoch == N_EPOCHS - 1:
        with torch.no_grad():
            r_val = torch.rand(5000, D_BASE, device=device)
            x_val = pf_net(r_val).cpu().numpy().squeeze()
        elapsed = pytime.time() - t_start
        eta     = elapsed / (epoch + 1) * (N_EPOCHS - epoch - 1)
        print(f"Epoch {epoch:5d}/{N_EPOCHS}  loss={g_loss.item():.4e}  "
              f"mean={x_val.mean():.4f} (exact {MU_STAR:.4f})  "
              f"std={x_val.std():.4f} (exact {SIG_STAR:.4f})  "
              f"ETA={eta/60:.1f}min")

total_time = pytime.time() - t_start
print(f"\nTraining completed in {total_time/60:.2f} min")

# ============================================================================
# Plotting
# ============================================================================

n_samp = 20000
with torch.no_grad():
    r_plot = torch.rand(n_samp, D_BASE, device=device)
    x_plot = pf_net(r_plot).cpu().numpy().squeeze()

x_grid = np.linspace(-3.0, 3.0, 500)
pdf_exact = scipy_norm.pdf(x_grid, loc=MU_STAR, scale=SIG_STAR)

# ---------- Figure 1: Learned distribution vs exact --------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.hist(x_plot, bins=80, density=True, alpha=0.6,
        color='royalblue', edgecolor='none', label='WANPM (learned)')
ax.plot(x_grid, pdf_exact, color='firebrick', lw=2,
        linestyle='--', label=f'Exact $\\mathcal{{N}}(0,\\ {SIG_STAR**2:.3f})$')
ax.set_xlabel('$x$', fontsize=13)
ax.set_ylabel('Density', fontsize=13)
ax.set_title('Stationary Distribution', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Q-Q plot
ax = axes[1]
x_sorted  = np.sort(x_plot)
quantiles = np.linspace(0.001, 0.999, len(x_sorted))
q_exact   = scipy_norm.ppf(quantiles, loc=MU_STAR, scale=SIG_STAR)
ax.scatter(q_exact, x_sorted, s=1, alpha=0.3, color='royalblue', label='WANPM')
ax.plot([q_exact[0], q_exact[-1]], [q_exact[0], q_exact[-1]],
        color='firebrick', lw=2, linestyle='--', label='Exact (diagonal)')
ax.set_xlabel('Exact quantiles', fontsize=13)
ax.set_ylabel('Learned quantiles', fontsize=13)
ax.set_title('Q-Q Plot', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.suptitle('1D Stationary McKean–Vlasov: WANPM vs Exact Gaussian',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('results_mkv1d_stationary/distribution.png', dpi=150, bbox_inches='tight')
print("Saved: results_mkv1d_stationary/distribution.png")
plt.show()

# ---------- Figure 2: Training loss ------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(loss_log, color='steelblue', lw=1.5)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training Loss — 1D Stationary McKean–Vlasov', fontsize=12)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('results_mkv1d_stationary/loss.png', dpi=150, bbox_inches='tight')
print("Saved: results_mkv1d_stationary/loss.png")
plt.show()

# ============================================================================
# Numerical Summary
# ============================================================================

print("\n" + "=" * 50)
print("Numerical Summary")
print("=" * 50)
print(f"  Mean:     learned = {x_plot.mean():.6f},  exact = {MU_STAR:.6f},  "
      f"|error| = {abs(x_plot.mean() - MU_STAR):.2e}")
print(f"  Std dev:  learned = {x_plot.std():.6f},  exact = {SIG_STAR:.6f},  "
      f"|error| = {abs(x_plot.std() - SIG_STAR):.2e}")
print(f"  Variance: learned = {x_plot.var():.6f},  exact = {SIG_STAR**2:.6f},  "
      f"|error| = {abs(x_plot.var() - SIG_STAR**2):.2e}")
print(f"  Training time: {total_time/60:.2f} min")
print(f"  Final loss:    {loss_log[-1]:.4e}")
print("=" * 50)