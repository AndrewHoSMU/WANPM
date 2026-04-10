"""
1D Linear McKean-Vlasov (Exact Gaussian Benchmark)
Weak Adversarial Neural Pushforward Method (WANPM)

McKean-Vlasov equation (aggregation-diffusion form):
    ∂ρ/∂t = ∇·((∇V) ρ) + ∇·((∇W * ρ) ρ) + (σ²/2) Δρ

with confinement   V(x)      = (θ/2) x²
and interaction    W(x - y)  = (1/2)|x - y|²  (quadratic/granular-media kernel)

The mean-field drift simplifies to (Proposition 1 of the notes):
    b(x, ρ_t) = -θx - (x - m(t))  =  -(θ+1) x + m(t)

where m(t) = E[X_t] is the batch sample mean — no secondary sampling needed.

Exact Gaussian solution (equations (22)-(25) of the notes):
    m(t)  = m_0  exp(-(θ+1) t)
    Σ(t)  = [Σ_0 - σ²/(2(θ+1))] exp(-2(θ+1) t)  +  σ²/(2(θ+1))

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

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Problem Parameters
# ============================================================================
# V(x) = (θ/2) x²,  W(z) = (1/2)|z|²,  diffusion σ

DIM    = 1       # Spatial dimension
THETA  = 1.0     # Confinement strength
SIGMA  = 1.0     # Diffusion coefficient
T      = 2.0     # Final time
EPSILON = 0   # Small time offset to avoid t = 0 singularity

# Initial condition: ρ_0 = N(m_0, Σ_0)
M_0_INIT  = 2.0   # Initial mean  (displaced from equilibrium)
SIG_0     = 0.5   # Initial std dev

# Training hyper-parameters
K        = 1000   # Number of adversarial test functions
M        = 1000   # Interior batch size
M_0      = 100    # Initial-condition batch size
M_T      = 100    # Terminal batch size
D_BASE   = 2      # Base distribution dimension (= DIM for 1D)
N_EPOCHS = 2000
LR_GEN   = 1e-3
LR_TEST  = 1e-2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs('results_mkv1d', exist_ok=True)

# ============================================================================
# Exact Solution  (equations (22)-(25) of the notes)
# ============================================================================
# The effective mean-reversion rate is  λ = θ + 1  (confinement + quadratic interaction)

LAM = THETA + 1.0   # effective rate  λ = θ + 1

def exact_mean(t):
    """m(t) = m_0 exp(-λ t)"""
    return M_0_INIT * np.exp(-LAM * t)

def exact_variance(t):
    """Σ(t) = [Σ_0 - σ²/(2λ)] exp(-2λ t) + σ²/(2λ)"""
    sigma_eq2 = SIGMA**2 / (2.0 * LAM)   # equilibrium variance
    return (SIG_0**2 - sigma_eq2) * np.exp(-2.0 * LAM * t) + sigma_eq2

def exact_std(t):
    return np.sqrt(exact_variance(t))

# ============================================================================
# Neural Networks
# ============================================================================

class PushforwardNetwork(nn.Module):
    """
    Pushforward map:  F_θ(t, x_0, r) = x_0 + √t · F̃_θ(t, x_0, r)

    The √t factor enforces the initial condition F_θ(0, x_0, r) = x_0 by construction.
    Input to F̃_θ: (t, x_0, r) concatenated — all three are necessary (see WANPM paper).
    """
    def __init__(self, d_base, d_output, hidden_dims=(128, 128, 128)):
        super().__init__()
        layers = []
        input_dim = 1 + d_output + d_base   # t  |  x_0  |  r
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), nn.Tanh()]
            input_dim = h
        layers.append(nn.Linear(input_dim, d_output))
        self.net = nn.Sequential(*layers)

    def forward(self, t, x0, r):
        """
        t  : (B, 1)
        x0 : (B, DIM)
        r  : (B, D_BASE)
        returns x : (B, DIM)
        """
        inp = torch.cat([t, x0, r], dim=1)
        return x0 + torch.sqrt(t) * self.net(inp)


class TestFunctions(nn.Module):
    """
    Plane-wave test functions:
        f^(k)(t, x) = sin(w^(k) x + κ^(k) t + b^(k))

    Derivatives are analytic — no automatic differentiation needed for the
    test-function branch:
        ∂_t f = κ cos(wx + κt + b)
        ∇_x f = w cos(wx + κt + b)
        Δ f   = -w² sin(wx + κt + b)
    """
    def __init__(self, n_spatial, n_funcs):
        super().__init__()
        self.w     = nn.Parameter(torch.randn(n_spatial, n_funcs) * 0.3)
        self.kappa = nn.Parameter(torch.randn(n_funcs))
        self.b     = nn.Parameter(torch.rand(n_funcs) * 2 * np.pi)

    def _arg(self, x, t):
        """w·x + κt + b,  shapes: (B, K)"""
        return x @ self.w + self.kappa * t + self.b

    def eval(self, x, t):
        return torch.sin(self._arg(x, t))

    def dt(self, x, t):
        """∂f/∂t = κ cos(arg)"""
        return self.kappa * torch.cos(self._arg(x, t))

    def laplacian(self, x, t):
        """Δf = -||w||² sin(arg)"""
        w2 = (self.w**2).sum(dim=0, keepdim=True)   # (1, K)
        return -w2 * torch.sin(self._arg(x, t))

    def drift_dot_grad(self, x, t, m_batch):
        """
        b(x, ρ_t) · ∇_x f,  where b(x, ρ_t) = -(θ+1)x + m(t)
        ∇_x f = w cos(arg)
        b · ∇_x f = [-(θ+1)x + m(t)] · w  ·  cos(arg)
        m_batch: (B, 1)  — sample mean of the current batch
        """
        drift = -(THETA + 1.0) * x + m_batch    # (B, DIM)
        return (drift @ self.w) * torch.cos(self._arg(x, t))  # (B, K)

# ============================================================================
# Loss Function
# ============================================================================

def compute_loss(pf_net, tf):
    """
    Weak-form residual (equation (15) of the notes):
        R^(k) = E_T^(k) - E_0^(k) - Ê^(k)  ≈  0  ∀ k

    Loss = mean_k  (R^(k))²
    """

    # ---- Interior term  Ê^(k) ------------------------------------------
    t_int  = EPSILON + (T - EPSILON) * torch.rand(M, 1, device=device)
    x0_int = M_0_INIT + SIG_0 * torch.randn(M, DIM, device=device)
    r_int  = torch.randn(M, D_BASE, device=device)

    x_int  = pf_net(t_int, x0_int, r_int)                     # (M, 1)

    # Batch mean: m̂(t) used as MC estimate of E[X_t]
    m_batch = x_int.mean(dim=0, keepdim=True)                  # (1, 1)
    m_batch = m_batch.expand(M, 1)                             # (M, 1)

    df_dt   = tf.dt(x_int, t_int)                              # (M, K)
    lap_f   = (SIGMA**2 / 2.0) * tf.laplacian(x_int, t_int)   # (M, K)
    drft_f  = tf.drift_dot_grad(x_int, t_int, m_batch)         # (M, K)

    integrand = df_dt + lap_f + drft_f                          # (M, K)
    E_int = (T - EPSILON) * integrand.mean(dim=0)               # (K,)

    # ---- Initial term  Ê_0^(k) -----------------------------------------
    x0_ic  = M_0_INIT + SIG_0 * torch.randn(M_0, DIM, device=device)
    t0     = torch.zeros(M_0, 1, device=device)
    E_0    = tf.eval(x0_ic, t0).mean(dim=0)                    # (K,)

    # ---- Terminal term  Ê_T^(k) ----------------------------------------
    x0_T   = M_0_INIT + SIG_0 * torch.randn(M_T, DIM, device=device)
    r_T    = torch.randn(M_T, D_BASE, device=device)
    t_T    = T * torch.ones(M_T, 1, device=device)
    x_T    = pf_net(t_T, x0_T, r_T)
    E_T    = tf.eval(x_T, t_T).mean(dim=0)                     # (K,)

    # ---- Residual and loss ---------------------------------------------
    residual = E_T - E_0 - E_int                               # (K,)
    loss     = (residual**2).mean()
    return loss

# ============================================================================
# Validation
# ============================================================================

def compute_statistics(pf_net, t_vals, n_samples=8000):
    """Return learned mean and variance at each requested time."""
    means, variances = [], []
    with torch.no_grad():
        for tv in t_vals:
            t_b  = tv * torch.ones(n_samples, 1, device=device)
            x0_b = M_0_INIT + SIG_0 * torch.randn(n_samples, DIM, device=device)
            r_b  = torch.randn(n_samples, D_BASE, device=device)
            xs   = pf_net(t_b, x0_b, r_b).cpu().numpy().squeeze()   # (n_samples,)
            means.append(xs.mean())
            variances.append(xs.var())
    return np.array(means), np.array(variances)

# ============================================================================
# Training
# ============================================================================

print("=" * 65)
print("1D Linear McKean-Vlasov — WANPM")
print("=" * 65)
print(f"V(x) = (θ/2)x²,  W(z) = (1/2)|z|²")
print(f"θ = {THETA},  σ = {SIGMA},  T = {T}")
print(f"ρ_0 = N({M_0_INIT}, {SIG_0**2:.2f})")
print(f"Effective rate λ = θ+1 = {LAM}")
print(f"Equilibrium variance σ²/(2λ) = {SIGMA**2/(2*LAM):.4f}")
print(f"K = {K},  M = {M},  epochs = {N_EPOCHS}")
print("=" * 65)

pf_net = PushforwardNetwork(D_BASE, DIM, hidden_dims=(32, 32)).to(device)
tf     = TestFunctions(DIM, K).to(device)

print(f"Pushforward parameters : {sum(p.numel() for p in pf_net.parameters()):,}")
print(f"Test-function parameters: {sum(p.numel() for p in tf.parameters()):,}")

gen_opt  = optim.Adam(pf_net.parameters(), lr=LR_GEN)
test_opt = optim.SGD(tf.parameters(),      lr=LR_TEST)

loss_log = []
t_start  = pytime.time()

for epoch in range(N_EPOCHS):

    # Adversary update (maximize loss) every 10 steps
    if epoch % 2 == 0 and epoch > 0:
        for _ in range(1):
            adv_loss = compute_loss(pf_net, tf)
            test_opt.zero_grad()
            (-adv_loss).backward()
            test_opt.step()

    # Generator update (minimize loss)
    g_loss = compute_loss(pf_net, tf)
    gen_opt.zero_grad()
    g_loss.backward()
    gen_opt.step()

    loss_log.append(g_loss.item())

    if epoch % 500 == 0 or epoch == N_EPOCHS - 1:
        elapsed = pytime.time() - t_start
        eta     = elapsed / (epoch + 1) * (N_EPOCHS - epoch - 1)
        print(f"Epoch {epoch:5d}/{N_EPOCHS}  loss={g_loss.item():.6e}  "
              f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

total_time = pytime.time() - t_start
print(f"\nTraining completed in {total_time/60:.2f} min")

# ============================================================================
# Plotting
# ============================================================================

# Dense time grid for smooth curves
t_plot = np.linspace(0.0, T, 200)

# Learned statistics along the dense grid
means_learned, vars_learned = compute_statistics(pf_net, t_plot)

# Exact statistics
means_exact = exact_mean(t_plot)
vars_exact  = exact_variance(t_plot)
stds_exact  = np.sqrt(vars_exact)

# ---------- Figure 1: Mean and Variance vs Time --------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# -- Mean --
ax = axes[0]
ax.plot(t_plot, means_learned, color='royalblue', lw=2,    label='WANPM (learned)')
ax.plot(t_plot, means_exact,   color='firebrick', lw=2,
        linestyle='--',                                     label='Exact')
ax.axhline(y=0.0, color='seagreen', lw=1.2, linestyle=':',
           label='Equilibrium ($m=0$)')
ax.set_xlabel('$t$',  fontsize=13)
ax.set_ylabel('Mean $m(t)$', fontsize=13)
ax.set_title('Mean vs Time', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, T)
ax.grid(True, alpha=0.3)

# -- Variance --
sigma_eq2 = SIGMA**2 / (2.0 * LAM)
ax = axes[1]
ax.plot(t_plot, vars_learned, color='royalblue', lw=2,    label='WANPM (learned)')
ax.plot(t_plot, vars_exact,   color='firebrick', lw=2,
        linestyle='--',                                     label='Exact')
ax.axhline(y=sigma_eq2, color='seagreen', lw=1.2, linestyle=':',
           label=f'Equilibrium ($\\Sigma_{{eq}}={sigma_eq2:.3f}$)')
ax.set_xlabel('$t$',  fontsize=13)
ax.set_ylabel('Variance $\\Sigma(t)$', fontsize=13)
ax.set_title('Variance vs Time', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, T)
ax.grid(True, alpha=0.3)

plt.suptitle('1D Linear McKean–Vlasov: WANPM vs Exact Gaussian Solution',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('results_mkv1d/mean_variance_vs_time.png', dpi=150, bbox_inches='tight')
print("Saved: results_mkv1d/mean_variance_vs_time.png")
plt.show()

# ---------- Figure 2: PDF snapshots at several times --------------------------
snap_times = [0.0, 0.25, 0.5, 1.0, T]
n_snap     = len(snap_times)
fig, axes  = plt.subplots(1, n_snap, figsize=(4 * n_snap, 4), sharey=False)

n_samp = 10000
x_grid = np.linspace(-4.0, 6.0, 500)

with torch.no_grad():
    for ax, tv in zip(axes, snap_times):
        if tv == 0.0:
            # Draw directly from the known initial density
            samples = (M_0_INIT + SIG_0 * torch.randn(n_samp)).numpy()
        else:
            t_b  = tv * torch.ones(n_samp, 1, device=device)
            x0_b = M_0_INIT + SIG_0 * torch.randn(n_samp, DIM, device=device)
            r_b  = torch.randn(n_samp, D_BASE, device=device)
            samples = pf_net(t_b, x0_b, r_b).cpu().numpy().squeeze()

        m_e = exact_mean(tv)
        s_e = exact_std(tv)
        from scipy.stats import norm as scipy_norm
        pdf_e = scipy_norm.pdf(x_grid, loc=m_e, scale=s_e)

        ax.hist(samples, bins=60, density=True, alpha=0.55,
                color='royalblue', edgecolor='none', label='WANPM')
        ax.plot(x_grid, pdf_e, color='firebrick', lw=2, label='Exact')
        ax.set_title(f'$t = {tv:.2f}$', fontsize=12)
        ax.set_xlabel('$x$', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

plt.suptitle('1D Linear McKean–Vlasov: Distribution Snapshots', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('results_mkv1d/pdf_snapshots.png', dpi=150, bbox_inches='tight')
print("Saved: results_mkv1d/pdf_snapshots.png")
plt.show()

# ---------- Figure 3: Error vs Time ------------------------------------------
mean_error = np.abs(means_learned - means_exact)
var_error  = np.abs(vars_learned  - vars_exact)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].semilogy(t_plot, mean_error + 1e-10, color='royalblue', lw=2)
axes[0].set_xlabel('$t$', fontsize=13)
axes[0].set_ylabel('$|m_{\\mathrm{learned}}(t) - m_{\\mathrm{exact}}(t)|$', fontsize=12)
axes[0].set_title('Mean Absolute Error vs Time', fontsize=13)
axes[0].set_xlim(0, T)
axes[0].grid(True, which='both', alpha=0.3)

axes[1].semilogy(t_plot, var_error + 1e-10, color='firebrick', lw=2)
axes[1].set_xlabel('$t$', fontsize=13)
axes[1].set_ylabel('$|\\Sigma_{\\mathrm{learned}}(t) - \\Sigma_{\\mathrm{exact}}(t)|$', fontsize=12)
axes[1].set_title('Variance Absolute Error vs Time', fontsize=13)
axes[1].set_xlim(0, T)
axes[1].grid(True, which='both', alpha=0.3)

plt.suptitle('1D Linear McKean–Vlasov: Pointwise Errors', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('results_mkv1d/errors_vs_time.png', dpi=150, bbox_inches='tight')
print("Saved: results_mkv1d/errors_vs_time.png")
plt.show()

# ---------- Figure 4: Training Loss ------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(loss_log, color='steelblue', lw=1.5)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training Loss Convergence — 1D Linear McKean–Vlasov', fontsize=12)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('results_mkv1d/loss.png', dpi=150, bbox_inches='tight')
print("Saved: results_mkv1d/loss.png")
plt.show()

# ============================================================================
# Numerical Summary
# ============================================================================

val_times = np.array([0.1, 0.25, 0.5, 1.0, T])
m_lrn, v_lrn = compute_statistics(pf_net, val_times)
m_ext = exact_mean(val_times)
v_ext = exact_variance(val_times)

print("\n" + "=" * 65)
print("Numerical Summary")
print("=" * 65)
print(f"{'t':>6}  {'m_exact':>10}  {'m_learned':>10}  {'|Δm|':>10}  "
      f"{'Σ_exact':>10}  {'Σ_learned':>10}  {'|ΔΣ|':>10}")
print("-" * 65)
for i, tv in enumerate(val_times):
    print(f"{tv:6.2f}  {m_ext[i]:10.5f}  {m_lrn[i]:10.5f}  "
          f"{abs(m_lrn[i]-m_ext[i]):10.2e}  "
          f"{v_ext[i]:10.5f}  {v_lrn[i]:10.5f}  "
          f"{abs(v_lrn[i]-v_ext[i]):10.2e}")
print("=" * 65)
print(f"Total training time: {total_time/60:.2f} min")
print(f"Final loss: {loss_log[-1]:.4e}")
print("Outputs saved to:  results_mkv1d/")
print("=" * 65)