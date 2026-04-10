"""
Time-Dependent 1D Double Well Fokker-Planck - Weak Adversarial Neural Pushforward Method
Solves the time-dependent Fokker-Planck equation in 1D with double well potential
∂ρ/∂t = ∇·(∇V(x)ρ) + (σ²/2)Δρ where V(x) = (x² - 1)²

MODIFIED: Uses ResNet architecture for the PushforwardNetwork
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
# Problem Parameters: 1D Double Well Potential
# ============================================================================
# dX_t = -∇V(X_t)dt + σ dW_t where V(x) = (x² - 1)²
# ∂ρ/∂t = ∇·(∇V(x)ρ) + (σ²/2)Δρ
# Drift: b(x) = -dV/dx = -4x³ + 4x

DIM = 1  # Spatial dimension
SIGMA = 1.0  # Noise strength
T = 1.0  # Final time
EPSILON = 1e-3  # Small time to avoid t=0 singularity

# Initial condition: Gaussian centered at x=2 (far from equilibrium)
MU_0 = 0.0  # Initial mean
SIGMA_0 = 1.0  # Initial std dev

# Training parameters
K = 5000  # Number of test functions
M = 5000  # Batch size (interior)
M_0 = 1000  # Batch size (t=0)
M_T = 1000  # Batch size (t=T)
D_BASE = 8  # Base distribution dimension
N_EPOCHS = 100_000
LR_GEN = 5e-2
LR_TEST = 1e-1

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
os.makedirs('results_doublewell1d', exist_ok=True)


# ============================================================================
# Double Well Potential Functions
# ============================================================================

def potential(x):
    """V(x) = (x² - 1)²"""
    return (x ** 2 - 1) ** 2


def drift_function(x):
    """Drift: b(x) = -dV/dx = -4x³ + 4x"""
    return -4 * x ** 3 + 4 * x


def analytical_steady_state_pdf(x, sigma=SIGMA):
    """
    Analytical steady-state distribution (Boltzmann distribution)
    ρ_∞(x) ∝ exp(-2V(x)/σ²)
    """
    V = potential(x)
    f = np.exp(-2 * V / (sigma ** 2))
    # Normalize
    dx = x[1] - x[0] if len(x) > 1 else 0.01
    f = f / (np.sum(f) * dx)
    return f


# ============================================================================
# Neural Networks
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        residual = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out = out + residual  # Skip connection
        out = self.activation(out)
        return out


class PushforwardNetwork(nn.Module):
    """F_θ(t, x_0, r) = x_0 + √t * F̃_θ(t, r) with ResNet architecture"""

    def __init__(self, d_base, d_output, hidden_dim=32, n_residual_blocks=3):
        super().__init__()
        input_dim = 1 + d_base
        
        # Initial embedding layer
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_residual_blocks)
        ])
        
        # Output layer
        self.output = nn.Linear(hidden_dim, d_output)

    def forward(self, t, x_0, r):
        """
        Args:
            t: (batch, 1) - time
            x_0: (batch, DIM) - initial positions
            r: (batch, D_BASE) - base noise
        Returns:
            x: (batch, DIM) - pushed forward samples
        """
        t_r = torch.cat([t, r], dim=1)
        
        # Embed input
        h = self.embed(t_r)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            h = block(h)
        
        # Compute output
        delta = self.output(h)
        return x_0 + torch.sqrt(t) * delta


class TestFunctions(nn.Module):
    """f^(k)(x,t) = sin(w^(k)x + κ^(k)t + b^(k))"""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_in, n_out) * 0.5)
        self.kappa = nn.Parameter(torch.randn(n_out))
        self.b = nn.Parameter(torch.rand(n_out) * 2 * np.pi)

    def forward(self, x, t):
        """Evaluate test functions"""
        return torch.sin(x @ self.w + self.kappa * t + self.b)

    def time_derivative(self, x, t):
        """∂f/∂t = κ cos(wx + κt + b)"""
        return self.kappa * torch.cos(x @ self.w + self.kappa * t + self.b)

    def laplacian(self, x, t):
        """Δf = -w² sin(wx + κt + b)"""
        w_squared = (self.w ** 2).sum(dim=0, keepdim=True)
        return -w_squared * torch.sin(x @ self.w + self.kappa * t + self.b)

    def gradient_dot_drift(self, x, t):
        """b(x)·∇f where b(x) = -4x³ + 4x (double well drift)
        ∇f = w cos(wx + κt + b)
        b·∇f = b(x)·w cos(wx + κt + b)
        """
        drift = drift_function(x)
        return (drift @ self.w) * torch.cos(x @ self.w + self.kappa * t + self.b)


# ============================================================================
# Loss Function
# ============================================================================

def compute_loss(pushforward_net, test_funcs):
    """Compute the three-term weak form loss"""

    # ========== Interior term: ∫_ε^T 𝔼[∂f/∂t + Lf] dt ==========
    t_interior = EPSILON + (T - EPSILON) * torch.rand(M, 1, device=device)
    r_interior = torch.randn(M, D_BASE, device=device)
    x_0_interior = MU_0 + SIGMA_0 * torch.randn(M, DIM, device=device)

    x_interior = pushforward_net(t_interior, x_0_interior, r_interior)

    # ∂f/∂t
    df_dt = test_funcs.time_derivative(x_interior, t_interior)

    # Lf = (σ²/2)Δf + b·∇f
    laplacian_term = (SIGMA ** 2 / 2) * test_funcs.laplacian(x_interior, t_interior)
    drift_term = test_funcs.gradient_dot_drift(x_interior, t_interior)

    interior_integrand = df_dt + laplacian_term + drift_term
    E_interior = (T - EPSILON) * interior_integrand.mean(dim=0)

    # ========== Initial condition term: -𝔼_{ρ_0}[f(0,·)] ==========
    x_0_samples = MU_0 + SIGMA_0 * torch.randn(M_0, DIM, device=device)
    t_0 = torch.zeros(M_0, 1, device=device)

    f_at_0 = test_funcs(x_0_samples, t_0)
    E_0 = f_at_0.mean(dim=0)

    # ========== Terminal term: 𝔼_{ρ(T,·)}[f(T,·)] ==========
    t_T = T * torch.ones(M_T, 1, device=device)
    r_T = torch.randn(M_T, D_BASE, device=device)
    x_0_T = MU_0 + SIGMA_0 * torch.randn(M_T, DIM, device=device)

    x_T = pushforward_net(t_T, x_0_T, r_T)
    f_at_T = test_funcs(x_T, t_T)
    E_T = f_at_T.mean(dim=0)

    # ========== Combined loss ==========
    residual = E_T - E_0 - E_interior
    loss = (residual ** 2).mean()

    return loss


# ============================================================================
# Validation and Visualization
# ============================================================================

def plot_distributions_at_times(pushforward_net, times, n_samples=10000):
    """Plot learned distributions at different times"""
    n_times = len(times)
    fig, axes = plt.subplots(1, n_times, figsize=(5 * n_times, 4))
    if n_times == 1:
        axes = [axes]

    with torch.no_grad():
        for idx, t_val in enumerate(times):
            t_batch = t_val * torch.ones(n_samples, 1, device=device)
            r_batch = torch.randn(n_samples, D_BASE, device=device)
            x_0_batch = MU_0 + SIGMA_0 * torch.randn(n_samples, DIM, device=device)

            x_samples = pushforward_net(t_batch, x_0_batch, r_batch).cpu().numpy()

            # Plot histogram
            axes[idx].hist(x_samples.flatten(), bins=60, density=True, alpha=0.7,
                           label='Learned', color='skyblue', edgecolor='black')

            # Plot steady-state for comparison (at final time)
            if t_val == times[-1]:
                x_analytical = np.linspace(x_samples.min(), x_samples.max(), 1000)
                pdf_analytical = analytical_steady_state_pdf(x_analytical, sigma=SIGMA)
                axes[idx].plot(x_analytical, pdf_analytical, 'r-', linewidth=2,
                               label='Steady State')

            axes[idx].set_xlabel('x', fontsize=11)
            axes[idx].set_ylabel('Probability Density', fontsize=11)
            axes[idx].set_title(f't = {t_val:.2f}', fontsize=12)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_doublewell1d/distributions.png', dpi=150, bbox_inches='tight')
    print("Saved: results_doublewell1d/distributions.png")
    plt.show()


def plot_time_evolution(pushforward_net, n_times=20, n_samples=5000):
    """Plot time evolution as a heatmap"""
    times = np.linspace(EPSILON, T, n_times)
    x_range = np.linspace(-2.5, 2.5, 100)

    density_grid = np.zeros((len(x_range), n_times))

    with torch.no_grad():
        for t_idx, t_val in enumerate(times):
            t_batch = t_val * torch.ones(n_samples, 1, device=device)
            r_batch = torch.randn(n_samples, D_BASE, device=device)
            x_0_batch = MU_0 + SIGMA_0 * torch.randn(n_samples, DIM, device=device)

            x_samples = pushforward_net(t_batch, x_0_batch, r_batch).cpu().numpy().flatten()

            # Compute histogram density
            hist, _ = np.histogram(x_samples, bins=x_range, density=True)
            density_grid[:len(hist), t_idx] = hist

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    extent = [times[0], times[-1], x_range[0], x_range[-1]]
    im = ax.imshow(density_grid, aspect='auto', origin='lower', extent=extent,
                   cmap='viridis', interpolation='bilinear')

    # Mark the potential minima
    ax.axhline(y=-1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Potential minima')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('x', fontsize=12)
    ax.set_title('Probability Density Evolution (Double Well)', fontsize=14)
    plt.colorbar(im, ax=ax, label='Density')
    ax.legend()

    plt.tight_layout()
    plt.savefig('results_doublewell1d/time_evolution.png', dpi=150, bbox_inches='tight')
    print("Saved: results_doublewell1d/time_evolution.png")
    plt.show()


def plot_mean_variance(pushforward_net, n_samples=5000):
    """Plot mean and variance evolution over time"""
    times = np.linspace(EPSILON, T, 50)
    means = []
    variances = []

    with torch.no_grad():
        for t_val in times:
            t_batch = t_val * torch.ones(n_samples, 1, device=device)
            r_batch = torch.randn(n_samples, D_BASE, device=device)
            x_0_batch = MU_0 + SIGMA_0 * torch.randn(n_samples, DIM, device=device)

            x_samples = pushforward_net(t_batch, x_0_batch, r_batch).cpu().numpy()
            means.append(x_samples.mean())
            variances.append(x_samples.var())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Mean evolution
    ax1.plot(times, means, 'b-', linewidth=2, label='Learned Mean')
    ax1.axhline(y=0, color='r', linestyle='--', label='Equilibrium (0)')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Mean', fontsize=12)
    ax1.set_title('Mean Evolution', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Variance evolution
    ax2.plot(times, variances, 'g-', linewidth=2, label='Learned Variance')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Variance', fontsize=12)
    ax2.set_title('Variance Evolution', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_doublewell1d/mean_variance.png', dpi=150, bbox_inches='tight')
    print("Saved: results_doublewell1d/mean_variance.png")
    plt.show()


def plot_potential_and_drift():
    """Visualize the double well potential and drift"""
    x = np.linspace(-2, 2, 1000)
    V = potential(x)
    b = drift_function(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Potential
    ax1.plot(x, V, 'b-', linewidth=2)
    ax1.axvline(x=-1, color='r', linestyle='--', alpha=0.5, label='Minima at ±1')
    ax1.axvline(x=1, color='r', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='g', linestyle=':', alpha=0.5, label='Maximum at 0')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('V(x)', fontsize=12)
    ax1.set_title('Double Well Potential V(x) = (x² - 1)²', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drift
    ax2.plot(x, b, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=-1, color='r', linestyle='--', alpha=0.5, label='Stable points')
    ax2.axvline(x=1, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='orange', linestyle=':', alpha=0.5, label='Unstable point')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('b(x)', fontsize=12)
    ax2.set_title('Drift b(x) = -4x³ + 4x', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_doublewell1d/potential_drift.png', dpi=150, bbox_inches='tight')
    print("Saved: results_doublewell1d/potential_drift.png")
    plt.show()


# ============================================================================
# Training Loop
# ============================================================================

print("=" * 70)
print("Time-Dependent 1D Double Well Fokker-Planck")
print("=" * 70)
print(f"Parameters: σ={SIGMA}, T={T}")
print(f"Potential: V(x) = (x² - 1)²")
print(f"Drift: b(x) = -4x³ + 4x")
print(f"Initial condition: N({MU_0}, {SIGMA_0 ** 2})")
print(f"Training: K={K} test functions, M={M} batch size, {N_EPOCHS} epochs")
print(f"Architecture: ResNet with residual blocks")
print("=" * 70)

# Plot the potential and drift
plot_potential_and_drift()

# Initialize networks
print("\nInitializing networks...")
pushforward_net = PushforwardNetwork(D_BASE, DIM).to(device)
test_funcs = TestFunctions(DIM, K).to(device)

print(f"Pushforward network parameters: {sum(p.numel() for p in pushforward_net.parameters()):,}")
print(f"Test functions parameters: {sum(p.numel() for p in test_funcs.parameters()):,}")

# Optimizers
gen_optimizer = optim.Adam(pushforward_net.parameters(), lr=LR_GEN)
test_optimizer = optim.SGD(test_funcs.parameters(), lr=LR_TEST)

loss_log = []
epoch_times = []

print("\nStarting training...")
print("-" * 70)

training_start = pytime.time()

for epoch in range(N_EPOCHS):
    epoch_start = pytime.time()

    # Update test functions (adversary - maximize loss)
    if epoch % 10 == 0 and epoch > 0:
        for _ in range(1):
            test_loss = compute_loss(pushforward_net, test_funcs)
            test_optimizer.zero_grad()
            (-test_loss).backward()
            test_optimizer.step()

    # Update pushforward network (minimize loss)
    gen_loss = compute_loss(pushforward_net, test_funcs)
    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()

    epoch_time = pytime.time() - epoch_start
    epoch_times.append(epoch_time)
    loss_log.append(gen_loss.item())

    if epoch % 500 == 0 or epoch == N_EPOCHS - 1:
        with torch.no_grad():
            t_test = 0.5 * torch.ones(1000, 1, device=device)
            r_test = torch.randn(1000, D_BASE, device=device)
            x_0_test = MU_0 + SIGMA_0 * torch.randn(1000, DIM, device=device)
            x_test = pushforward_net(t_test, x_0_test, r_test)
            mean_test = x_test.mean(dim=0)
            std_test = x_test.std(dim=0)

        avg_time = np.mean(epoch_times[-500:]) if len(epoch_times) >= 500 else np.mean(epoch_times)
        eta = avg_time * (N_EPOCHS - epoch - 1)

        print(f"Epoch {epoch:5d}/{N_EPOCHS}, Loss: {gen_loss.item():.6f}, "
              f"t=0.5: mean={mean_test.item():.4f}, std={std_test.item():.4f}, "
              f"Time/epoch: {avg_time:.3f}s, ETA: {eta / 60:.1f}min")

training_time = pytime.time() - training_start
print(f"\nTraining completed in {training_time / 60:.2f} minutes!")

# ============================================================================
# Final Validation and Visualization
# ============================================================================

print("\n" + "=" * 70)
print("Final Validation and Visualization")
print("=" * 70)

# Plot loss convergence
plt.figure(figsize=(10, 6))
plt.semilogy(loss_log)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Convergence (1D Double Well - ResNet)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results_doublewell1d/loss.png', dpi=150, bbox_inches='tight')
print("\nSaved: results_doublewell1d/loss.png")
plt.show()

# Plot distributions at different times
plot_distributions_at_times(pushforward_net, times=[0.01, 0.5, 1.0, 2.0])

# Plot time evolution
plot_time_evolution(pushforward_net)

# Plot mean and variance evolution
plot_mean_variance(pushforward_net)

# ============================================================================
# Save Results
# ============================================================================

# Save trained model
torch.save({
    'epoch': N_EPOCHS,
    'pushforward_net_state_dict': pushforward_net.state_dict(),
    'test_funcs_state_dict': test_funcs.state_dict(),
    'gen_optimizer_state_dict': gen_optimizer.state_dict(),
    'test_optimizer_state_dict': test_optimizer.state_dict(),
    'loss_log': loss_log,
    'training_time': training_time,
    'hyperparameters': {
        'DIM': DIM,
        'SIGMA': SIGMA,
        'MU_0': MU_0,
        'SIGMA_0': SIGMA_0,
        'T': T,
        'EPSILON': EPSILON,
        'K': K,
        'M': M,
        'M_0': M_0,
        'M_T': M_T,
        'D_BASE': D_BASE,
        'N_EPOCHS': N_EPOCHS,
    }
}, 'results_doublewell1d/checkpoint.pth')
print("\nSaved: results_doublewell1d/checkpoint.pth")

# Print final statistics
print("\n" + "=" * 70)
print("Summary Statistics:")
print("=" * 70)
print(f"Total training time: {training_time / 60:.2f} minutes")
print(f"Average time per epoch: {np.mean(epoch_times):.3f} seconds")
print(f"Final loss: {loss_log[-1]:.6e}")

with torch.no_grad():
    t_final = T * torch.ones(10000, 1, device=device)
    r_final = torch.randn(10000, D_BASE, device=device)
    x_0_final = MU_0 + SIGMA_0 * torch.randn(10000, DIM, device=device)
    x_final = pushforward_net(t_final, x_0_final, r_final).cpu().numpy()
    print(f"Final distribution (t={T}): mean={x_final.mean():.4f}, std={x_final.std():.4f}")

print("\n" + "=" * 70)
print("All done! Check the 'results_doublewell1d' directory for outputs.")
print("=" * 70)