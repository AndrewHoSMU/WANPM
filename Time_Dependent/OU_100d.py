"""
Time-Dependent 100D Ornstein-Uhlenbeck Process - Weak Adversarial Neural Pushforward Method
Solves the time-dependent Fokker-Planck equation in 100 dimensions with Gaussian initial condition
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
# Problem Parameters: 100D Ornstein-Uhlenbeck Process
# ============================================================================
# dX_t = -θ(X_t - μ)dt + σ dW_t
# ∂ρ/∂t = θ∇·((x-μ)ρ) + (σ²/2)Δρ

DIM = 100  # Spatial dimension
THETA = 1.0  # Mean reversion rate
MU = torch.zeros(DIM)  # Equilibrium mean
SIGMA = 1.0  # Noise strength
T = 1.0  # Final time
EPSILON = 1e-3  # Small time to avoid t=0 singularity

# Initial condition: Gaussian far from equilibrium
# Create diverse initial means
np.random.seed(42)
mu_0_values = np.random.randn(DIM) * 2.0  # Random means in [-4, 4] roughly
MU_0 = torch.tensor(mu_0_values, dtype=torch.float32)
SIGMA_0 = 0.5  # Initial std dev (isotropic)

# Training parameters
K = 5000  # Number of test functions (increased for higher dimension)
M = 10000  # Batch size (interior)
M_0 = 2000  # Batch size (t=0)
M_T = 2000  # Batch size (t=T)
D_BASE = 50  # Base distribution dimension
N_EPOCHS = 20000
LR_GEN = 1e-3
LR_TEST = 1e-2

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Create output directory
os.makedirs('results_100d', exist_ok=True)


# ============================================================================
# Analytical Solution
# ============================================================================

def analytical_mean(t):
    """Mean: m(t) = μ_0·e^(-θt) + μ(1 - e^(-θt))"""
    return MU_0.numpy() * np.exp(-THETA * t) + MU.numpy() * (1 - np.exp(-THETA * t))


def analytical_variance(t):
    """Variance: v(t) = σ_0²·e^(-2θt) + (σ²/2θ)(1 - e^(-2θt))
    Returns scalar (isotropic case)"""
    return SIGMA_0 ** 2 * np.exp(-2 * THETA * t) + (SIGMA ** 2 / (2 * THETA)) * (1 - np.exp(-2 * THETA * t))


# ============================================================================
# Neural Networks
# ============================================================================

class PushforwardNetwork(nn.Module):
    """F_θ(t, x_0, r) = x_0 + √t * F̃_θ(t, r)"""

    def __init__(self, d_base, d_output, hidden_dims=[128, 128, 128]):
        super().__init__()
        layers = []
        input_dim = 1 + d_base

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, d_output))
        self.network = nn.Sequential(*layers)

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
        delta = self.network(t_r)
        return x_0 + torch.sqrt(t) * delta


class TestFunctions(nn.Module):
    """f^(k)(x,t) = sin(∑w_i^(k)x_i + κ^(k)t + b^(k))"""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_in, n_out) * 0.1)  # Smaller init for stability
        self.kappa = nn.Parameter(torch.randn(n_out))  # Temporal frequency
        self.b = nn.Parameter(torch.rand(n_out) * 2 * np.pi)

    def forward(self, x, t):
        """Evaluate test functions"""
        return torch.sin(x @ self.w + self.kappa * t + self.b)

    def time_derivative(self, x, t):
        """∂f/∂t = κ cos(w·x + κt + b)"""
        return self.kappa * torch.cos(x @ self.w + self.kappa * t + self.b)

    def laplacian(self, x, t):
        """Δf = -||w||² sin(w·x + κt + b)"""
        w_squared = (self.w ** 2).sum(dim=0, keepdim=True)
        return -w_squared * torch.sin(x @ self.w + self.kappa * t + self.b)

    def gradient_dot_drift(self, x, t):
        """b(x)·∇f where b(x) = -θ(x - μ)
        ∇f = w cos(w·x + κt + b)
        b·∇f = -θ(x-μ)·w cos(w·x + κt + b)
        """
        drift = -THETA * (x - MU.to(x.device))
        return (drift @ self.w) * torch.cos(x @ self.w + self.kappa * t + self.b)


# ============================================================================
# Loss Function
# ============================================================================

def compute_loss(pushforward_net, test_funcs):
    """Compute the three-term weak form loss"""

    # ========== Interior term: ∫_ε^T 𝔼[∂f/∂t + Lf] dt ==========
    t_interior = EPSILON + (T - EPSILON) * torch.rand(M, 1, device=device)
    r_interior = torch.randn(M, D_BASE, device=device)
    x_0_interior = MU_0.to(device) + SIGMA_0 * torch.randn(M, DIM, device=device)

    x_interior = pushforward_net(t_interior, x_0_interior, r_interior)

    # ∂f/∂t
    df_dt = test_funcs.time_derivative(x_interior, t_interior)

    # Lf = (σ²/2)Δf + b·∇f
    laplacian_term = (SIGMA ** 2 / 2) * test_funcs.laplacian(x_interior, t_interior)
    drift_term = test_funcs.gradient_dot_drift(x_interior, t_interior)

    interior_integrand = df_dt + laplacian_term + drift_term
    E_interior = (T - EPSILON) * interior_integrand.mean(dim=0)

    # ========== Initial condition term: -𝔼_{ρ_0}[f(0,·)] ==========
    x_0_samples = MU_0.to(device) + SIGMA_0 * torch.randn(M_0, DIM, device=device)
    t_0 = torch.zeros(M_0, 1, device=device)

    f_at_0 = test_funcs(x_0_samples, t_0)
    E_0 = f_at_0.mean(dim=0)

    # ========== Terminal term: 𝔼_{ρ(T,·)}[f(T,·)] ==========
    t_T = T * torch.ones(M_T, 1, device=device)
    r_T = torch.randn(M_T, D_BASE, device=device)
    x_0_T = MU_0.to(device) + SIGMA_0 * torch.randn(M_T, DIM, device=device)

    x_T = pushforward_net(t_T, x_0_T, r_T)
    f_at_T = test_funcs(x_T, t_T)
    E_T = f_at_T.mean(dim=0)

    # ========== Combined loss ==========
    residual = E_T - E_0 - E_interior
    loss = (residual ** 2).mean()

    return loss


# ============================================================================
# Validation Functions
# ============================================================================

def validate_statistics(pushforward_net, times, n_samples=5000):
    """Validate mean and variance at different times"""
    print("\n" + "=" * 70)
    print("Validation: Mean and Variance Statistics")
    print("=" * 70)
    print(f"{'Time':>8} {'Mean Error':>15} {'Var Error':>15} "
          f"{'Mean L2':>15} {'Var L2':>15}")
    print("-" * 70)

    results = {'times': [], 'means_learned': [], 'means_true': [],
               'vars_learned': [], 'vars_true': [],
               'mean_errors': [], 'var_errors': []}

    with torch.no_grad():
        for t_val in times:
            t_batch = t_val * torch.ones(n_samples, 1, device=device)
            r_batch = torch.randn(n_samples, D_BASE, device=device)
            x_0_batch = MU_0.to(device) + SIGMA_0 * torch.randn(n_samples, DIM, device=device)

            x_samples = pushforward_net(t_batch, x_0_batch, r_batch).cpu().numpy()

            mean_learned = x_samples.mean(axis=0)
            var_learned = x_samples.var(axis=0)

            mean_true = analytical_mean(t_val)
            var_true = analytical_variance(t_val)

            # Compute errors
            mean_error = np.abs(mean_learned - mean_true).mean()
            var_error = np.abs(var_learned - var_true).mean()
            mean_l2 = np.sqrt(np.sum((mean_learned - mean_true) ** 2))
            var_l2 = np.sqrt(np.sum((var_learned - var_true) ** 2))

            results['times'].append(t_val)
            results['means_learned'].append(mean_learned)
            results['means_true'].append(mean_true)
            results['vars_learned'].append(var_learned)
            results['vars_true'].append(var_true)
            results['mean_errors'].append(mean_error)
            results['var_errors'].append(var_error)

            print(f"{t_val:8.3f} {mean_error:15.6f} {var_error:15.6f} "
                  f"{mean_l2:15.6f} {var_l2:15.6f}")

    return results


def plot_sample_dimensions(pushforward_net, times, dims_to_plot=[0, 1, 2], n_samples=10000):
    """Plot distributions at different times for selected dimensions"""
    n_times = len(times)
    n_dims = len(dims_to_plot)
    fig, axes = plt.subplots(n_dims, n_times, figsize=(5 * n_times, 4 * n_dims))

    if n_times == 1:
        axes = axes.reshape(-1, 1)
    if n_dims == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for time_idx, t_val in enumerate(times):
            t_batch = t_val * torch.ones(n_samples, 1, device=device)
            r_batch = torch.randn(n_samples, D_BASE, device=device)
            x_0_batch = MU_0.to(device) + SIGMA_0 * torch.randn(n_samples, DIM, device=device)

            x_samples = pushforward_net(t_batch, x_0_batch, r_batch).cpu().numpy()

            for dim_idx, dim in enumerate(dims_to_plot):
                # Learned distribution
                axes[dim_idx, time_idx].hist(x_samples[:, dim], bins=50, density=True, alpha=0.7,
                                             label='Learned', color='skyblue', edgecolor='black')

                # Analytical distribution
                m_true = analytical_mean(t_val)[dim]
                v_true = analytical_variance(t_val)

                x_range = np.linspace(x_samples[:, dim].min(), x_samples[:, dim].max(), 1000)
                pdf_true = norm.pdf(x_range, loc=m_true, scale=np.sqrt(v_true))
                axes[dim_idx, time_idx].plot(x_range, pdf_true, 'r-', lw=2,
                                             label=f'True N({m_true:.2f}, {v_true:.2f})')

                axes[dim_idx, time_idx].set_xlabel(f'x_{dim + 1}')
                if time_idx == 0:
                    axes[dim_idx, time_idx].set_ylabel('Probability Density')
                axes[dim_idx, time_idx].set_title(f'Dim {dim + 1}, t = {t_val:.3f}')
                axes[dim_idx, time_idx].legend(fontsize=8)
                axes[dim_idx, time_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_100d/distributions.png', dpi=150, bbox_inches='tight')
    print("Saved: results_100d/distributions.png")
    plt.show()


def plot_mean_variance_evolution(pushforward_net, n_time_points=100, dims_to_plot=[0, 1, 2]):
    """Plot evolution of mean with ±σ shaded area for selected dimensions"""
    times = np.linspace(EPSILON, T, n_time_points)

    means_learned_list = []
    stds_learned_list = []

    print("\nComputing evolution over time...")
    with torch.no_grad():
        for t_val in times:
            t_batch = t_val * torch.ones(3000, 1, device=device)
            r_batch = torch.randn(3000, D_BASE, device=device)
            x_0_batch = MU_0.to(device) + SIGMA_0 * torch.randn(3000, DIM, device=device)

            x_samples = pushforward_net(t_batch, x_0_batch, r_batch).cpu().numpy()
            means_learned_list.append(x_samples.mean(axis=0))
            stds_learned_list.append(x_samples.std(axis=0))

    means_learned = np.array(means_learned_list)
    stds_learned = np.array(stds_learned_list)

    means_true = np.array([analytical_mean(t) for t in times])
    stds_true = np.array([np.sqrt(analytical_variance(t)) for t in times])

    # Plot for selected dimensions
    n_dims = len(dims_to_plot)
    fig, axes = plt.subplots(1, n_dims, figsize=(6 * n_dims, 5))

    if n_dims == 1:
        axes = [axes]

    for dim_idx, dim in enumerate(dims_to_plot):
        ax = axes[dim_idx]
        
        # Learned: mean with ±σ shaded area
        ax.plot(times, means_learned[:, dim], 'b-', lw=2, label='Learned Mean', zorder=3)
        ax.fill_between(times, 
                        means_learned[:, dim] - stds_learned[:, dim],
                        means_learned[:, dim] + stds_learned[:, dim],
                        alpha=0.3, color='blue', label='Learned ±σ', zorder=1)
        
        # True: mean with ±σ shaded area
        ax.plot(times, means_true[:, dim], 'r--', lw=2, label='True Mean', zorder=2)
        ax.fill_between(times,
                        means_true[:, dim] - stds_true,
                        means_true[:, dim] + stds_true,
                        alpha=0.2, color='red', label='True ±σ', zorder=0)
        
        # Equilibrium line
        ax.axhline(y=MU[dim].item(), color='g', linestyle=':', lw=1.5,
                  label=f'Equilibrium ({MU[dim].item()})')
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f'Evolution - Dimension {dim + 1}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_100d/evolution.png', dpi=150, bbox_inches='tight')
    print("Saved: results_100d/evolution.png")
    plt.show()


def plot_error_heatmap(validation_results):
    """Plot heatmap of mean errors across dimensions and times"""
    times = validation_results['times']
    means_learned = np.array(validation_results['means_learned'])
    means_true = np.array(validation_results['means_true'])

    errors = np.abs(means_learned - means_true)  # Shape: (n_times, DIM)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Heatmap of errors
    im1 = ax1.imshow(errors.T, aspect='auto', cmap='viridis',
                     extent=[times[0], times[-1], 0, DIM])
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Dimension')
    ax1.set_title('Mean Absolute Error Across Dimensions')
    plt.colorbar(im1, ax=ax1, label='|Mean_learned - Mean_true|')

    # Average error per dimension
    avg_error_per_dim = errors.mean(axis=0)
    ax2.bar(range(DIM), avg_error_per_dim, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Average Absolute Error')
    ax2.set_title('Time-Averaged Mean Error by Dimension')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results_100d/error_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved: results_100d/error_heatmap.png")
    plt.show()


# ============================================================================
# Training Loop
# ============================================================================

print("=" * 70)
print("Time-Dependent 100D Ornstein-Uhlenbeck Process")
print("=" * 70)
print(f"Parameters: σ={SIGMA}, θ={THETA}, μ=0 (all dims), T={T}")
print(f"Initial condition: N(μ₀, {SIGMA_0 ** 2}·I)")
print(f"  μ₀[0:5] = {MU_0.numpy()[:5]}")
print(f"Equilibrium: N(0, {SIGMA ** 2 / (2 * THETA)}·I)")
print(f"Training: K={K} test functions, M={M} batch size, {N_EPOCHS} epochs")
print("=" * 70)

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
        for _ in range(3):
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
            x_0_test = MU_0.to(device) + SIGMA_0 * torch.randn(1000, DIM, device=device)
            x_test = pushforward_net(t_test, x_0_test, r_test)
            mean_test = x_test.mean(dim=0)
            std_test = x_test.std(dim=0)

        avg_time = np.mean(epoch_times[-500:]) if len(epoch_times) >= 500 else np.mean(epoch_times)
        eta = avg_time * (N_EPOCHS - epoch - 1)

        print(f"Epoch {epoch:5d}/{N_EPOCHS}, Loss: {gen_loss.item():.6f}, "
              f"t=0.5: mean[0]={mean_test[0].item():.4f}, std_avg={std_test.mean().item():.4f}, "
              f"Time/epoch: {avg_time:.2f}s, ETA: {eta / 60:.1f}min")

    # Intermediate validation
    if epoch % 5000 == 0 and epoch > 0:
        validate_statistics(pushforward_net, times=[0.01, 0.5, 1.0])

training_time = pytime.time() - training_start
print(f"\nTraining completed in {training_time / 60:.2f} minutes!")

# ============================================================================
# Final Validation and Visualization
# ============================================================================

print("\n" + "=" * 70)
print("Final Validation")
print("=" * 70)

validation_results = validate_statistics(pushforward_net,
                                         times=[0.001, 0.1, 0.2, 0.5, 0.8, 1.0])

# Plot loss convergence
plt.figure(figsize=(12, 6))
plt.semilogy(loss_log)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Convergence (100D Time-Dependent OU)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results_100d/loss.png', dpi=150, bbox_inches='tight')
print("\nSaved: results_100d/loss.png")
plt.show()

# Plot distributions at different times (dimensions 0, 1, 2)
plot_sample_dimensions(pushforward_net, times=[0.01, 0.2, 0.5, 1.0], dims_to_plot=[0, 1, 2])

# Plot mean and variance evolution (dimensions 0, 1, 2)
plot_mean_variance_evolution(pushforward_net, dims_to_plot=[0, 1, 2])

# Plot error heatmap
plot_error_heatmap(validation_results)

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
        'THETA': THETA,
        'MU': MU.numpy(),
        'SIGMA': SIGMA,
        'MU_0': MU_0.numpy(),
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
}, 'results_100d/checkpoint.pth')
print("Saved: results_100d/checkpoint.pth")

# Save validation results
np.savez('results_100d/validation_results.npz',
         times=validation_results['times'],
         means_learned=np.array(validation_results['means_learned']),
         means_true=np.array(validation_results['means_true']),
         vars_learned=np.array(validation_results['vars_learned']),
         vars_true=np.array(validation_results['vars_true']),
         mean_errors=np.array(validation_results['mean_errors']),
         var_errors=np.array(validation_results['var_errors']))
print("Saved: results_100d/validation_results.npz")

print("\n" + "=" * 70)
print("Summary Statistics:")
print("=" * 70)
print(f"Total training time: {training_time / 60:.2f} minutes")
print(f"Average time per epoch: {np.mean(epoch_times):.2f} seconds")
print(f"Final loss: {loss_log[-1]:.6e}")
print(f"Mean absolute error (t=1.0): {validation_results['mean_errors'][-1]:.6f}")
print(f"Variance error (t=1.0): {validation_results['var_errors'][-1]:.6f}")
print("\n" + "=" * 70)
print("All done! Check the 'results_100d' directory for outputs.")
print("=" * 70)