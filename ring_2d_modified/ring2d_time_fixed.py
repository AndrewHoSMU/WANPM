"""
Time-Dependent Ring Potential with Rotational Drift - CORRECTED VERSION
========================================================================

Solves the time-dependent Fokker-Planck equation:
    ∂ρ/∂t - (σ²/2)Δρ + ∇·(b(x)ρ) = 0

With:
- Ring potential V(x,y) = (r² - r₀²)² / 4 with r₀ = 2.0
- Rotational drift: b(x,y) = -∇V + ω(-y, x)
- Initial condition: Gaussian at (0, 1.2) with σ_ic = 0.4
- Time horizon: T = 0.5
- Fixed noise: σ = 1.0 (constant throughout training)

Key fixes:
1. Network now takes x_0 as input (CRITICAL!)
2. Larger EPSILON for numerical stability
3. Better test function initialization
4. Deeper network architecture
5. Adjusted learning rates
6. More interior samples
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# PARAMETERS
# ============================================================================

# Problem parameters
DIM = 2
SIGMA = 1.0  # Fixed noise level
T_FINAL = 0.5  # Time horizon
EPSILON = 0.01  # Small offset from t=0 (INCREASED from 1e-3)

# Ring potential parameters
R0 = 2.0  # Ring radius
OMEGA = 2.0  # Rotation strength

# Initial condition - Gaussian
MU_IC = torch.tensor([[0.0, 1.2]], device='cuda' if torch.cuda.is_available() else 'cpu')
SIGMA_IC = 0.4

# Neural network training
K = 300  # Number of test functions
M = 10000  # Interior samples (INCREASED from 5000)
M_0 = 2000  # Samples at t=0
M_T = 2000  # Samples at t=T
D_BASE = 8  # Base distribution dimension
N_EPOCHS = 30000  # Total training epochs
LR_GEN = 5e-4  # Generator learning rate (DECREASED)
LR_TEST = 5e-3  # Test function learning rate (INCREASED)

# Learning rate decay
USE_LR_DECAY = True
LR_DECAY_START_EPOCH = 10000
LR_DECAY_FACTOR = 0.995

# Stabilization
USE_GRADIENT_CLIPPING = True
GRAD_CLIP_VALUE = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
MU_IC = MU_IC.to(device)

os.makedirs('results_ring2d_time_fixed', exist_ok=True)


# ============================================================================
# RING POTENTIAL AND DRIFT
# ============================================================================

def potential(x):
    """Ring potential: V(x,y) = (r² - r₀²)² / 4"""
    r_squared = (x ** 2).sum(dim=1, keepdim=True)
    return ((r_squared - R0 ** 2) ** 2) / 4


def drift_function(x):
    """
    Total drift: b = -∇V + ω(-y, x)

    Radial component (from ∇V):
        ∂V/∂x = 2x(r² - r₀²)
        ∂V/∂y = 2y(r² - r₀²)

    Tangential component (rotation):
        b_tangent = ω(-y, x)

    Total:
        b_x = -2x(r² - r₀²) - ωy
        b_y = -2y(r² - r₀²) + ωx
    """
    x1, x2 = x[:, 0:1], x[:, 1:2]
    r_squared = x1 ** 2 + x2 ** 2
    r_sq_minus_r0_sq = r_squared - R0 ** 2

    # Radial component: -∇V
    b_radial_x = -2 * x1 * r_sq_minus_r0_sq
    b_radial_y = -2 * x2 * r_sq_minus_r0_sq

    # Tangential component: ω(-y, x)
    b_tangent_x = -OMEGA * x2
    b_tangent_y = OMEGA * x1

    # Total drift
    b_x = b_radial_x + b_tangent_x
    b_y = b_radial_y + b_tangent_y

    return torch.cat([b_x, b_y], dim=1)


def sample_initial_condition(n_samples):
    """Sample from Gaussian initial condition at (0, 1.2) with σ = 0.4"""
    samples = MU_IC + SIGMA_IC * torch.randn(n_samples, DIM, device=device)
    return samples


def initial_condition_pdf(x):
    """PDF of Gaussian initial condition"""
    diff = x - MU_IC.cpu().numpy()
    exponent = -0.5 * np.sum(diff ** 2, axis=1) / (SIGMA_IC ** 2)
    normalization = 1 / ((2 * np.pi * SIGMA_IC ** 2) ** (DIM / 2))
    return normalization * np.exp(exponent)


# ============================================================================
# NEURAL NETWORK COMPONENTS - CORRECTED!
# ============================================================================

class PushforwardNetwork(nn.Module):
    """
    Neural network that maps (t, x_0, r) → x_t
    where x_0 ~ initial condition, r ~ N(0, I)
    
    CRITICAL FIX: Network now takes x_0 as input!
    """

    def __init__(self, d_base, d_output, hidden_dims=[256, 256, 256, 256]):
        super().__init__()
        layers = []
        # Input: time (1) + initial position (d_output) + base sample (d_base)
        input_dim = 1 + d_output + d_base  # FIXED: was 1 + d_base
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, d_output))
        self.network = nn.Sequential(*layers)

    def forward(self, t, x_0, r):
        """
        Args:
            t: time values (batch_size, 1)
            x_0: initial samples (batch_size, DIM)
            r: base distribution samples (batch_size, D_BASE)
        Returns:
            x_t: pushed samples at time t (batch_size, DIM)
        """
        # CRITICAL FIX: Concatenate ALL inputs including x_0
        t_x0_r = torch.cat([t, x_0, r], dim=1)
        delta = self.network(t_x0_r)
        # At t=0, should return x_0; at t>0, allows deviation scaled by sqrt(t)
        return x_0 + torch.sqrt(t + 1e-8) * delta


class TestFunctions(nn.Module):
    """Test functions: plane waves with time dependence"""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_in, n_out) * 0.5)
        # FIXED: Larger initial kappa for better time variation
        self.kappa = nn.Parameter(torch.randn(n_out) * 2.0)  # was 0.1
        self.b = nn.Parameter(torch.rand(n_out) * 2 * np.pi)

    def forward(self, x, t):
        """Evaluate test function: sin(w·x + κt + b)"""
        return torch.sin(x @ self.w + self.kappa * t + self.b)

    def time_derivative(self, x, t):
        """∂f/∂t = κ cos(w·x + κt + b)"""
        return self.kappa * torch.cos(x @ self.w + self.kappa * t + self.b)

    def laplacian(self, x, t):
        """Δf = -||w||² sin(w·x + κt + b)"""
        w_squared = (self.w ** 2).sum(dim=0, keepdim=True)
        return -w_squared * torch.sin(x @ self.w + self.kappa * t + self.b)

    def gradient_dot_drift(self, x, t):
        """∇f · b = (w · b) cos(w·x + κt + b)"""
        drift = drift_function(x)
        return (drift @ self.w) * torch.cos(x @ self.w + self.kappa * t + self.b)


# ============================================================================
# LOSS COMPUTATION
# ============================================================================

def compute_loss(pushforward_net, test_funcs):
    """
    Compute weak form loss for time-dependent FP equation:

    ∫_0^T ∫ [∂f/∂t + (σ²/2)Δf + ∇f·b] ρ dx dt = ∫ f(x,T)ρ(x,T) dx - ∫ f(x,0)ρ₀(x) dx

    where ρ is the solution we're learning via the pushforward.
    """
    # Interior integral: ∫_ε^T ∫ [∂f/∂t + (σ²/2)Δf + ∇f·b] ρ dx dt
    t_interior = EPSILON + (T_FINAL - EPSILON) * torch.rand(M, 1, device=device)
    r_interior = torch.randn(M, D_BASE, device=device)
    x_0_interior = sample_initial_condition(M)
    x_interior = pushforward_net(t_interior, x_0_interior, r_interior)

    df_dt = test_funcs.time_derivative(x_interior, t_interior)
    laplacian_term = (SIGMA ** 2 / 2) * test_funcs.laplacian(x_interior, t_interior)
    drift_term = test_funcs.gradient_dot_drift(x_interior, t_interior)

    interior_integrand = df_dt + laplacian_term + drift_term
    E_interior = (T_FINAL - EPSILON) * interior_integrand.mean(dim=0)

    # Boundary at t=0: ∫ f(x,0) ρ₀(x) dx
    x_0_samples = sample_initial_condition(M_0)
    t_0 = torch.zeros(M_0, 1, device=device)
    f_at_0 = test_funcs(x_0_samples, t_0)
    E_0 = f_at_0.mean(dim=0)

    # Boundary at t=T: ∫ f(x,T) ρ(x,T) dx
    t_T = T_FINAL * torch.ones(M_T, 1, device=device)
    r_T = torch.randn(M_T, D_BASE, device=device)
    x_0_T = sample_initial_condition(M_T)
    x_T = pushforward_net(t_T, x_0_T, r_T)
    f_at_T = test_funcs(x_T, t_T)
    E_T = f_at_T.mean(dim=0)

    # Weak form residual
    residual = E_T - E_0 - E_interior
    loss = (residual ** 2).mean()

    return loss


# ============================================================================
# TRAINING WITH DIAGNOSTICS
# ============================================================================

def train_neural_network():
    """Train with fixed σ = 1.0 and enhanced diagnostics"""
    pushforward_net = PushforwardNetwork(D_BASE, DIM).to(device)
    test_funcs = TestFunctions(DIM, K).to(device)

    optimizer_gen = optim.Adam(pushforward_net.parameters(), lr=LR_GEN)
    optimizer_test = optim.Adam(test_funcs.parameters(), lr=LR_TEST)

    loss_log = []
    ic_error_log = []
    lr_log = []

    print(f"\nTraining for {N_EPOCHS} epochs...")
    print(f"Fixed noise: σ = {SIGMA}")
    print(f"Network architecture: deeper (4 layers, 256 units each)")
    print(f"Learning rates: LR_gen={LR_GEN}, LR_test={LR_TEST}")
    if USE_LR_DECAY:
        print(f"LR decay: factor={LR_DECAY_FACTOR}, starts at epoch {LR_DECAY_START_EPOCH}")

    for epoch in range(N_EPOCHS):
        # Apply learning rate decay
        if USE_LR_DECAY and epoch >= LR_DECAY_START_EPOCH:
            lr_decay = LR_DECAY_FACTOR ** (epoch - LR_DECAY_START_EPOCH)
            current_lr_gen = LR_GEN * lr_decay
            current_lr_test = LR_TEST * lr_decay
            
            for param_group in optimizer_gen.param_groups:
                param_group['lr'] = current_lr_gen
            for param_group in optimizer_test.param_groups:
                param_group['lr'] = current_lr_test
        else:
            current_lr_gen = LR_GEN
            current_lr_test = LR_TEST
        
        lr_log.append(current_lr_gen)
        
        # Train test functions (maximize loss)
        optimizer_test.zero_grad()
        loss = compute_loss(pushforward_net, test_funcs)
        (-loss).backward()
        if USE_GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_(test_funcs.parameters(), GRAD_CLIP_VALUE)
        optimizer_test.step()

        # Train generator (minimize loss)
        optimizer_gen.zero_grad()
        loss = compute_loss(pushforward_net, test_funcs)
        loss.backward()
        if USE_GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_(pushforward_net.parameters(), GRAD_CLIP_VALUE)
        optimizer_gen.step()

        loss_log.append(loss.item())

        if epoch % 1000 == 0 or epoch == N_EPOCHS - 1:
            # Enhanced diagnostics
            with torch.no_grad():
                # Check IC preservation at t=0
                t_0_test = torch.zeros(1000, 1, device=device)
                r_0_test = torch.randn(1000, D_BASE, device=device)
                x_0_test = sample_initial_condition(1000)
                x_at_0 = pushforward_net(t_0_test, x_0_test, r_0_test)
                ic_error = (x_at_0 - x_0_test).abs().mean().item()
                ic_error_log.append(ic_error)
                
                # Sample at final time
                t_test = T_FINAL * torch.ones(1000, 1, device=device)
                r_test = torch.randn(1000, D_BASE, device=device)
                x_0_test_final = sample_initial_condition(1000)
                x_test = pushforward_net(t_test, x_0_test_final, r_test).cpu().numpy()
                mean_x = x_test.mean(axis=0)
                r_test_vals = np.sqrt((x_test ** 2).sum(axis=1))
                mean_r = r_test_vals.mean()
                
                # Test function statistics
                t_sample = torch.rand(100, 1, device=device) * T_FINAL
                x_sample = sample_initial_condition(100)
                f_vals = test_funcs(x_sample, t_sample)
                kappa_range = [test_funcs.kappa.min().item(), test_funcs.kappa.max().item()]

            print(f"Epoch {epoch:5d} | Loss: {loss.item():.6e} | IC error: {ic_error:.6e} | LR: {current_lr_gen:.6e}")
            print(f"  At t=T: mean=({mean_x[0]:.3f},{mean_x[1]:.3f}), r={mean_r:.3f}")
            print(f"  Test κ range: [{kappa_range[0]:.2f}, {kappa_range[1]:.2f}]")

    return pushforward_net, loss_log, ic_error_log, lr_log


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_samples_at_times(pushforward_net, time_points, n_samples=5000):
    """Generate samples at specified time points"""
    samples_dict = {}

    with torch.no_grad():
        for t_val in time_points:
            t = t_val * torch.ones(n_samples, 1, device=device)
            r = torch.randn(n_samples, D_BASE, device=device)
            x_0 = sample_initial_condition(n_samples)
            x_t = pushforward_net(t, x_0, r).cpu().numpy()
            samples_dict[t_val] = x_t

    return samples_dict


def plot_time_evolution(samples_dict, time_points):
    """Plot samples at different time points as scatter plots"""
    n_times = len(time_points)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Circle for reference
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = R0 * np.cos(theta)
    circle_y = R0 * np.sin(theta)

    for idx, t_val in enumerate(time_points):
        ax = axes[idx]
        samples = samples_dict[t_val]

        # Scatter plot
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1, c='blue')

        # Reference circle
        ax.plot(circle_x, circle_y, 'r--', linewidth=2, label=f'Target ring r={R0}')

        # Initial condition location
        if t_val == 0:
            ax.plot(MU_IC[0, 0].cpu(), MU_IC[0, 1].cpu(), 'go', markersize=15,
                    label=f'IC center', markeredgecolor='black', markeredgewidth=2)

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f't = {t_val:.1f}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim([-3.5, 3.5])
        ax.set_ylim([-3.5, 3.5])

        # Add statistics
        mean_x = samples.mean(axis=0)
        r_vals = np.sqrt((samples ** 2).sum(axis=1))
        mean_r = r_vals.mean()
        std_r = r_vals.std()

        textstr = f'Mean: ({mean_x[0]:.2f}, {mean_x[1]:.2f})\n' \
                  f'Mean r: {mean_r:.2f} ± {std_r:.2f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('results_ring2d_time_fixed/time_evolution_scatter.png', dpi=150, bbox_inches='tight')
    print("Saved: results_ring2d_time_fixed/time_evolution_scatter.png")
    plt.show()


def plot_training_diagnostics(loss_log, ic_error_log, lr_log):
    """Plot training loss, IC error, and learning rate"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].semilogy(loss_log, linewidth=1)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    if USE_LR_DECAY:
        axes[0].axvline(LR_DECAY_START_EPOCH, color='r', linestyle='--', 
                       alpha=0.5, label=f'LR decay starts')
        axes[0].legend()

    # IC error - ensure matching lengths
    epochs_ic = np.arange(0, len(ic_error_log) * 1000, 1000)
    axes[1].semilogy(epochs_ic, ic_error_log, linewidth=2, marker='o', markersize=3)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('IC Error (|x(t=0) - x_0|)', fontsize=12)
    axes[1].set_title('Initial Condition Preservation', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    axes[2].semilogy(lr_log, linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate (Generator)', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    if USE_LR_DECAY:
        axes[2].axvline(LR_DECAY_START_EPOCH, color='r', linestyle='--', 
                       alpha=0.5, label=f'Decay starts')
        axes[2].legend()

    plt.tight_layout()
    plt.savefig('results_ring2d_time_fixed/training_diagnostics.png', dpi=150, bbox_inches='tight')
    print("Saved: results_ring2d_time_fixed/training_diagnostics.png")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Time-Dependent Ring Potential - CORRECTED VERSION")
    print("=" * 70)
    print(f"\nProblem Setup:")
    print(f"  Ring radius: r₀ = {R0}")
    print(f"  Rotation: ω = {OMEGA}")
    print(f"  Time horizon: T = {T_FINAL}")
    print(f"  Initial condition: Gaussian at ({MU_IC[0, 0].item():.1f}, {MU_IC[0, 1].item():.1f}), σ_ic = {SIGMA_IC}")
    print(f"  Noise level: σ = {SIGMA}")
    print(f"\nKey fixes applied:")
    print(f"  1. Network now takes x_0 as input")
    print(f"  2. EPSILON increased to {EPSILON}")
    print(f"  3. Deeper network (4 layers, 256 units)")
    print(f"  4. Better test function initialization (κ ~ 2.0)")
    print(f"  5. More interior samples (M = {M})")

    # Train
    print("\n" + "=" * 70)
    print("Training Neural Network")
    print("=" * 70)
    pushforward_net, loss_log, ic_error_log, lr_log = train_neural_network()

    # Generate samples at time points
    print("\n" + "=" * 70)
    print("Generating Samples at Time Points")
    print("=" * 70)
    time_points = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    samples_dict = generate_samples_at_times(pushforward_net, time_points)

    # Plot time evolution
    print("\n" + "=" * 70)
    print("Plotting Time Evolution")
    print("=" * 70)
    plot_time_evolution(samples_dict, time_points)

    # Plot training diagnostics
    print("\n" + "=" * 70)
    print("Plotting Training Diagnostics")
    print("=" * 70)
    plot_training_diagnostics(loss_log, ic_error_log, lr_log)

    # Save results
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)
    torch.save({
        'pushforward_net_state_dict': pushforward_net.state_dict(),
        'loss_log': loss_log,
        'ic_error_log': ic_error_log,
        'lr_log': lr_log,
        'samples_dict': {t: samples for t, samples in samples_dict.items()},
        'hyperparameters': {
            'DIM': DIM,
            'R0': R0,
            'OMEGA': OMEGA,
            'SIGMA': SIGMA,
            'T_FINAL': T_FINAL,
            'MU_IC': MU_IC.cpu().numpy(),
            'SIGMA_IC': SIGMA_IC,
            'K': K,
            'M': M,
            'D_BASE': D_BASE,
            'N_EPOCHS': N_EPOCHS,
            'USE_LR_DECAY': USE_LR_DECAY,
            'LR_DECAY_START_EPOCH': LR_DECAY_START_EPOCH,
            'LR_DECAY_FACTOR': LR_DECAY_FACTOR,
        }
    }, 'results_ring2d_time_fixed/checkpoint.pth')

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Final loss: {loss_log[-1]:.6e}")
    print(f"Minimum loss: {min(loss_log):.6e} at epoch {np.argmin(loss_log)}")
    print(f"Final IC error: {ic_error_log[-1]:.6e}")

    # Final time statistics
    final_samples = samples_dict[T_FINAL]
    mean_final = final_samples.mean(axis=0)
    r_final = np.sqrt((final_samples ** 2).sum(axis=1))
    print(f"\nAt t={T_FINAL}:")
    print(f"  Mean position: ({mean_final[0]:.3f}, {mean_final[1]:.3f})")
    print(f"  Mean radius: {r_final.mean():.3f} ± {r_final.std():.3f}")
    print(f"  Target radius: {R0}")

    print("\n" + "=" * 70)
    print("✓ Complete! Results saved to 'results_ring2d_time_fixed/'")
    print("=" * 70)


if __name__ == '__main__':
    main()