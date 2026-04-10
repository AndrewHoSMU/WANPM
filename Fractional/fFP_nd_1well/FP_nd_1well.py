"""
N-Dimensional Time-Dependent Fractional Fokker-Planck Equation Solver
Weak Adversarial Neural Pushforward Method for time-dependent fFPE with single well potential

Problem: ∂ρ/∂t + (−∆)^(α/2) ρ + ∇·(bρ) = 0
where b(x) = -∇V(x) for potential V(x) = 0.5 * ||x||^2 (harmonic well)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Problem parameters
    dim = 5  # Spatial dimension (adjustable - try 2, 5, 10, 20, 50, 100)
    alpha = 1.5  # Fractional order: α ∈ (0, 2]
    potential_strength = 1.0  # Coefficient for V(x) = 0.5 * k * ||x||^2

    # Time parameters
    T_final = 1.0  # Final time

    # Initial condition (Gaussian) - off-center from potential well
    initial_mean = 3.0  # Mean of initial Gaussian (scalar, broadcast to all dims)
    # Starting away from equilibrium at x=0 to observe relaxation
    initial_std = 0.5  # Std of initial Gaussian (tight initial distribution)

    # Network architecture
    base_dim = dim  # Dimension of base distribution
    hidden_dims = [128, 128, 128, 128]  # Hidden layer sizes
    activation = 'tanh'  # 'tanh', 'silu', or 'relu'

    # Training parameters
    num_epochs = 500
    batch_size_interior = 2000  # For interior term
    batch_size_initial = 1000  # For initial condition term
    batch_size_terminal = 1000  # For terminal term

    lr_generator = 1e-3
    lr_adversary = 1e-2
    adversary_steps = 1  # Adversary updates per generator update
    adversary_update_freq = 1  # Update adversary every N epochs

    # Test functions
    num_test_functions = 2000  # K in paper

    # Regularization and scheduling
    grad_clip = 1.0
    lr_decay_start = 5000
    lr_decay_rate = 0.95
    lr_decay_steps = 1000

    # Logging
    print_interval = 500
    plot_interval = 5000

    # Validation
    num_validation_samples = 5000
    validation_times = [0.0, 0.25, 0.5, 0.75, 1.0]  # Times to visualize


config = Config()


# ============================================================================
# PUSHFORWARD NETWORK (TIME-DEPENDENT)
# ============================================================================

class TimeDependentPushforwardNetwork(nn.Module):
    """
    Neural network F_θ(t, x0, r): R × R^d × R^{base_dim} → R^d

    Structure: F(t, x0, r) = x0 + √t * F̃_θ(t, x0, r)
    This ensures F(0, x0, r) = x0 (initial condition satisfied)
    """

    def __init__(self, dim, base_dim, hidden_dims, activation='tanh'):
        super().__init__()

        self.dim = dim
        self.base_dim = base_dim

        # Input: [t, x0 (dim), r (base_dim)]
        input_dim = 1 + dim + base_dim

        # Build network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, x0, r):
        """
        Args:
            t: (batch_size, 1) - time values
            x0: (batch_size, dim) - initial positions
            r: (batch_size, base_dim) - samples from base distribution
        Returns:
            x: (batch_size, dim) - positions at time t
        """
        batch_size = t.shape[0]

        # Concatenate inputs
        inputs = torch.cat([t, x0, r], dim=1)  # (batch_size, 1 + dim + base_dim)

        # Compute F̃_θ(t, x0, r)
        F_tilde = self.network(inputs)  # (batch_size, dim)

        # F(t, x0, r) = x0 + √t * F̃_θ(t, x0, r)
        # Add small epsilon to avoid division by zero at t=0
        sqrt_t = (t + 1e-8) ** (1.0 / Config.alpha)  # (batch_size, 1)
        x = x0 + sqrt_t * F_tilde

        return x


# ============================================================================
# TEST FUNCTIONS (TIME-DEPENDENT)
# ============================================================================

class TimeDependentPlaneWaveTestFunctions:
    """Manages plane-wave test functions f^(k)(t,x) = sin(w^(k)·x + κ^(k)t + b^(k))"""

    def __init__(self, num_functions, dim, device):
        self.K = num_functions
        self.dim = dim
        self.device = device

        # Initialize parameters: w^(k) ∈ R^d, κ^(k) ∈ R, b^(k) ∈ R
        self.w = nn.Parameter(torch.randn(num_functions, dim, device=device))
        self.kappa = nn.Parameter(torch.randn(num_functions, device=device))
        self.b = nn.Parameter(torch.rand(num_functions, device=device) * 2 * np.pi)

        # Normalize w to control frequency content
        with torch.no_grad():
            self.w.data = self.w.data / torch.norm(self.w.data, dim=1, keepdim=True)

    def compute_values_and_derivatives(self, t, x, alpha):
        """
        Compute f^(k)(t,x), ∂f^(k)/∂t, ∇f^(k), and (−∆)^(α/2) f^(k)

        Args:
            t: (batch_size, 1)
            x: (batch_size, dim)
            alpha: fractional order

        Returns:
            f: (batch_size, K) - test function values
            df_dt: (batch_size, K) - time derivatives
            grad_f: (batch_size, K, dim) - spatial gradients
            laplacian_f: (batch_size, K) - fractional Laplacian
        """
        batch_size = x.shape[0]

        # Compute w·x + κt + b for all test functions: (batch_size, K)
        wx = torch.matmul(x, self.w.t())  # (batch_size, K)
        kt = t * self.kappa.unsqueeze(0)  # (batch_size, K)
        argument = wx + kt + self.b.unsqueeze(0)  # (batch_size, K)

        # f^(k)(t,x) = sin(w·x + κt + b): (batch_size, K)
        f = torch.sin(argument)

        # ∂f^(k)/∂t = κ^(k) * cos(w·x + κt + b): (batch_size, K)
        cos_term = torch.cos(argument)  # (batch_size, K)
        df_dt = self.kappa.unsqueeze(0) * cos_term  # (batch_size, K)

        # ∇f^(k) = w^(k) * cos(w·x + κt + b): (batch_size, K, dim)
        grad_f = self.w.unsqueeze(0) * cos_term.unsqueeze(2)  # (batch_size, K, dim)

        # (−∆)^(α/2) f^(k) = |w^(k)|^α * f^(k)
        w_norm = torch.norm(self.w, dim=1)  # (K,)
        laplacian_f = (w_norm ** alpha).unsqueeze(0) * f  # (batch_size, K)

        return f, df_dt, grad_f, laplacian_f

    def parameters(self):
        return [self.w, self.kappa, self.b]


# ============================================================================
# POTENTIAL AND DRIFT
# ============================================================================

def drift_field(x, potential_strength):
    """
    Drift b(x) = -∇V(x) for harmonic potential V(x) = 0.5 * k * ||x||^2

    Args:
        x: (batch_size, dim)
        potential_strength: coefficient k

    Returns:
        b: (batch_size, dim) - drift field
    """
    return -potential_strength * x


# ============================================================================
# INITIAL CONDITION
# ============================================================================

def sample_initial_condition(batch_size, dim, mean, std, device):
    """
    Sample from initial Gaussian distribution

    Args:
        batch_size: number of samples
        dim: dimension
        mean: mean (scalar, broadcast to all dims)
        std: standard deviation (scalar, broadcast to all dims)
        device: torch device

    Returns:
        x0: (batch_size, dim) - initial samples
    """
    return mean + std * torch.randn(batch_size, dim, device=device)


# ============================================================================
# WEAK FORM RESIDUAL (TIME-DEPENDENT)
# ============================================================================

def compute_time_dependent_residual(generator, test_funcs, config):
    """
    Compute time-dependent weak form residual:
    R^(k) = E[f^(k)(T, x_T)] - E[f^(k)(0, x_0)] - ∫_0^T E[∂f/∂t + Lf] dt

    where Lf = -(−∆)^(α/2) f + b·∇f

    Args:
        generator: TimeDependentPushforwardNetwork
        test_funcs: TimeDependentPlaneWaveTestFunctions
        config: Config object

    Returns:
        residual: (K,) - residual for each test function
        loss: scalar - mean squared residual
    """

    # ========================================================================
    # TERM 1: Initial condition E[f^(k)(0, x_0)]
    # ========================================================================
    x0_initial = sample_initial_condition(
        config.batch_size_initial,
        config.dim,
        config.initial_mean,
        config.initial_std,
        device
    )

    t_initial = torch.zeros(config.batch_size_initial, 1, device=device)

    f_initial, _, _, _ = test_funcs.compute_values_and_derivatives(
        t_initial, x0_initial, config.alpha
    )

    E_initial = torch.mean(f_initial, dim=0)  # (K,)

    # ========================================================================
    # TERM 2: Terminal condition E[f^(k)(T, x_T)]
    # ========================================================================
    x0_terminal = sample_initial_condition(
        config.batch_size_terminal,
        config.dim,
        config.initial_mean,
        config.initial_std,
        device
    )

    r_terminal = torch.randn(config.batch_size_terminal, config.base_dim, device=device)
    t_terminal = torch.full((config.batch_size_terminal, 1), config.T_final, device=device)

    x_terminal = generator(t_terminal, x0_terminal, r_terminal)

    f_terminal, _, _, _ = test_funcs.compute_values_and_derivatives(
        t_terminal, x_terminal, config.alpha
    )

    E_terminal = torch.mean(f_terminal, dim=0)  # (K,)

    # ========================================================================
    # TERM 3: Interior integral ∫_0^T E[∂f/∂t + Lf] dt
    # ========================================================================
    # Sample time uniformly: t ~ U(0, T)
    t_interior = torch.rand(config.batch_size_interior, 1, device=device) * config.T_final

    # Sample initial positions
    x0_interior = sample_initial_condition(
        config.batch_size_interior,
        config.dim,
        config.initial_mean,
        config.initial_std,
        device
    )

    # Sample base distribution
    r_interior = torch.randn(config.batch_size_interior, config.base_dim, device=device)

    # Compute x(t) = F(t, x0, r)
    x_interior = generator(t_interior, x0_interior, r_interior)

    # Compute test function derivatives
    f, df_dt, grad_f, laplacian_f = test_funcs.compute_values_and_derivatives(
        t_interior, x_interior, config.alpha
    )

    # Compute drift field
    b = drift_field(x_interior, config.potential_strength)  # (batch_size, dim)

    # Compute b·∇f for all test functions
    # b: (batch_size, dim), grad_f: (batch_size, K, dim)
    b_dot_grad_f = torch.sum(b.unsqueeze(1) * grad_f, dim=2)  # (batch_size, K)

    # Lf = -(−∆)^(α/2) f + b·∇f
    Lf = -laplacian_f + b_dot_grad_f  # (batch_size, K)

    # ∂f/∂t + Lf
    integrand = df_dt + Lf  # (batch_size, K)

    # Monte Carlo approximation: ∫_0^T E[...] dt ≈ T * (1/M) Σ_m [...]
    E_interior = config.T_final * torch.mean(integrand, dim=0)  # (K,)

    # ========================================================================
    # RESIDUAL: R^(k) = E_terminal - E_initial - E_interior
    # ========================================================================
    residual = E_terminal - E_initial - E_interior  # (K,)

    # Loss: mean squared residual
    loss = torch.mean(residual ** 2)

    return residual, loss


# ============================================================================
# REFERENCE SOLUTION (α = 2 CASE ONLY)
# ============================================================================

def reference_gaussian_alpha2(t, x0_mean, x0_std, potential_strength):
    """
    Reference solution for α = 2 (classical case) only.
    For harmonic potential with α = 2, this is the exact Ornstein-Uhlenbeck solution.

    WARNING: This is ONLY valid for α = 2. For α < 2, NO analytical solution exists.
    For α < 2, the distribution has heavy tails (Lévy-stable-like) and is NOT Gaussian.

    Mean: μ(t) = μ_0 * exp(-kt)
    Variance: σ²(t) = σ²_eq + (σ²_0 - σ²_eq) * exp(-2kt)
    where σ²_eq = 1/k
    """
    # Time-dependent mean (decay toward 0)
    mean_t = x0_mean * np.exp(-potential_strength * t)

    # Time-dependent variance (approaches equilibrium)
    equilibrium_var = 1.0 / potential_strength
    var_0 = x0_std ** 2

    # Exponential relaxation toward equilibrium
    var_t = equilibrium_var + (var_0 - equilibrium_var) * np.exp(-2 * potential_strength * t)

    return mean_t, np.sqrt(var_t)


# ============================================================================
# PARTICLE SIMULATION (FOR VALIDATION)
# ============================================================================

def simulate_fractional_langevin(config, t_final, num_particles=5000, num_steps=200):
    """
    Direct simulation of the fractional Langevin equation using symmetric stable Lévy flights

    For α < 2: dX_t = -k X_t dt + dL^α_t
    where L^α is a symmetric α-stable Lévy process

    This uses Euler-Maruyama with α-stable increments

    Args:
        config: Config object
        t_final: final time to simulate to
        num_particles: number of particles to simulate
        num_steps: number of time steps

    Returns:
        particles: (num_particles, dim) array of particle positions at t_final
    """
    dt = t_final / num_steps
    dim = config.dim
    k = config.potential_strength
    alpha = config.alpha

    # Initialize particles from initial distribution
    particles = config.initial_mean + config.initial_std * np.random.randn(num_particles, dim)

    # Simulate forward in time
    for step in range(num_steps):
        # Drift term: -k * X_t * dt
        drift = -k * particles * dt

        # Diffusion term: increments of α-stable Lévy process
        # For α = 2, this reduces to Brownian motion
        # For α < 2, we use the Chambers-Mallows-Stuck method for symmetric stable

        if alpha == 2.0:
            # Brownian increments
            dL = np.sqrt(dt) * np.random.randn(num_particles, dim)
        else:
            # Symmetric α-stable increments
            # Using the property that for small dt: dL^α ~ dt^(1/α) * Z
            # where Z is α-stable random variable

            # Generate symmetric α-stable random variables
            # Chambers-Mallows-Stuck method
            U = np.random.uniform(-np.pi / 2, np.pi / 2, (num_particles, dim))
            W = np.random.exponential(1.0, (num_particles, dim))

            if alpha != 1.0:
                # α ≠ 1 case
                const = np.sin(alpha * U) / (np.cos(U) ** (1 / alpha))
                dL = const * (np.cos((1 - alpha) * U) / W) ** ((1 - alpha) / alpha)
                # Scale by dt^(1/α)
                dL = dt ** (1 / alpha) * dL
            else:
                # α = 1 (Cauchy case)
                dL = np.tan(U)
                dL = dt * dL  # dt^(1/1) = dt

        # Update particles
        particles = particles + drift + dL

    return particles


# ============================================================================
# TRAINING
# ============================================================================

def train():
    """Main training loop"""

    print("=" * 80)
    print(f"N-Dimensional Time-Dependent Fractional Fokker-Planck Equation Solver")
    print(f"Dimension: {config.dim}")
    print(f"Fractional order α: {config.alpha}")
    print(f"Time interval: [0, {config.T_final}]")
    print(f"Initial condition: Gaussian(μ={config.initial_mean}, σ={config.initial_std})")
    print("=" * 80)

    # Initialize networks
    generator = TimeDependentPushforwardNetwork(
        config.dim,
        config.base_dim,
        config.hidden_dims,
        config.activation
    ).to(device)

    test_funcs = TimeDependentPlaneWaveTestFunctions(
        config.num_test_functions,
        config.dim,
        device
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.lr_generator)
    optimizer_A = torch.optim.Adam(test_funcs.parameters(), lr=config.lr_adversary)

    # Learning rate schedulers
    scheduler_G = torch.optim.lr_scheduler.StepLR(
        optimizer_G,
        step_size=config.lr_decay_steps,
        gamma=config.lr_decay_rate
    )
    scheduler_A = torch.optim.lr_scheduler.StepLR(
        optimizer_A,
        step_size=config.lr_decay_steps,
        gamma=config.lr_decay_rate
    )

    # Training history
    history = {
        'loss': [],
        'residual_norm': [],
        'epoch': []
    }

    # Training loop
    for epoch in range(config.num_epochs):

        # Update generator
        optimizer_G.zero_grad()
        residual, loss = compute_time_dependent_residual(generator, test_funcs, config)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(generator.parameters(), config.grad_clip)

        optimizer_G.step()

        # Update adversary (test functions)
        if epoch % config.adversary_update_freq == 0:
            for _ in range(config.adversary_steps):
                optimizer_A.zero_grad()
                _, loss_adv = compute_time_dependent_residual(generator, test_funcs, config)
                (-loss_adv).backward()  # Ascent on loss
                optimizer_A.step()

        # Learning rate decay
        if epoch >= config.lr_decay_start:
            scheduler_G.step()
            scheduler_A.step()

        # Logging
        if epoch % config.print_interval == 0 or epoch == config.num_epochs - 1:
            residual_norm = torch.norm(residual).item()
            history['loss'].append(loss.item())
            history['residual_norm'].append(residual_norm)
            history['epoch'].append(epoch)

            print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | "
                  f"||Residual||: {residual_norm:.6f}")

        # Plotting
        if epoch % config.plot_interval == 0 and epoch > 0:
            plot_results(generator, config, epoch, history)

    print("\nTraining completed!")

    return generator, test_funcs, history


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(generator, config, epoch, history):
    """Plot marginal distributions at different times with particle simulation comparison"""

    generator.eval()

    # Number of marginals to plot
    num_marginals = min(config.dim, 4)
    num_times = len(config.validation_times)

    # Create figure
    fig = plt.figure(figsize=(5 * num_times, 4 * num_marginals + 6))

    # Generate samples at different times
    with torch.no_grad():
        for time_idx, t_val in enumerate(config.validation_times):
            # ================================================================
            # LEARNED SOLUTION
            # ================================================================
            # Sample initial conditions
            x0 = sample_initial_condition(
                config.num_validation_samples,
                config.dim,
                config.initial_mean,
                config.initial_std,
                device
            )

            # Sample base distribution
            r = torch.randn(config.num_validation_samples, config.base_dim, device=device)

            # Time tensor
            t = torch.full((config.num_validation_samples, 1), t_val, device=device)

            # Generate samples from learned solution
            x_learned = generator(t, x0, r).cpu().numpy()

            # ================================================================
            # PARTICLE SIMULATION (for comparison)
            # ================================================================
            if t_val > 0:  # Don't simulate for t=0 (just initial condition)
                x_particles = simulate_fractional_langevin(config, t_val,
                                                           num_particles=config.num_validation_samples,
                                                           num_steps=200)
            else:
                # At t=0, just use initial condition
                x_particles = config.initial_mean + config.initial_std * np.random.randn(
                    config.num_validation_samples, config.dim)

            # Reference solution (only valid for α = 2)
            if config.alpha == 2.0:
                mean_ref, std_ref = reference_gaussian_alpha2(
                    t_val, config.initial_mean, config.initial_std,
                    config.potential_strength
                )
                has_reference = True
            else:
                has_reference = False

            # Plot each marginal
            for dim_idx in range(num_marginals):
                ax = plt.subplot(num_marginals + 2, num_times,
                                 dim_idx * num_times + time_idx + 1)

                # Fixed x-axis range
                x_range = np.linspace(-5, 5, 200)

                # Histogram of learned distribution
                ax.hist(x_learned[:, dim_idx], bins=40, density=True,
                        alpha=0.5, label='Learned (Neural)', color='blue',
                        edgecolor='blue', linewidth=1.5, range=(-5, 5))

                # Histogram of particle simulation
                ax.hist(x_particles[:, dim_idx], bins=40, density=True,
                        alpha=0.5, label='Particle Simulation', color='green',
                        edgecolor='green', linewidth=1.5, range=(-5, 5))

                # Reference Gaussian (only for α = 2)
                if has_reference:
                    reference = np.exp(-(x_range - mean_ref) ** 2 /
                                       (2 * std_ref ** 2)) / (std_ref * np.sqrt(2 * np.pi))
                    ax.plot(x_range, reference, 'r--', linewidth=2,
                            label=f'Exact α=2', alpha=0.8)

                ax.set_xlabel(f'$x_{dim_idx + 1}$')
                ax.set_xlim(-5, 5)
                if time_idx == 0:
                    ax.set_ylabel('Density')
                ax.set_title(f'Dim {dim_idx + 1}, t={t_val:.2f}')
                ax.legend(fontsize=7, loc='best')
                ax.grid(True, alpha=0.3)

    generator.train()

    # Plot training metrics
    ax_loss = plt.subplot(num_marginals + 2, 2, 2 * num_marginals + 1)
    ax_loss.semilogy(history['epoch'], history['loss'], 'b-', linewidth=2)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training Loss')
    ax_loss.grid(True, alpha=0.3)

    ax_res = plt.subplot(num_marginals + 2, 2, 2 * num_marginals + 2)
    ax_res.semilogy(history['epoch'], history['residual_norm'], 'r-', linewidth=2)
    ax_res.set_xlabel('Epoch')
    ax_res.set_ylabel('||Residual||')
    ax_res.set_title('Weak Form Residual Norm')
    ax_res.grid(True, alpha=0.3)

    # Statistics panel
    ax_stats = plt.subplot(num_marginals + 2, 1, num_marginals + 2)
    ax_stats.axis('off')

    stats_text = f"""
    Statistics (Epoch {epoch}):
    
    Dimension: {config.dim}
    Fractional order α: {config.alpha}
    Time interval: [0, {config.T_final}]
    
    Initial: Gaussian(μ={config.initial_mean}, σ={config.initial_std})
    (Starting off-center to observe relaxation to x=0)
    
    Current loss: {history['loss'][-1]:.6e}
    Residual norm: {history['residual_norm'][-1]:.6e}
    """

    ax_stats.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                  verticalalignment='center')

    plt.tight_layout()

    # Save figure
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results/fFP_time_{config.dim}D_alpha{config.alpha}_epoch{epoch}_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    plt.close()


# ============================================================================
# FINAL VALIDATION
# ============================================================================

def validate_solution(generator, config):
    """Comprehensive validation of learned solution with particle simulation comparison"""

    print("\n" + "=" * 80)
    print("FINAL VALIDATION")
    print("=" * 80)

    generator.eval()

    for t_val in config.validation_times:
        print(f"\n--- Time t = {t_val:.2f} ---")

        # ====================================================================
        # LEARNED SOLUTION
        # ====================================================================
        with torch.no_grad():
            # Sample initial conditions
            x0 = sample_initial_condition(
                config.num_validation_samples,
                config.dim,
                config.initial_mean,
                config.initial_std,
                device
            )

            # Sample base distribution
            r = torch.randn(config.num_validation_samples, config.base_dim, device=device)

            # Time tensor
            t = torch.full((config.num_validation_samples, 1), t_val, device=device)

            # Generate samples
            x_learned = generator(t, x0, r).cpu().numpy()

        # ====================================================================
        # PARTICLE SIMULATION
        # ====================================================================
        if t_val > 0:
            print("Running particle simulation for comparison...")
            x_particles = simulate_fractional_langevin(config, t_val,
                                                       num_particles=config.num_validation_samples,
                                                       num_steps=200)
        else:
            x_particles = config.initial_mean + config.initial_std * np.random.randn(
                config.num_validation_samples, config.dim)

        # ====================================================================
        # COMPARE STATISTICS
        # ====================================================================
        # Learned statistics
        means_learned = np.mean(x_learned, axis=0)
        stds_learned = np.std(x_learned, axis=0)

        # Particle statistics
        means_particles = np.mean(x_particles, axis=0)
        stds_particles = np.std(x_particles, axis=0)

        print(f"\nMean per dimension:")
        print(f"  Learned:   {np.mean(means_learned):.6f} ± {np.std(means_learned):.6f}")
        print(f"  Particles: {np.mean(means_particles):.6f} ± {np.std(means_particles):.6f}")
        print(f"  Difference: {np.abs(np.mean(means_learned) - np.mean(means_particles)):.6f}")

        print(f"\nStd per dimension:")
        print(f"  Learned:   {np.mean(stds_learned):.6f} ± {np.std(stds_learned):.6f}")
        print(f"  Particles: {np.mean(stds_particles):.6f} ± {np.std(stds_particles):.6f}")
        print(f"  Difference: {np.abs(np.mean(stds_learned) - np.mean(stds_particles)):.6f}")

        # Reference values (only valid for α = 2)
        if config.alpha == 2.0:
            mean_ref, std_ref = reference_gaussian_alpha2(
                t_val, config.initial_mean, config.initial_std,
                config.potential_strength
            )

            print(f"\nReference (α=2 exact):")
            print(f"  Mean: {mean_ref:.6f}")
            print(f"  Std:  {std_ref:.6f}")
            print(f"  Learned vs Reference - Mean error: {np.abs(np.mean(means_learned) - mean_ref):.6f}")
            print(f"  Learned vs Reference - Std error:  {np.abs(np.mean(stds_learned) - std_ref):.6f}")
        else:
            print(f"\nNote: For α={config.alpha:.2f} < 2, no analytical solution exists")
            print(f"      Validation relies on particle simulation comparison")

    print("=" * 80)

    generator.train()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Train the model
    generator, test_funcs, history = train()

    # Final validation
    validate_solution(generator, config)

    # Create final comprehensive plot
    plot_results(generator, config, config.num_epochs, history)

    # Save model
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/fFP_time_{config.dim}D_alpha{config.alpha}_{timestamp}.pt'
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'test_funcs_w': test_funcs.w,
        'test_funcs_kappa': test_funcs.kappa,
        'test_funcs_b': test_funcs.b,
        'config': vars(config),
        'history': history
    }, model_path)
    print(f"\nModel saved: {model_path}")
