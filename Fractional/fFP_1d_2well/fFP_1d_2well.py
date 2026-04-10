import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import stats
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
ALPHA = 1.5  # Fractional Laplacian order (0 < alpha <= 2)
D = 1  # Spatial dimension
D_BASE = 8  # Base distribution dimension (can be > D for expressivity)

# Double-well potential parameters: V(x) = a*(x^2 - b^2)^2
# Two wells at x = ±b with barrier at x = 0
WELL_A = 1.0  # Coefficient (controls barrier height)
WELL_B = 1.0  # Well separation (wells located at ±b)

# Initial condition: Gaussian centered at midpoint (x=0) between the two wells
IC_CENTER = 0.0  # Midpoint between wells at x = -b and x = +b
IC_SIGMA = 0.3

# Time domain
T_FINAL = 0.8
N_TIME_STEPS = 50  # Number of time steps to visualize

# Training parameters
N_EPOCHS = 10000
M_INTERIOR = 3000  # Batch size for interior term
M_INITIAL = 1000  # Batch size for initial condition term
M_TERMINAL = 1000  # Batch size for terminal condition term
K_TEST = 3000  # Number of test functions

LR_GENERATOR = 1e-3
LR_ADVERSARY = 1e-2
ADV_UPDATE_FREQ = 1  # Update adversary every N epochs
ADV_STEPS_PER_UPDATE = 1

# Visualization
N_BINS = 100
X_RANGE = (-3, 3)

# Particle simulation parameters (for benchmark)
N_PARTICLES = 10000  # Number of particles for benchmark simulation
DT_PARTICLE = 0.01  # Time step for particle simulation


# ============================================================================
# POTENTIAL AND DRIFT
# ============================================================================
def double_well_potential(x):
    """Double-well potential: V(x) = a*(x^2 - b^2)^2

    Two minima at x = ±b, barrier at x = 0
    """
    # x shape: (batch, 1)
    return WELL_A * (x ** 2 - WELL_B ** 2) ** 2


def drift_field(t, x):
    """Drift b(t,x) = -dV/dx for double-well potential

    V(x) = a*(x^2 - b^2)^2
    dV/dx = 2*a*(x^2 - b^2)*2*x = 4*a*x*(x^2 - b^2)
    b(x) = -4*a*x*(x^2 - b^2)
    """
    return -4.0 * WELL_A * x * (x ** 2 - WELL_B ** 2)


# ============================================================================
# INITIAL CONDITION
# ============================================================================
def sample_initial_condition(n_samples):
    """Sample from initial Gaussian distribution centered at midpoint"""
    samples = torch.randn(n_samples, D, device=device) * IC_SIGMA + IC_CENTER
    return samples


# ============================================================================
# PARTICLE SIMULATION (BENCHMARK)
# ============================================================================
def particle_simulation_benchmark(time_points):
    """
    Simulate fractional Fokker-Planck equation using particle method.

    For the equation: ∂ρ/∂t + (−Δ)^(α/2) ρ + ∇·(bρ) = 0

    The corresponding SDE is: dx = b(x)dt + dL_α

    Two cases:
    - α = 2: Standard Brownian motion, dL = sqrt(2)*dW where W is Wiener process
    - α < 2: Symmetric α-stable Lévy process with scale dt^(1/α)

    Returns:
        trajectories: dict mapping time -> numpy array of particle positions
    """
    print("\n" + "=" * 80)
    print("Running Particle Simulation Benchmark")
    print("=" * 80)
    print(f"Number of particles: {N_PARTICLES}")
    print(f"Time step: {DT_PARTICLE}")
    print(f"Fractional order α: {ALPHA}")

    if abs(ALPHA - 2.0) < 1e-6:
        print(f"Using standard Brownian motion (α=2)")
        print(f"Diffusion: sqrt(2*dt) = {np.sqrt(2 * DT_PARTICLE):.6f}")
    else:
        print(f"Using α-stable Lévy process (α={ALPHA})")
        print(f"Scale parameter: dt^(1/α) = {DT_PARTICLE ** (1 / ALPHA):.6f}")

    print("-" * 80)

    # Initialize particles from initial condition
    particles = np.random.randn(N_PARTICLES) * IC_SIGMA + IC_CENTER

    # Store snapshots at requested times
    trajectories = {}
    trajectories[0.0] = particles.copy()

    # Time integration
    current_time = 0.0
    time_idx = 1
    next_snapshot_time = time_points[time_idx] if time_idx < len(time_points) else float('inf')

    n_steps = int(np.ceil(time_points[-1] / DT_PARTICLE))

    # Check if we're in the Brownian motion regime
    is_brownian = abs(ALPHA - 2.0) < 1e-6

    for step in range(n_steps):
        current_time = step * DT_PARTICLE

        # Drift term: b(x) = -4*a*x*(x^2 - b^2)
        drift = -4.0 * WELL_A * particles * (particles ** 2 - WELL_B ** 2)

        # Diffusion term
        if is_brownian:
            # Standard case: α = 2, use Gaussian increments
            # For (−Δ)ρ (Laplacian), the SDE is dx = b*dt + sqrt(2)*dW
            noise_increments = np.random.randn(N_PARTICLES) * np.sqrt(2 * DT_PARTICLE)
        else:
            # Fractional case: α < 2, use Lévy α-stable increments
            # For (−Δ)^(α/2) with coefficient 1, scale = dt^(1/α)
            noise_increments = stats.levy_stable.rvs(
                alpha=ALPHA,
                beta=0,  # Symmetric
                loc=0,
                scale=DT_PARTICLE ** (1.0 / ALPHA),
                size=N_PARTICLES
            )

        # Update particles: x_{n+1} = x_n + b(x_n)*dt + noise
        particles = particles + drift * DT_PARTICLE + noise_increments

        # Store snapshot if we've reached a requested time point
        if current_time >= next_snapshot_time - DT_PARTICLE / 2:
            trajectories[next_snapshot_time] = particles.copy()
            print(f"  Snapshot at t = {next_snapshot_time:.3f}, "
                  f"mean = {np.mean(particles):.4f}, std = {np.std(particles):.4f}")
            time_idx += 1
            next_snapshot_time = time_points[time_idx] if time_idx < len(time_points) else float('inf')

    print("-" * 80)
    print("Particle simulation complete!")
    print("=" * 80 + "\n")

    return trajectories


# ============================================================================
# NEURAL NETWORK: PUSHFORWARD MAP
# ============================================================================
class PushforwardNetwork(nn.Module):
    def __init__(self, d_base, d_out, hidden_dims=[128, 128, 128]):
        super().__init__()

        # Input: (t, x0, r) where r ~ P_base
        # t: scalar, x0: d_out dimensional, r: d_base dimensional
        input_dim = 1 + d_out + d_base

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, d_out))

        self.network = nn.Sequential(*layers)

    def forward(self, t, x0, r):
        """
        Args:
            t: (batch, 1) - time
            x0: (batch, d_out) - initial position
            r: (batch, d_base) - base distribution sample
        Returns:
            x: (batch, d_out) - pushforward sample at time t
        """
        # Concatenate inputs
        inputs = torch.cat([t, x0, r], dim=1)

        # Compute displacement
        displacement = self.network(inputs)

        # Ensure initial condition: x(0, x0, r) = x0
        x = x0 + (t + 1e-8) ** (1.0 / ALPHA) * displacement

        return x


# ============================================================================
# TEST FUNCTIONS
# ============================================================================
class PlaneWaveTestFunctions:
    def __init__(self, k_test, d, device):
        self.k_test = k_test
        self.d = d
        self.device = device

        # Initialize test function parameters
        self.w = nn.Parameter(torch.randn(k_test, d, device=device))
        self.kappa = nn.Parameter(torch.randn(k_test, 1, device=device))
        self.b = nn.Parameter(torch.rand(k_test, 1, device=device) * 2 * np.pi)

        # Normalize frequencies
        with torch.no_grad():
            self.w.data = self.w.data / torch.norm(self.w.data, dim=1, keepdim=True)

    def f(self, t, x):
        """Evaluate test function f^(k)(t,x) = sin(w^(k)·x + κ^(k)t + b^(k))"""
        # x: (batch, d), output: (batch, k_test)
        phase = torch.matmul(x, self.w.T) + self.kappa.T * t + self.b.T
        return torch.sin(phase)

    def df_dt(self, t, x):
        """Time derivative of test function"""
        phase = torch.matmul(x, self.w.T) + self.kappa.T * t + self.b.T
        return self.kappa.T * torch.cos(phase)

    def grad_f(self, t, x):
        """Spatial gradient of test function"""
        # Returns: (batch, k_test, d)
        phase = torch.matmul(x, self.w.T) + self.kappa.T * t + self.b.T
        cos_phase = torch.cos(phase)  # (batch, k_test)
        return cos_phase.unsqueeze(-1) * self.w.unsqueeze(0)  # (batch, k_test, d)

    def fractional_laplacian_f(self, t, x):
        """Fractional Laplacian: (-Δ)^(α/2) f = |w|^α f"""
        w_norm = torch.norm(self.w, dim=1)  # (k_test,)
        return (w_norm ** ALPHA).unsqueeze(0) * self.f(t, x)

    def parameters(self):
        return [self.w, self.kappa, self.b]


# ============================================================================
# LOSS COMPUTATION
# ============================================================================
def compute_loss(generator, test_functions, t_final):
    """Compute weak form residual loss"""

    # --- Initial term: E_{ρ0}[f(0, x0)] ---
    x0_initial = sample_initial_condition(M_INITIAL)
    t0 = torch.zeros(M_INITIAL, 1, device=device)
    f_initial = test_functions.f(t0, x0_initial)  # (M_INITIAL, K)
    E_initial = f_initial.mean(dim=0)  # (K,)

    # --- Terminal term: E_{ρ(T,·)}[f(T, x)] ---
    x0_terminal = sample_initial_condition(M_TERMINAL)
    r_terminal = torch.randn(M_TERMINAL, D_BASE, device=device)
    t_terminal = torch.full((M_TERMINAL, 1), t_final, device=device)
    x_terminal = generator(t_terminal, x0_terminal, r_terminal)
    f_terminal = test_functions.f(t_terminal, x_terminal)  # (M_TERMINAL, K)
    E_terminal = f_terminal.mean(dim=0)  # (K,)

    # --- Interior term: ∫_0^T E_{ρ(t,·)}[∂f/∂t + Lf] dt ---
    # Sample times uniformly
    t_samples = torch.rand(M_INTERIOR, 1, device=device) * t_final
    x0_interior = sample_initial_condition(M_INTERIOR)
    r_interior = torch.randn(M_INTERIOR, D_BASE, device=device)
    x_interior = generator(t_samples, x0_interior, r_interior)

    # Compute operator Lf = -(−Δ)^(α/2) f + b·∇f
    frac_laplacian_f = test_functions.fractional_laplacian_f(t_samples, x_interior)  # (M, K)
    grad_f = test_functions.grad_f(t_samples, x_interior)  # (M, K, D)
    b = drift_field(t_samples, x_interior)  # (M, D)
    b_dot_grad_f = (b.unsqueeze(1) * grad_f).sum(dim=2)  # (M, K)

    L_f = -frac_laplacian_f + b_dot_grad_f

    # Time derivative
    df_dt = test_functions.df_dt(t_samples, x_interior)  # (M, K)

    # Interior integral (Monte Carlo with importance sampling from U(0,T))
    interior_integrand = df_dt + L_f  # (M, K)
    E_interior = t_final * interior_integrand.mean(dim=0)  # (K,)

    # --- Residual: R^(k) = E_terminal - E_initial - E_interior ---
    residual = E_terminal - E_initial - E_interior  # (K,)

    # --- Loss: mean squared residual ---
    loss = (residual ** 2).mean()

    return loss, residual


# ============================================================================
# TRAINING LOOP
# ============================================================================
def train():
    # Initialize networks
    generator = PushforwardNetwork(D_BASE, D).to(device)
    test_functions = PlaneWaveTestFunctions(K_TEST, D, device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR_GENERATOR)
    optimizer_A = torch.optim.Adam(test_functions.parameters(), lr=LR_ADVERSARY)

    # Learning rate schedulers
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, N_EPOCHS)

    # Training history
    history = {'loss': [], 'residual_norm': []}

    print("Starting training...")
    print(f"Epochs: {N_EPOCHS}, K_test: {K_TEST}, M_interior: {M_INTERIOR}")
    print(f"Fractional order α: {ALPHA}")
    print(f"Double-well potential: V(x) = a*(x² - b²)², a={WELL_A}, b={WELL_B}")
    print(f"Wells at x = ±{WELL_B}, barrier at x = 0")
    print(f"Drift: b(x) = -4*a*x*(x² - b²)")
    print(f"Initial condition: center={IC_CENTER}, sigma={IC_SIGMA}")
    print("-" * 80)

    for epoch in range(N_EPOCHS):
        # --- Generator step (minimize loss) ---
        optimizer_G.zero_grad()
        loss, residual = compute_loss(generator, test_functions, T_FINAL)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer_G.step()
        scheduler_G.step()

        # --- Adversary step (maximize loss) ---
        if epoch % ADV_UPDATE_FREQ == 0:
            for _ in range(ADV_STEPS_PER_UPDATE):
                optimizer_A.zero_grad()
                loss_adv, _ = compute_loss(generator, test_functions, T_FINAL)
                (-loss_adv).backward()  # Negative for maximization
                torch.nn.utils.clip_grad_norm_(test_functions.parameters(), 1.0)
                optimizer_A.step()

        # Record history
        history['loss'].append(loss.item())
        history['residual_norm'].append(torch.norm(residual).item())

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{N_EPOCHS} | Loss: {loss.item():.6f} | "
                  f"Residual norm: {torch.norm(residual).item():.6f}")

    print("-" * 80)
    print("Training complete!")

    return generator, test_functions, history


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_training_history(history):
    """Plot training loss history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['residual_norm'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Residual Norm')
    axes[1].set_title('Weak Form Residual Norm')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_time_evolution(generator, particle_trajectories):
    """Create histograms at each time step comparing neural network and particle simulation"""
    time_steps = torch.linspace(0, T_FINAL, N_TIME_STEPS, device=device)

    # Create figure with subplots
    n_rows = 5
    n_cols = 10
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    axes = axes.flatten()

    print("\nGenerating time evolution comparison visualization...")

    for idx, t in enumerate(time_steps):
        if idx >= len(axes):
            break

        # Generate neural network samples at time t
        with torch.no_grad():
            x0 = sample_initial_condition(5000)
            r = torch.randn(5000, D_BASE, device=device)
            t_vec = torch.full((5000, 1), t.item(), device=device)
            x_samples = generator(t_vec, x0, r)
            x_nn = x_samples.cpu().numpy().flatten()

        # Get particle simulation data at nearest time
        t_particle = min(particle_trajectories.keys(), key=lambda x: abs(x - t.item()))
        x_particle = particle_trajectories[t_particle]

        # Plot histogram
        ax = axes[idx]
        ax.hist(x_nn, bins=N_BINS, range=X_RANGE, density=True,
                alpha=0.5, color='blue', edgecolor='blue', linewidth=0.5, label='Neural Network')
        ax.hist(x_particle, bins=N_BINS, range=X_RANGE, density=True,
                alpha=0.5, color='green', edgecolor='green', linewidth=0.5, label='Particle Sim')

        # Add potential well visualization (double-well)
        x_plot = np.linspace(X_RANGE[0], X_RANGE[1], 200)
        V_plot = WELL_A * (x_plot ** 2 - WELL_B ** 2) ** 2
        ax2 = ax.twinx()
        ax2.plot(x_plot, V_plot, 'r-', linewidth=1.5, alpha=0.3)
        ax2.set_ylim([0, np.max(V_plot) * 1.2])
        ax2.set_yticks([])

        ax.set_xlim(X_RANGE)
        ax.set_title(f't = {t.item():.3f}', fontsize=8)
        ax.set_xlabel('x', fontsize=7)
        ax.set_ylabel('ρ(t,x)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

        # Add legend only to first subplot
        if idx == 0:
            ax.legend(fontsize=6, loc='upper right')

    # Hide unused subplots
    for idx in range(len(time_steps), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Fractional Fokker-Planck Evolution (α={ALPHA}) - Neural Network vs Particle Simulation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/time_evolution_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print("Time evolution comparison visualization saved!")


def compute_comparison_metrics(generator, particle_trajectories):
    """Compute quantitative metrics comparing neural network and particle simulation"""
    time_steps = torch.linspace(0, T_FINAL, N_TIME_STEPS, device=device)

    metrics = {
        'time': [],
        'mean_nn': [],
        'mean_particle': [],
        'median_nn': [],
        'median_particle': [],
        'iqr_nn': [],
        'iqr_particle': [],
        'mad_nn': [],
        'mad_particle': [],
        'std_nn': [],
        'std_particle': [],
        'p90_nn': [],
        'p90_particle': [],
        'p10_nn': [],
        'p10_particle': [],
        'ks_statistic': [],
        'wasserstein_distance': []
    }

    print("\nComputing quantitative comparison metrics...")
    print("-" * 80)
    print(
        f"{'Time':>8} | {'Mean NN':>10} | {'Mean Part':>10} | {'IQR NN':>10} | {'IQR Part':>10} | {'KS stat':>10} | {'W-dist':>10}")
    print("-" * 80)

    for t in time_steps:
        # Generate neural network samples
        with torch.no_grad():
            x0 = sample_initial_condition(5000)
            r = torch.randn(5000, D_BASE, device=device)
            t_vec = torch.full((5000, 1), t.item(), device=device)
            x_samples = generator(t_vec, x0, r)
            x_nn = x_samples.cpu().numpy().flatten()

        # Get particle simulation data at nearest time
        t_particle = min(particle_trajectories.keys(), key=lambda x: abs(x - t.item()))
        x_particle = particle_trajectories[t_particle]

        # Compute robust statistics
        mean_nn = np.mean(x_nn)
        mean_particle = np.mean(x_particle)

        median_nn = np.median(x_nn)
        median_particle = np.median(x_particle)

        # Standard deviation (for comparison - problematic for heavy tails!)
        std_nn = np.std(x_nn)
        std_particle = np.std(x_particle)

        # Interquartile range (IQR)
        q75_nn, q25_nn = np.percentile(x_nn, [75, 25])
        q75_particle, q25_particle = np.percentile(x_particle, [75, 25])
        iqr_nn = q75_nn - q25_nn
        iqr_particle = q75_particle - q25_particle

        # Median Absolute Deviation (MAD)
        mad_nn = np.median(np.abs(x_nn - median_nn))
        mad_particle = np.median(np.abs(x_particle - median_particle))

        # 10th and 90th percentiles
        p10_nn, p90_nn = np.percentile(x_nn, [10, 90])
        p10_particle, p90_particle = np.percentile(x_particle, [10, 90])

        # Kolmogorov-Smirnov test
        ks_stat, _ = stats.ks_2samp(x_nn, x_particle)

        # Wasserstein distance (Earth Mover's Distance)
        w_dist = stats.wasserstein_distance(x_nn, x_particle)

        # Store metrics
        metrics['time'].append(t.item())
        metrics['mean_nn'].append(mean_nn)
        metrics['mean_particle'].append(mean_particle)
        metrics['median_nn'].append(median_nn)
        metrics['median_particle'].append(median_particle)
        metrics['iqr_nn'].append(iqr_nn)
        metrics['iqr_particle'].append(iqr_particle)
        metrics['mad_nn'].append(mad_nn)
        metrics['mad_particle'].append(mad_particle)
        metrics['std_nn'].append(std_nn)
        metrics['std_particle'].append(std_particle)
        metrics['p10_nn'].append(p10_nn)
        metrics['p10_particle'].append(p10_particle)
        metrics['p90_nn'].append(p90_nn)
        metrics['p90_particle'].append(p90_particle)
        metrics['ks_statistic'].append(ks_stat)
        metrics['wasserstein_distance'].append(w_dist)

        print(f"{t.item():8.3f} | {mean_nn:10.4f} | {mean_particle:10.4f} | "
              f"{iqr_nn:10.4f} | {iqr_particle:10.4f} | {ks_stat:10.4f} | {w_dist:10.4f}")

    print("-" * 80)

    # Plot comparison metrics
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # Mean comparison
    axes[0, 0].plot(metrics['time'], metrics['mean_nn'], 'b-', linewidth=2, label='Neural Network')
    axes[0, 0].plot(metrics['time'], metrics['mean_particle'], 'g--', linewidth=2, label='Particle Simulation')
    axes[0, 0].set_xlabel('Time', fontsize=12)
    axes[0, 0].set_ylabel('Mean', fontsize=12)
    axes[0, 0].set_title('Mean Evolution', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Median comparison
    axes[0, 1].plot(metrics['time'], metrics['median_nn'], 'b-', linewidth=2, label='Neural Network')
    axes[0, 1].plot(metrics['time'], metrics['median_particle'], 'g--', linewidth=2, label='Particle Simulation')
    axes[0, 1].set_xlabel('Time', fontsize=12)
    axes[0, 1].set_ylabel('Median', fontsize=12)
    axes[0, 1].set_title('Median Evolution (Robust Central Tendency)', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # IQR comparison
    axes[1, 0].plot(metrics['time'], metrics['iqr_nn'], 'b-', linewidth=2, label='Neural Network')
    axes[1, 0].plot(metrics['time'], metrics['iqr_particle'], 'g--', linewidth=2, label='Particle Simulation')
    axes[1, 0].set_xlabel('Time', fontsize=12)
    axes[1, 0].set_ylabel('Interquartile Range (IQR)', fontsize=12)
    axes[1, 0].set_title('IQR Evolution (Robust Spread)', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # MAD comparison
    axes[1, 1].plot(metrics['time'], metrics['mad_nn'], 'b-', linewidth=2, label='Neural Network')
    axes[1, 1].plot(metrics['time'], metrics['mad_particle'], 'g--', linewidth=2, label='Particle Simulation')
    axes[1, 1].set_xlabel('Time', fontsize=12)
    axes[1, 1].set_ylabel('Median Absolute Deviation (MAD)', fontsize=12)
    axes[1, 1].set_title('MAD Evolution (Robust Scale)', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Percentiles comparison (spread)
    axes[2, 0].fill_between(metrics['time'],
                            metrics['p10_nn'], metrics['p90_nn'],
                            alpha=0.3, color='blue', label='NN 10-90th percentile')
    axes[2, 0].fill_between(metrics['time'],
                            metrics['p10_particle'], metrics['p90_particle'],
                            alpha=0.3, color='green', label='Particle 10-90th percentile')
    axes[2, 0].plot(metrics['time'], metrics['median_nn'], 'b-', linewidth=2, label='NN Median')
    axes[2, 0].plot(metrics['time'], metrics['median_particle'], 'g--', linewidth=2, label='Particle Median')
    axes[2, 0].set_xlabel('Time', fontsize=12)
    axes[2, 0].set_ylabel('Position', fontsize=12)
    axes[2, 0].set_title('Percentile Bands (10th, 50th, 90th)', fontsize=13, fontweight='bold')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Standard deviation comparison (showing the problem with heavy tails!)
    axes[2, 1].plot(metrics['time'], metrics['std_nn'], 'b-', linewidth=2, label='Neural Network')
    axes[2, 1].plot(metrics['time'], metrics['std_particle'], 'g--', linewidth=2, label='Particle Simulation')
    axes[2, 1].set_xlabel('Time', fontsize=12)
    axes[2, 1].set_ylabel('Standard Deviation', fontsize=12)
    axes[2, 1].set_title('Standard Deviation Evolution\n(Unreliable for α<2 due to heavy tails)', fontsize=13,
                         fontweight='bold')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Robust Comparison Metrics: Neural Network vs Particle Simulation (α={ALPHA})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/quantitative_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Quantitative comparison plots saved!")

    return metrics


def plot_potential_and_initial_condition():
    """Plot the potential well and initial condition"""
    x = np.linspace(X_RANGE[0], X_RANGE[1], 500)
    V = WELL_A * (x ** 2 - WELL_B ** 2) ** 2

    # Initial condition PDF
    rho0 = (1 / (IC_SIGMA * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - IC_CENTER) / IC_SIGMA) ** 2)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot potential
    ax1.plot(x, V, 'r-', linewidth=2, label=f'Potential V(x) = a(x²-b²)²')
    ax1.axvline(-WELL_B, color='r', linestyle=':', alpha=0.5, label=f'Wells at x=±{WELL_B}')
    ax1.axvline(WELL_B, color='r', linestyle=':', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Barrier at x=0')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('V(x)', color='r', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_ylim([0, np.max(V) * 1.2])
    ax1.grid(True, alpha=0.3)

    # Plot initial condition
    ax2 = ax1.twinx()
    ax2.plot(x, rho0, 'b-', linewidth=2, label='Initial ρ₀(x)')
    ax2.set_ylabel('ρ₀(x)', color='b', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.fill_between(x, rho0, alpha=0.3, color='blue')

    plt.title(f'Double-Well Potential and Initial Condition\n' +
              f'α={ALPHA}, a={WELL_A}, b={WELL_B}, IC center={IC_CENTER}',
              fontsize=13, fontweight='bold')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig('outputs/potential_and_ic.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)

    print("=" * 80)
    print("Fractional Fokker-Planck Equation Solver")
    print("Weak Adversarial Neural Sampler Method")
    print("=" * 80)

    # Plot setup
    plot_potential_and_initial_condition()
    print("Potential and initial condition plotted.")

    # Run particle simulation benchmark BEFORE training
    time_points = np.linspace(0, T_FINAL, N_TIME_STEPS)
    particle_trajectories = particle_simulation_benchmark(time_points)

    # Train the model
    generator, test_functions, history = train()

    # Plot training history
    plot_training_history(history)
    print("Training history plotted.")

    # Visualize time evolution with comparison
    visualize_time_evolution(generator, particle_trajectories)

    # Compute quantitative comparison metrics
    metrics = compute_comparison_metrics(generator, particle_trajectories)

    # Save model and metrics
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'test_functions_w': test_functions.w.data,
        'test_functions_kappa': test_functions.kappa.data,
        'test_functions_b': test_functions.b.data,
        'hyperparameters': {
            'alpha': ALPHA,
            'd': D,
            'd_base': D_BASE,
            'well_a': WELL_A,
            'well_b': WELL_B,
            'ic_center': IC_CENTER,
            'ic_sigma': IC_SIGMA,
            't_final': T_FINAL,
            'k_test': K_TEST,
            'n_particles': N_PARTICLES,
            'dt_particle': DT_PARTICLE
        },
        'comparison_metrics': metrics
    }, 'outputs/fractional_fp_model.pt')

    print("\nModel and metrics saved to fractional_fp_model.pt")
    print("=" * 80)
    print("\nSUMMARY:")
    print("-" * 80)
    print(f"Average KS statistic: {np.mean(metrics['ks_statistic']):.4f}")
    print(f"Average Wasserstein distance: {np.mean(metrics['wasserstein_distance']):.4f}")
    print(f"Final mean error: {abs(metrics['mean_nn'][-1] - metrics['mean_particle'][-1]):.4f}")
    print(f"Final median error: {abs(metrics['median_nn'][-1] - metrics['median_particle'][-1]):.4f}")
    print(f"Final IQR error: {abs(metrics['iqr_nn'][-1] - metrics['iqr_particle'][-1]):.4f}")
    print(f"Final MAD error: {abs(metrics['mad_nn'][-1] - metrics['mad_particle'][-1]):.4f}")
    print(f"Final std error: {abs(metrics['std_nn'][-1] - metrics['std_particle'][-1]):.4f} (unreliable for α<2)")
    print("=" * 80)


if __name__ == "__main__":
    main()