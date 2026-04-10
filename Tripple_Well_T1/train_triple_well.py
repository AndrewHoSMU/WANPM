"""
Triple-Well Potential Training Script
======================================

Trains a neural network to solve the time-dependent Fokker-Planck equation:
    ∂ρ/∂t - (σ²/2)Δρ + ∇·(b(x)ρ) = 0

With:
- Triple-well potential V(x) with minima at x = -1, 0, +1
- Drift: b(x) = -dV/dx
- Initial condition: Mixture of two Gaussians at x = -0.5 and x = 0.5

Also runs finite volume solver for comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# PARAMETERS
# ============================================================================

# Problem parameters
DIM = 1
SIGMA_INITIAL = 5.0
SIGMA_FINAL = 0.5
T_FINAL = 2.5
EPSILON = 1e-3

# Initial condition - Mixture Gaussian at x = ±0.5
MU_1 = -0.5
MU_2 = 0.5
SIGMA_IC = 0.15
MIXTURE_WEIGHT = 0.5

# Neural network training
K = 4000
M = 4000
M_0 = 1000
M_T = 1000
D_BASE = 8
N_EPOCHS = 40000
LR_GEN_INITIAL = 1e-3
LR_TEST_INITIAL = 1e-2

# Curriculum schedule
ANNEALING_START_EPOCH = 2000
ANNEALING_END_EPOCH = 25000
FINETUNING_EPOCHS = 15000

# Stabilization options
USE_LR_DECAY = True
USE_GRADIENT_CLIPPING = True
GRAD_CLIP_VALUE = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs('results_triple_well', exist_ok=True)


# ============================================================================
# TRIPLE-WELL POTENTIAL
# ============================================================================

def potential(x):
    """Triple-well potential: V(x) = x²(x² - 1)²"""
    return (x**2) * ((x**2 - 1)**2)


def drift_function(x):
    """Drift: b(x) = -dV/dx = -2x(x² - 1)(3x² - 1)"""
    return -2 * x * (x**2 - 1) * (3*x**2 - 1)


def sample_mixture_gaussian(n_samples):
    """Sample from mixture of two Gaussians at x = ±0.5"""
    assignments = torch.rand(n_samples, 1, device=device) < MIXTURE_WEIGHT
    samples_1 = MU_1 + SIGMA_IC * torch.randn(n_samples, DIM, device=device)
    samples_2 = MU_2 + SIGMA_IC * torch.randn(n_samples, DIM, device=device)
    samples = torch.where(assignments, samples_1, samples_2)
    return samples


def mixture_gaussian_pdf(x):
    """PDF of the mixture Gaussian initial condition"""
    pdf_1 = (1 / (np.sqrt(2 * np.pi) * SIGMA_IC)) * np.exp(-0.5 * ((x - MU_1) / SIGMA_IC)**2)
    pdf_2 = (1 / (np.sqrt(2 * np.pi) * SIGMA_IC)) * np.exp(-0.5 * ((x - MU_2) / SIGMA_IC)**2)
    return MIXTURE_WEIGHT * pdf_1 + (1 - MIXTURE_WEIGHT) * pdf_2


# ============================================================================
# CURRICULUM SCHEDULES
# ============================================================================

def get_sigma_schedule(epoch):
    """Cosine annealing schedule for σ"""
    if epoch < ANNEALING_START_EPOCH:
        return SIGMA_INITIAL
    elif epoch < ANNEALING_END_EPOCH:
        progress = (epoch - ANNEALING_START_EPOCH) / (ANNEALING_END_EPOCH - ANNEALING_START_EPOCH)
        cosine_progress = 0.5 * (1 - np.cos(np.pi * progress))
        sigma = SIGMA_INITIAL + cosine_progress * (SIGMA_FINAL - SIGMA_INITIAL)
        return sigma
    else:
        return SIGMA_FINAL


def get_learning_rate_schedule(epoch, initial_lr):
    """Learning rate decay schedule"""
    if not USE_LR_DECAY:
        return initial_lr

    if epoch < ANNEALING_START_EPOCH:
        return initial_lr
    elif epoch < ANNEALING_END_EPOCH:
        decay_factor = 0.9999
        epochs_in_annealing = epoch - ANNEALING_START_EPOCH
        return initial_lr * (decay_factor ** epochs_in_annealing)
    else:
        decay_at_end = initial_lr * (0.9999 ** (ANNEALING_END_EPOCH - ANNEALING_START_EPOCH))
        epochs_after = epoch - ANNEALING_END_EPOCH
        slow_decay = 0.99995
        return decay_at_end * (slow_decay ** epochs_after)


# ============================================================================
# NEURAL NETWORK COMPONENTS
# ============================================================================

class PushforwardNetwork(nn.Module):
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
        t_r = torch.cat([t, r], dim=1)
        delta = self.network(t_r)
        return x_0 + torch.sqrt(t) * delta


class TestFunctions(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_in, n_out) * 0.5)
        self.kappa = nn.Parameter(torch.randn(n_out))
        self.b = nn.Parameter(torch.rand(n_out) * 2 * np.pi)

    def forward(self, x, t):
        return torch.sin(x @ self.w + self.kappa * t + self.b)

    def time_derivative(self, x, t):
        return self.kappa * torch.cos(x @ self.w + self.kappa * t + self.b)

    def laplacian(self, x, t):
        w_squared = (self.w ** 2).sum(dim=0, keepdim=True)
        return -w_squared * torch.sin(x @ self.w + self.kappa * t + self.b)

    def gradient_dot_drift(self, x, t):
        drift = drift_function(x)
        return (drift @ self.w) * torch.cos(x @ self.w + self.kappa * t + self.b)


# ============================================================================
# LOSS COMPUTATION
# ============================================================================

def compute_loss(pushforward_net, test_funcs, sigma):
    """Compute weak form loss with mixture Gaussian initial condition"""
    # Interior integral
    t_interior = EPSILON + (T_FINAL - EPSILON) * torch.rand(M, 1, device=device)
    r_interior = torch.randn(M, D_BASE, device=device)
    x_0_interior = sample_mixture_gaussian(M)
    x_interior = pushforward_net(t_interior, x_0_interior, r_interior)

    df_dt = test_funcs.time_derivative(x_interior, t_interior)
    laplacian_term = (sigma ** 2 / 2) * test_funcs.laplacian(x_interior, t_interior)
    drift_term = test_funcs.gradient_dot_drift(x_interior, t_interior)
    interior_integrand = df_dt + laplacian_term + drift_term
    E_interior = (T_FINAL - EPSILON) * interior_integrand.mean(dim=0)

    # Boundary at t=0 (mixture Gaussian)
    x_0_samples = sample_mixture_gaussian(M_0)
    t_0 = torch.zeros(M_0, 1, device=device)
    f_at_0 = test_funcs(x_0_samples, t_0)
    E_0 = f_at_0.mean(dim=0)

    # Boundary at t=T
    t_T = T_FINAL * torch.ones(M_T, 1, device=device)
    r_T = torch.randn(M_T, D_BASE, device=device)
    x_0_T = sample_mixture_gaussian(M_T)
    x_T = pushforward_net(t_T, x_0_T, r_T)
    f_at_T = test_funcs(x_T, t_T)
    E_T = f_at_T.mean(dim=0)

    residual = E_T - E_0 - E_interior
    loss = (residual ** 2).mean()
    return loss


# ============================================================================
# TRAINING
# ============================================================================

def train_neural_network():
    """Train with curriculum learning"""
    pushforward_net = PushforwardNetwork(D_BASE, DIM).to(device)
    test_funcs = TestFunctions(DIM, K).to(device)

    optimizer_gen = optim.Adam(pushforward_net.parameters(), lr=LR_GEN_INITIAL)
    optimizer_test = optim.Adam(test_funcs.parameters(), lr=LR_TEST_INITIAL)

    loss_log = []
    sigma_log = []
    lr_gen_log = []
    lr_test_log = []

    print(f"Training for {N_EPOCHS} epochs...")
    print(f"Warmup: epochs 0-{ANNEALING_START_EPOCH}")
    print(f"Annealing: epochs {ANNEALING_START_EPOCH}-{ANNEALING_END_EPOCH}")
    print(f"Fine-tuning: epochs {ANNEALING_END_EPOCH}-{N_EPOCHS}")

    for epoch in range(N_EPOCHS):
        current_sigma = get_sigma_schedule(epoch)
        current_lr_gen = get_learning_rate_schedule(epoch, LR_GEN_INITIAL)
        current_lr_test = get_learning_rate_schedule(epoch, LR_TEST_INITIAL)

        for param_group in optimizer_gen.param_groups:
            param_group['lr'] = current_lr_gen
        for param_group in optimizer_test.param_groups:
            param_group['lr'] = current_lr_test

        # Train test functions
        optimizer_test.zero_grad()
        loss = compute_loss(pushforward_net, test_funcs, current_sigma)
        (-loss).backward()
        if USE_GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_(test_funcs.parameters(), GRAD_CLIP_VALUE)
        optimizer_test.step()

        # Train generator
        optimizer_gen.zero_grad()
        loss = compute_loss(pushforward_net, test_funcs, current_sigma)
        loss.backward()
        if USE_GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_(pushforward_net.parameters(), GRAD_CLIP_VALUE)
        optimizer_gen.step()

        loss_log.append(loss.item())
        sigma_log.append(current_sigma)
        lr_gen_log.append(current_lr_gen)
        lr_test_log.append(current_lr_test)

        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.6e} | σ: {current_sigma:.4f} | "
                  f"lr_gen: {current_lr_gen:.6e}")

    return pushforward_net, loss_log, sigma_log, lr_gen_log, lr_test_log


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Triple-Well Potential Training")
    print("=" * 70)

    # Train neural network
    print("\n1. Training neural network with curriculum learning...")
    pushforward_net, loss_log, sigma_log, lr_gen_log, lr_test_log = train_neural_network()

    # Save results
    print("\n2. Saving results...")
    torch.save({
        'pushforward_net_state_dict': pushforward_net.state_dict(),
        'loss_log': loss_log,
        'sigma_log': sigma_log,
        'lr_gen_log': lr_gen_log,
        'lr_test_log': lr_test_log,
        'hyperparameters': {
            'DIM': DIM,
            'SIGMA_INITIAL': SIGMA_INITIAL,
            'SIGMA_FINAL': SIGMA_FINAL,
            'T_FINAL': T_FINAL,
            'MU_1': MU_1,
            'MU_2': MU_2,
            'SIGMA_IC': SIGMA_IC,
            'MIXTURE_WEIGHT': MIXTURE_WEIGHT,
            'K': K,
            'M': M,
            'M_0': M_0,
            'M_T': M_T,
            'D_BASE': D_BASE,
            'N_EPOCHS': N_EPOCHS,
            'ANNEALING_START_EPOCH': ANNEALING_START_EPOCH,
            'ANNEALING_END_EPOCH': ANNEALING_END_EPOCH,
            'USE_LR_DECAY': USE_LR_DECAY,
            'USE_GRADIENT_CLIPPING': USE_GRADIENT_CLIPPING,
        }
    }, 'results_triple_well/checkpoint.pth')

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nTraining:")
    print(f"  Final loss: {loss_log[-1]:.6e}")
    print(f"  Minimum loss: {min(loss_log):.6e} at epoch {np.argmin(loss_log)}")
    print(f"  Final σ: {sigma_log[-1]:.3f}")

    print("\n" + "=" * 70)
    print("✓ Training complete! Results saved to 'results_triple_well/'")
    print("\nSaved files:")
    print("  1. checkpoint.pth - Trained model and training logs")
    print("\nRun plot_triple_well.py to visualize results and compare with FV solution")
    print("=" * 70)

    # Import and run plotting if available
    try:
        import plot_triple_well
        print("\n" + "=" * 70)
        print("Running plotting script...")
        print("=" * 70)
        plot_triple_well.main()
    except ImportError:
        print("\nNote: plot_triple_well.py not found. Run it separately to generate plots.")



if __name__ == '__main__':
    main()