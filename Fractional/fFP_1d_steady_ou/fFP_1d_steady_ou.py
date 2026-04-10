import matplotlib; matplotlib.use('Agg')
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
ALPHA   = 1.5
D       = 1
D_BASE  = 5
K_TEST  = 200
N_EPOCHS = 3000

MU      = 2.0    # OU target mean
THETA   = 1.0    # OU mean-reversion rate

M_SAMPLES = 2000  # batch size for loss computation

LR_GENERATOR  = 1e-3
LR_ADVERSARY  = 1e-2
ADV_UPDATE_FREQ    = 1
ADV_STEPS_PER_UPDATE = 1

N_PARTICLES = 10000
T_SDE  = 50.0
DT_SDE = 0.01

N_BINS  = 80
X_RANGE = (-2, 6)


# ============================================================================
# DRIFT
# ============================================================================
def drift_field(x):
    """b(x) = -theta*(x - mu)"""
    return -THETA * (x - MU)


# ============================================================================
# STEADY-STATE NETWORK
# ============================================================================
class SteadyPushforwardNetwork(nn.Module):
    def __init__(self, d_base, d_out, hidden_dims=[128, 128, 128]):
        super().__init__()
        layers = []
        prev = d_base
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, r):
        return self.net(r)


# ============================================================================
# TEST FUNCTIONS
# ============================================================================
class SteadyPlaneWaveTestFunctions:
    def __init__(self, k_test, d, device):
        self.w = nn.Parameter(torch.randn(k_test, d, device=device))
        self.b = nn.Parameter(torch.rand(k_test, device=device) * 2 * np.pi)
        with torch.no_grad():
            norms = torch.norm(self.w.data, dim=1, keepdim=True).clamp(min=1e-8)
            self.w.data = self.w.data / norms

    def parameters(self):
        return [self.w, self.b]


# ============================================================================
# LOSS
# ============================================================================
def compute_steady_loss(generator, test_functions, M):
    r = torch.randn(M, D_BASE, device=device)
    x = generator(r)                          # (M, D)
    phase = x @ test_functions.w.T + test_functions.b.unsqueeze(0)  # (M, K)
    f     = torch.sin(phase)
    cos_p = torch.cos(phase)
    w_norm_alpha = torch.norm(test_functions.w, dim=1) ** ALPHA    # (K,)
    b_vals   = drift_field(x)                  # (M, D)
    b_dot_w  = b_vals @ test_functions.w.T     # (M, K)
    Lf = -w_norm_alpha.unsqueeze(0) * f + b_dot_w * cos_p
    residual = Lf.mean(dim=0)                  # (K,)
    loss = (residual ** 2).mean()
    return loss, residual


# ============================================================================
# PARTICLE BENCHMARK
# ============================================================================
def run_particle_benchmark():
    print("\nRunning particle simulation benchmark...")
    particles = np.random.randn(N_PARTICLES)   # start from N(0,1)
    n_steps = int(T_SDE / DT_SDE)
    for step in range(n_steps):
        drift = drift_field(particles)
        if abs(ALPHA - 2.0) < 1e-6:
            noise = np.random.randn(N_PARTICLES) * np.sqrt(2 * DT_SDE)
        else:
            noise = stats.levy_stable.rvs(
                alpha=ALPHA, beta=0, loc=0,
                scale=DT_SDE ** (1.0 / ALPHA), size=N_PARTICLES)
        particles = particles + drift * DT_SDE + noise
        if (step + 1) % 1000 == 0:
            print(f"  step {step+1}/{n_steps}, mean={np.mean(particles):.3f}, "
                  f"median={np.median(particles):.3f}")
    print("Particle simulation complete.")
    return particles


# ============================================================================
# TRAINING
# ============================================================================
def train():
    generator     = SteadyPushforwardNetwork(D_BASE, D).to(device)
    test_functions = SteadyPlaneWaveTestFunctions(K_TEST, D, device)

    opt_G = torch.optim.Adam(generator.parameters(), lr=LR_GENERATOR)
    opt_A = torch.optim.Adam(test_functions.parameters(), lr=LR_ADVERSARY)
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, N_EPOCHS)

    history = {'loss': [], 'residual_norm': []}
    print(f"Training: N_EPOCHS={N_EPOCHS}, K_TEST={K_TEST}, ALPHA={ALPHA}")

    for epoch in range(N_EPOCHS):
        opt_G.zero_grad()
        loss, residual = compute_steady_loss(generator, test_functions, M_SAMPLES)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_G.step()
        sched_G.step()

        if epoch % ADV_UPDATE_FREQ == 0:
            for _ in range(ADV_STEPS_PER_UPDATE):
                opt_A.zero_grad()
                loss_adv, _ = compute_steady_loss(generator, test_functions, M_SAMPLES)
                (-loss_adv).backward()
                torch.nn.utils.clip_grad_norm_(test_functions.parameters(), 1.0)
                opt_A.step()

        history['loss'].append(loss.item())
        history['residual_norm'].append(torch.norm(residual).item())

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{N_EPOCHS} | Loss: {loss.item():.6f} | "
                  f"Residual norm: {torch.norm(residual).item():.6f}")

    return generator, test_functions, history


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(generator, particle_samples, history):
    # Generate learned samples
    with torch.no_grad():
        r = torch.randn(10000, D_BASE, device=device)
        x_learned = generator(r).cpu().numpy().flatten()

    # --- Histogram comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(x_learned, bins=N_BINS, range=X_RANGE, density=True,
            alpha=0.6, color='steelblue', label='Learned (WANPM)')
    ax.hist(particle_samples, bins=N_BINS, range=X_RANGE, density=True,
            alpha=0.6, color='darkorange', label='Particle SDE')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Steady-State 1D OU (α={ALPHA})\nLearned vs Particle', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(history['loss'], color='steelblue', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss', fontsize=13)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Fractional Steady-State OU  (α={ALPHA}, μ={MU}, θ={THETA})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'steady_ou_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Fractional Steady-State 1D OU  (Section 4.1 variant)")
    print("=" * 70)

    particle_samples = run_particle_benchmark()
    generator, test_functions, history = train()
    plot_results(generator, particle_samples, history)

    torch.save({
        'generator_state_dict': generator.state_dict(),
        'hyperparameters': {
            'alpha': ALPHA, 'd': D, 'd_base': D_BASE,
            'mu': MU, 'theta': THETA,
        }
    }, os.path.join(OUTPUT_DIR, 'model.pt'))
    print("Model saved.")
    print("Done.")


if __name__ == '__main__':
    main()
