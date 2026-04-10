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
D       = 2
D_BASE  = 4
K_TEST  = 5000
N_EPOCHS = 10000

R0 = 2.0  # target ring radius
OMEGA = 2.0  # rotation frequency

M_SAMPLES = 5000

LR_GENERATOR  = 1e-3
LR_ADVERSARY  = 1e-2
ADV_UPDATE_FREQ      = 1
ADV_STEPS_PER_UPDATE = 1

N_PARTICLES = 10000
T_SDE  = 20.0
DT_SDE = 0.01

MC_SIZES = [100, 300, 1000, 3000, 10000]


# ============================================================================
# DRIFT
# b_rad = -2(r^2 - r0^2) * [x1, x2]  where r^2 = x1^2+x2^2
# b_tan = omega * [-x2, x1]
# ============================================================================
def drift_field(x):
    """x: (M, 2) -> b: (M, 2)"""
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    r2 = x1 ** 2 + x2 ** 2
    b_rad1 = -2.0 * (r2 - R0 ** 2) * x1
    b_rad2 = -2.0 * (r2 - R0 ** 2) * x2
    b_tan1 = OMEGA * (-x2)
    b_tan2 = OMEGA * x1
    return torch.cat([b_rad1 + b_tan1, b_rad2 + b_tan2], dim=1)


def drift_field_np(x):
    """x: (N, 2) numpy -> b: (N, 2)"""
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    r2 = x1 ** 2 + x2 ** 2
    b1 = -2.0 * (r2 - R0 ** 2) * x1 + OMEGA * (-x2)
    b2 = -2.0 * (r2 - R0 ** 2) * x2 + OMEGA * x1
    return np.concatenate([b1, b2], axis=1)


# ============================================================================
# NETWORK
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
    x = generator(r)
    phase = x @ test_functions.w.T + test_functions.b.unsqueeze(0)
    f     = torch.sin(phase)
    cos_p = torch.cos(phase)
    w_norm_alpha = torch.norm(test_functions.w, dim=1) ** ALPHA
    b_vals  = drift_field(x)
    b_dot_w = b_vals @ test_functions.w.T
    Lf = -w_norm_alpha.unsqueeze(0) * f + b_dot_w * cos_p
    residual = Lf.mean(dim=0)
    loss = (residual ** 2).mean()
    return loss, residual


# ============================================================================
# PARTICLE BENCHMARK
# ============================================================================
def run_particle_benchmark():
    print("\nRunning 2D ring particle simulation benchmark...")
    particles = np.random.randn(N_PARTICLES, D)
    n_steps = int(T_SDE / DT_SDE)
    for step in range(n_steps):
        drift = drift_field_np(particles)
        if abs(ALPHA - 2.0) < 1e-6:
            noise = np.random.randn(N_PARTICLES, D) * np.sqrt(2 * DT_SDE)
        else:
            noise = np.stack([
                stats.levy_stable.rvs(alpha=ALPHA, beta=0, loc=0,
                                      scale=DT_SDE ** (1.0 / ALPHA), size=N_PARTICLES)
                for _ in range(D)
            ], axis=1)
        particles = particles + drift * DT_SDE + noise
        if (step + 1) % 500 == 0:
            r_mean = np.mean(np.sqrt(particles[:, 0]**2 + particles[:, 1]**2))
            print(f"  step {step+1}/{n_steps}, mean_r={r_mean:.3f}")
    print("Particle simulation complete.")
    return particles


# ============================================================================
# TRAINING
# ============================================================================
def train():
    generator      = SteadyPushforwardNetwork(D_BASE, D).to(device)
    test_functions = SteadyPlaneWaveTestFunctions(K_TEST, D, device)

    opt_G  = torch.optim.Adam(generator.parameters(), lr=LR_GENERATOR)
    opt_A  = torch.optim.Adam(test_functions.parameters(), lr=LR_ADVERSARY)
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

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{N_EPOCHS} | Loss: {loss.item():.6f}")

    return generator, test_functions, history


# ============================================================================
# MC INTEGRATION
# ============================================================================
def mc_integration(generator, particle_samples):
    """Compare MC estimates from learned sampler vs reference from particles."""
    print("\nComputing MC convergence...")

    def f1(x):
        return np.exp(-(x[:, 0] - 1)**2 - (x[:, 1] - 1)**2)

    def f2(x):
        return np.cos(x[:, 0]) * np.sin(x[:, 1])

    # Reference values from particle samples
    ref1 = f1(particle_samples).mean()
    ref2 = f2(particle_samples).mean()
    print(f"  Reference f1: {ref1:.6f}")
    print(f"  Reference f2: {ref2:.6f}")

    mc_errors_f1 = []
    mc_errors_f2 = []

    with torch.no_grad():
        # Generate large batch of learned samples
        r_all = torch.randn(MC_SIZES[-1], D_BASE, device=device)
        x_all = generator(r_all).cpu().numpy()

    for N in MC_SIZES:
        x_sub = x_all[:N]
        est1 = f1(x_sub).mean()
        est2 = f2(x_sub).mean()
        mc_errors_f1.append(abs(est1 - ref1))
        mc_errors_f2.append(abs(est2 - ref2))
        print(f"  N={N:6d}: |f1_err|={abs(est1-ref1):.6f}, |f2_err|={abs(est2-ref2):.6f}")

    return mc_errors_f1, mc_errors_f2, ref1, ref2


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(generator, particle_samples, history, mc_errors_f1, mc_errors_f2):
    with torch.no_grad():
        r = torch.randn(5000, D_BASE, device=device)
        x_learned = generator(r).cpu().numpy()

    theta_ring = np.linspace(0, 2 * np.pi, 200)
    ring_x = R0 * np.cos(theta_ring)
    ring_y = R0 * np.sin(theta_ring)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(x_learned[:, 0], x_learned[:, 1], s=2, alpha=0.4, color='steelblue')
    ax.plot(ring_x, ring_y, 'r--', linewidth=2, label=f'r={R0} ring')
    ax.set_title('Learned Samples (WANPM)', fontsize=12)
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.legend(); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(history['loss'], color='steelblue', linewidth=1.5)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.loglog(MC_SIZES, mc_errors_f1, 'b-o', label='f₁ = exp(-...) error')
    ax3.loglog(MC_SIZES, mc_errors_f2, 'r-s', label='f₂ = cos·sin error')
    # Reference 1/sqrt(N) line
    ref_line = mc_errors_f1[0] * np.sqrt(MC_SIZES[0]) / np.sqrt(np.array(MC_SIZES))
    ax3.loglog(MC_SIZES, ref_line, 'k--', alpha=0.5, label='1/√N reference')
    ax3.set_xlabel('Sample size N')
    ax3.set_ylabel('|MC error|')
    ax3.set_title('MC Convergence (Section 4.6)')
    ax3.legend(); ax3.grid(True, alpha=0.3)

    plt.suptitle(f'Fractional Steady-State 2D Ring  (α={ALPHA}, r₀={R0}, ω={OMEGA})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ring_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Fractional Steady-State 2D Ring  (Sections 4.5+4.6 variant)")
    print("=" * 70)

    particle_samples = run_particle_benchmark()
    generator, test_functions, history = train()
    mc_errors_f1, mc_errors_f2, ref1, ref2 = mc_integration(generator, particle_samples)
    plot_results(generator, particle_samples, history, mc_errors_f1, mc_errors_f2)

    torch.save({
        'generator_state_dict': generator.state_dict(),
        'hyperparameters': {'alpha': ALPHA, 'd': D, 'd_base': D_BASE, 'r0': R0, 'omega': OMEGA},
        'mc': {'errors_f1': mc_errors_f1, 'errors_f2': mc_errors_f2, 'ref1': ref1, 'ref2': ref2}
    }, os.path.join(OUTPUT_DIR, 'model.pt'))
    print("Model saved.")
    print("Done.")


if __name__ == '__main__':
    main()
