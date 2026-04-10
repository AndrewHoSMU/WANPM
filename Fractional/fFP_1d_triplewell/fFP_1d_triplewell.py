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
ALPHA      = 1.5
D          = 1
D_BASE     = 8
T_FINAL    = 2.5
N_EPOCHS   = 5000
K_TEST     = 2000
M_INTERIOR = 3000
M_INITIAL  = 1000

LR_GENERATOR  = 1e-3
LR_ADVERSARY  = 1e-2
ADV_UPDATE_FREQ      = 1
ADV_STEPS_PER_UPDATE = 1

N_PARTICLES = 10000
DT          = 0.01

SNAPSHOT_TIMES = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5]
N_BINS  = 80
X_RANGE = (-1.5, 1.5)


# ============================================================================
# POTENTIAL & DRIFT
# V(x) = x^2*(x^2-1)^2,  b(x) = -V'(x) = -2x(x^2-1)(3x^2-1)
# ============================================================================
def drift_field_torch(x):
    """x: (M,1)"""
    return -2.0 * x * (x ** 2 - 1.0) * (3.0 * x ** 2 - 1.0)


def drift_field_np(x):
    return -2.0 * x * (x ** 2 - 1.0) * (3.0 * x ** 2 - 1.0)


# ============================================================================
# INITIAL CONDITION
# Equal-weight mixture: 0.5*N(-0.5, 0.15^2) + 0.5*N(+0.5, 0.15^2)
# ============================================================================
IC_SIGMA = 0.15
IC_CENTERS = [-0.5, 0.5]

def sample_initial_condition(n):
    idx = (torch.rand(n, device=device) > 0.5).long()
    centers = torch.tensor(IC_CENTERS, dtype=torch.float32, device=device)
    mu = centers[idx].unsqueeze(1)   # (n, 1)
    return mu + IC_SIGMA * torch.randn(n, 1, device=device)


def sample_initial_condition_np(n):
    idx = np.random.randint(0, 2, size=n)
    centers = np.array(IC_CENTERS)
    mu = centers[idx]
    return mu + IC_SIGMA * np.random.randn(n)


# ============================================================================
# PUSHFORWARD NETWORK  (time-dependent)
# F(t, x0, r) = x0 + (t + 1e-8)^(1/alpha) * F_tilde(t, x0, r)
# ============================================================================
class PushforwardNetwork(nn.Module):
    def __init__(self, d_base, d_out, hidden_dims=[128, 128, 128]):
        super().__init__()
        input_dim = 1 + d_out + d_base   # (t, x0, r)
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, t, x0, r):
        """t:(M,1), x0:(M,D), r:(M,D_BASE) -> x:(M,D)"""
        inp = torch.cat([t, x0, r], dim=1)
        return x0 + (t + 1e-8) ** (1.0 / ALPHA) * self.net(inp)


# ============================================================================
# TEST FUNCTIONS  (time-dependent plane waves)
# ============================================================================
class PlaneWaveTestFunctions:
    def __init__(self, k_test, d, device):
        self.w     = nn.Parameter(torch.randn(k_test, d, device=device))
        self.kappa = nn.Parameter(torch.randn(k_test, 1, device=device))
        self.b     = nn.Parameter(torch.rand(k_test, 1, device=device) * 2 * np.pi)
        with torch.no_grad():
            norms = torch.norm(self.w.data, dim=1, keepdim=True).clamp(min=1e-8)
            self.w.data = self.w.data / norms

    def parameters(self):
        return [self.w, self.kappa, self.b]

    def phase(self, t, x):
        return x @ self.w.T + self.kappa.T * t + self.b.T  # (M, K)

    def f(self, t, x):
        return torch.sin(self.phase(t, x))

    def df_dt(self, t, x):
        return self.kappa.T * torch.cos(self.phase(t, x))

    def grad_f(self, t, x):
        """(M, K, D)"""
        cos_ph = torch.cos(self.phase(t, x))   # (M,K)
        return cos_ph.unsqueeze(-1) * self.w.unsqueeze(0)

    def fractional_laplacian_f(self, t, x):
        w_norm_alpha = torch.norm(self.w, dim=1) ** ALPHA   # (K,)
        return w_norm_alpha.unsqueeze(0) * self.f(t, x)


# ============================================================================
# LOSS
# ============================================================================
def compute_loss(generator, tf, t_final):
    # --- Initial term ---
    x0_init = sample_initial_condition(M_INITIAL)
    t0 = torch.zeros(M_INITIAL, 1, device=device)
    E_initial = tf.f(t0, x0_init).mean(dim=0)                         # (K,)

    # --- Terminal term ---
    x0_T = sample_initial_condition(M_INITIAL)
    r_T  = torch.randn(M_INITIAL, D_BASE, device=device)
    t_T  = torch.full((M_INITIAL, 1), t_final, device=device)
    x_T  = generator(t_T, x0_T, r_T)
    E_terminal = tf.f(t_T, x_T).mean(dim=0)                           # (K,)

    # --- Interior term ---
    t_int = torch.rand(M_INTERIOR, 1, device=device) * t_final
    x0_int = sample_initial_condition(M_INTERIOR)
    r_int  = torch.randn(M_INTERIOR, D_BASE, device=device)
    x_int  = generator(t_int, x0_int, r_int)

    frac_L = tf.fractional_laplacian_f(t_int, x_int)    # (M, K)
    grad_f = tf.grad_f(t_int, x_int)                    # (M, K, D)
    b_vals = drift_field_torch(x_int)                    # (M, D)
    b_dot_grad = (b_vals.unsqueeze(1) * grad_f).sum(dim=2)  # (M, K)
    Lf = -frac_L + b_dot_grad

    df_dt = tf.df_dt(t_int, x_int)
    E_interior = t_final * (df_dt + Lf).mean(dim=0)     # (K,)

    residual = E_terminal - E_initial - E_interior
    loss = (residual ** 2).mean()
    return loss, residual


# ============================================================================
# PARTICLE BENCHMARK
# ============================================================================
def run_particle_benchmark(snapshot_times):
    print("\nRunning particle simulation benchmark (triple-well)...")
    particles = sample_initial_condition_np(N_PARTICLES)
    n_steps = int(np.ceil(T_FINAL / DT))
    trajectories = {}
    snap_set = set(snapshot_times)

    for step in range(n_steps):
        t_curr = step * DT
        # store snapshots
        for st in snapshot_times:
            if abs(t_curr - st) < DT / 2 and st not in trajectories:
                trajectories[st] = particles.copy()

        drift = drift_field_np(particles)
        if abs(ALPHA - 2.0) < 1e-6:
            noise = np.random.randn(N_PARTICLES) * np.sqrt(2 * DT)
        else:
            noise = stats.levy_stable.rvs(
                alpha=ALPHA, beta=0, loc=0,
                scale=DT ** (1.0 / ALPHA), size=N_PARTICLES)
        particles = particles + drift * DT + noise
        # Clip to prevent NaN propagation from heavy-tailed jumps
        particles = np.clip(particles, -20.0, 20.0)

        if (step + 1) % 50 == 0 and (step + 1) * DT <= T_FINAL:
            pass  # silent

    # final time
    trajectories[T_FINAL] = particles.copy()
    print("Particle simulation complete.")
    return trajectories


# ============================================================================
# TRAINING
# ============================================================================
def train():
    generator = PushforwardNetwork(D_BASE, D).to(device)
    tf        = PlaneWaveTestFunctions(K_TEST, D, device)

    opt_G  = torch.optim.Adam(generator.parameters(), lr=LR_GENERATOR)
    opt_A  = torch.optim.Adam(tf.parameters(), lr=LR_ADVERSARY)
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, N_EPOCHS)

    history = {'loss': [], 'residual_norm': []}
    print(f"Training: N_EPOCHS={N_EPOCHS}, K_TEST={K_TEST}, ALPHA={ALPHA}")

    for epoch in range(N_EPOCHS):
        opt_G.zero_grad()
        loss, residual = compute_loss(generator, tf, T_FINAL)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_G.step()
        sched_G.step()

        if epoch % ADV_UPDATE_FREQ == 0:
            for _ in range(ADV_STEPS_PER_UPDATE):
                opt_A.zero_grad()
                loss_adv, _ = compute_loss(generator, tf, T_FINAL)
                (-loss_adv).backward()
                torch.nn.utils.clip_grad_norm_(tf.parameters(), 1.0)
                opt_A.step()

        history['loss'].append(loss.item())
        history['residual_norm'].append(torch.norm(residual).item())

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{N_EPOCHS} | Loss: {loss.item():.6f}")

    return generator, tf, history


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(generator, particle_traj, history):
    n_snaps = len(SNAPSHOT_TIMES)
    fig, axes = plt.subplots(2, n_snaps, figsize=(3.5 * n_snaps, 8))

    for i, t_val in enumerate(SNAPSHOT_TIMES):
        # Learned samples
        with torch.no_grad():
            x0 = sample_initial_condition(5000)
            r  = torch.randn(5000, D_BASE, device=device)
            t_ = torch.full((5000, 1), t_val, device=device)
            x_nn = generator(t_, x0, r).cpu().numpy().flatten()

        # Particle samples
        t_key = min(particle_traj.keys(), key=lambda k: abs(k - t_val))
        x_part = particle_traj[t_key]

        ax = axes[0, i]
        ax.hist(x_nn, bins=N_BINS, range=X_RANGE, density=True,
                alpha=0.6, color='steelblue', label='WANPM')
        ax.hist(x_part, bins=N_BINS, range=X_RANGE, density=True,
                alpha=0.6, color='darkorange', label='Particle')
        ax.set_title(f't={t_val}', fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

        ax2 = axes[1, i]
        ax2.axis('off')

    # Loss plot in last bottom cell
    axes[1, -1].axis('on')
    axes[1, -1].plot(history['loss'], color='steelblue', linewidth=1.5)
    axes[1, -1].set_xlabel('Epoch'); axes[1, -1].set_ylabel('Loss')
    axes[1, -1].set_title('Training Loss')
    axes[1, -1].set_yscale('log')
    axes[1, -1].grid(True, alpha=0.3)

    plt.suptitle(f'Fractional 1D Triple-Well (α={ALPHA})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'triplewell_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Fractional Time-Dependent 1D Triple-Well  (Section 4.4 variant)")
    print("=" * 70)

    particle_traj = run_particle_benchmark(SNAPSHOT_TIMES)
    generator, tf, history = train()
    plot_results(generator, particle_traj, history)

    torch.save({
        'generator_state_dict': generator.state_dict(),
        'hyperparameters': {'alpha': ALPHA, 'd': D, 'd_base': D_BASE, 't_final': T_FINAL},
    }, os.path.join(OUTPUT_DIR, 'model.pt'))
    print("Model saved.")
    print("Done.")


if __name__ == '__main__':
    main()
