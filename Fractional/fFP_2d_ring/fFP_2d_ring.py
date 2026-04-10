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
ALPHA    = 1.5
D        = 2
D_BASE   = 8
T_FINAL  = 0.5
K_TEST   = 300
N_EPOCHS = 5000
M_INTERIOR = 3000
M_INITIAL  = 1000

R0    = 2.0
OMEGA = 2.0

# Initial condition: N((0, 1.2), 0.4^2 * I)
IC_MU    = np.array([0.0, 1.2])
IC_SIGMA = 0.4

LR_GENERATOR  = 1e-3
LR_ADVERSARY  = 1e-2
ADV_UPDATE_FREQ      = 1
ADV_STEPS_PER_UPDATE = 1

N_PARTICLES = 10000
DT          = 0.01

SNAPSHOT_TIMES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


# ============================================================================
# DRIFT
# ============================================================================
def drift_field(x):
    """x: (M, 2) -> b: (M, 2)"""
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    r2 = x1 ** 2 + x2 ** 2
    b1 = -2.0 * (r2 - R0 ** 2) * x1 + OMEGA * (-x2)
    b2 = -2.0 * (r2 - R0 ** 2) * x2 + OMEGA * x1
    return torch.cat([b1, b2], dim=1)


def drift_field_np(x):
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    r2 = x1 ** 2 + x2 ** 2
    b1 = -2.0 * (r2 - R0 ** 2) * x1 + OMEGA * (-x2)
    b2 = -2.0 * (r2 - R0 ** 2) * x2 + OMEGA * x1
    return np.concatenate([b1, b2], axis=1)


# ============================================================================
# INITIAL CONDITION
# ============================================================================
def sample_initial_condition(n):
    mu = torch.tensor(IC_MU, dtype=torch.float32, device=device)  # (D,)
    return mu.unsqueeze(0) + IC_SIGMA * torch.randn(n, D, device=device)


def sample_initial_condition_np(n):
    return IC_MU + IC_SIGMA * np.random.randn(n, D)


# ============================================================================
# PUSHFORWARD NETWORK
# F(t, x0, r) = x0 + (t + 1e-8)^(1/alpha) * F_tilde(t, x0, r)
# 4 hidden layers of width 256, Tanh
# ============================================================================
class PushforwardNetwork(nn.Module):
    def __init__(self, d_base, d_out, hidden_dims=[256, 256, 256, 256]):
        super().__init__()
        input_dim = 1 + d_out + d_base
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, t, x0, r):
        inp = torch.cat([t, x0, r], dim=1)
        return x0 + (t + 1e-8) ** (1.0 / ALPHA) * self.net(inp)


# ============================================================================
# TEST FUNCTIONS
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
        return x @ self.w.T + self.kappa.T * t + self.b.T

    def f(self, t, x):
        return torch.sin(self.phase(t, x))

    def df_dt(self, t, x):
        return self.kappa.T * torch.cos(self.phase(t, x))

    def grad_f(self, t, x):
        cos_ph = torch.cos(self.phase(t, x))   # (M, K)
        return cos_ph.unsqueeze(-1) * self.w.unsqueeze(0)  # (M, K, D)

    def fractional_laplacian_f(self, t, x):
        w_norm_alpha = torch.norm(self.w, dim=1) ** ALPHA
        return w_norm_alpha.unsqueeze(0) * self.f(t, x)


# ============================================================================
# LOSS
# ============================================================================
def compute_loss(generator, tf, t_final):
    x0_init = sample_initial_condition(M_INITIAL)
    t0 = torch.zeros(M_INITIAL, 1, device=device)
    E_initial = tf.f(t0, x0_init).mean(dim=0)

    x0_T = sample_initial_condition(M_INITIAL)
    r_T  = torch.randn(M_INITIAL, D_BASE, device=device)
    t_T  = torch.full((M_INITIAL, 1), t_final, device=device)
    x_T  = generator(t_T, x0_T, r_T)
    E_terminal = tf.f(t_T, x_T).mean(dim=0)

    t_int  = torch.rand(M_INTERIOR, 1, device=device) * t_final
    x0_int = sample_initial_condition(M_INTERIOR)
    r_int  = torch.randn(M_INTERIOR, D_BASE, device=device)
    x_int  = generator(t_int, x0_int, r_int)

    frac_L  = tf.fractional_laplacian_f(t_int, x_int)
    grad_f  = tf.grad_f(t_int, x_int)
    b_vals  = drift_field(x_int)
    b_dot_g = (b_vals.unsqueeze(1) * grad_f).sum(dim=2)
    Lf = -frac_L + b_dot_g

    df_dt = tf.df_dt(t_int, x_int)
    E_interior = t_final * (df_dt + Lf).mean(dim=0)

    residual = E_terminal - E_initial - E_interior
    loss = (residual ** 2).mean()
    return loss, residual


# ============================================================================
# PARTICLE BENCHMARK
# ============================================================================
def run_particle_benchmark(snapshot_times):
    print("\nRunning 2D ring particle simulation benchmark...")
    particles = sample_initial_condition_np(N_PARTICLES)    # (N, 2)
    n_steps = int(np.ceil(T_FINAL / DT))
    trajectories = {}

    for step in range(n_steps):
        t_curr = step * DT
        for st in snapshot_times:
            if abs(t_curr - st) < DT / 2 and st not in trajectories:
                trajectories[st] = particles.copy()

        drift = drift_field_np(particles)
        if abs(ALPHA - 2.0) < 1e-6:
            noise = np.random.randn(N_PARTICLES, D) * np.sqrt(2 * DT)
        else:
            noise = np.stack([
                stats.levy_stable.rvs(alpha=ALPHA, beta=0, loc=0,
                                      scale=DT ** (1.0 / ALPHA), size=N_PARTICLES)
                for _ in range(D)
            ], axis=1)
        particles = particles + drift * DT + noise

    trajectories[T_FINAL] = particles.copy()
    # Ensure t=0 is captured (exact)
    if 0.0 not in trajectories:
        trajectories[0.0] = sample_initial_condition_np(N_PARTICLES)
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
    fig, axes = plt.subplots(2, n_snaps, figsize=(4 * n_snaps, 9))

    theta_ring = np.linspace(0, 2 * np.pi, 200)
    ring_x = R0 * np.cos(theta_ring)
    ring_y = R0 * np.sin(theta_ring)

    for i, t_val in enumerate(SNAPSHOT_TIMES):
        # Learned
        if t_val == 0.0:
            with torch.no_grad():
                x_nn = sample_initial_condition(3000).cpu().numpy()
        else:
            with torch.no_grad():
                x0 = sample_initial_condition(3000)
                r  = torch.randn(3000, D_BASE, device=device)
                t_ = torch.full((3000, 1), t_val, device=device)
                x_nn = generator(t_, x0, r).cpu().numpy()

        # Particle
        t_key = min(particle_traj.keys(), key=lambda k: abs(k - t_val))
        x_part = particle_traj[t_key]

        for row, (data, label, color) in enumerate([
            (x_nn, 'WANPM', 'steelblue'),
            (x_part[:3000], 'Particle', 'darkorange')
        ]):
            ax = axes[row, i]
            ax.scatter(data[:, 0], data[:, 1], s=2, alpha=0.4, color=color)
            ax.plot(ring_x, ring_y, 'r--', linewidth=1.5, alpha=0.7)
            ax.set_title(f't={t_val} ({label})', fontsize=9)
            ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
            ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
            ax.grid(True, alpha=0.3)

    plt.suptitle(f'Fractional Time-Dependent 2D Ring  (α={ALPHA}, r₀={R0}, ω={OMEGA})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ring_time_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Separate loss plot
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(history['loss'], color='steelblue', linewidth=1.5)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Plots saved.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Fractional Time-Dependent 2D Ring  (Section 4.7 variant)")
    print("=" * 70)

    particle_traj = run_particle_benchmark(SNAPSHOT_TIMES)
    generator, tf, history = train()
    plot_results(generator, particle_traj, history)

    torch.save({
        'generator_state_dict': generator.state_dict(),
        'hyperparameters': {'alpha': ALPHA, 'd': D, 'd_base': D_BASE,
                            't_final': T_FINAL, 'r0': R0, 'omega': OMEGA},
    }, os.path.join(OUTPUT_DIR, 'model.pt'))
    print("Model saved.")
    print("Done.")


if __name__ == '__main__':
    main()
