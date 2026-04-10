import matplotlib; matplotlib.use('Agg')
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from dataclasses import dataclass, field
from typing import List

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# CONFIG
# ============================================================================
@dataclass
class Config:
    dim: int
    mu0: List[float]       # initial mean
    d_base: int
    hidden: List[int]
    n_epochs: int
    k_test: int
    m_interior: int
    output_dir: str
    alpha: float = 1.5
    theta: float = 1.0     # OU mean-reversion (toward 0)
    mu_target: float = 0.0 # OU target mean
    t_final: float = 1.0
    sigma0: float = 0.5    # IC std
    m_initial: int = 1000
    n_particles: int = 10000
    dt: float = 0.01

    def lr_generator(self):  return 1e-3
    def lr_adversary(self):  return 1e-2


CONFIGS = [
    Config(
        dim=5,
        mu0=[3.0, -2.5, 2.0, -3.5, 1.5],
        d_base=10,
        hidden=[64, 64],
        n_epochs=2000,
        k_test=200,
        m_interior=3000,
        output_dir=os.path.join(SCRIPT_DIR, 'outputs_5d'),
    ),
    Config(
        dim=10,
        mu0=[3.0, -2.5, 2.0, -3.5, 1.5, -1.0, 2.5, -2.0, 3.5, -1.5],
        d_base=20,
        hidden=[64, 64],
        n_epochs=2000,
        k_test=200,
        m_interior=5000,
        output_dir=os.path.join(SCRIPT_DIR, 'outputs_10d'),
    ),
]

SNAPSHOT_TIMES = [0.01, 0.2, 0.5, 1.0]


# ============================================================================
# DRIFT: OU toward mu=0, dX = -theta * X dt + dL
# ============================================================================
def drift_field(x, cfg: Config):
    return -cfg.theta * (x - cfg.mu_target)


def drift_field_np(x, cfg: Config):
    return -cfg.theta * (x - cfg.mu_target)


# ============================================================================
# INITIAL CONDITION: N(mu0, sigma0^2 I)
# ============================================================================
def sample_ic(n, cfg: Config):
    mu = torch.tensor(cfg.mu0, dtype=torch.float32, device=device)
    return mu.unsqueeze(0) + cfg.sigma0 * torch.randn(n, cfg.dim, device=device)


def sample_ic_np(n, cfg: Config):
    return np.array(cfg.mu0) + cfg.sigma0 * np.random.randn(n, cfg.dim)


# ============================================================================
# NETWORK
# ============================================================================
class PushforwardNetwork(nn.Module):
    def __init__(self, d_base, d_out, hidden_dims, alpha=1.5):
        super().__init__()
        self.alpha = alpha
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
        return x0 + (t + 1e-8) ** (1.0 / self.alpha) * self.net(inp)


# ============================================================================
# TEST FUNCTIONS
# ============================================================================
class PlaneWaveTestFunctions:
    def __init__(self, k_test, d, alpha, device):
        self.alpha = alpha
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
        cos_ph = torch.cos(self.phase(t, x))
        return cos_ph.unsqueeze(-1) * self.w.unsqueeze(0)

    def fractional_laplacian_f(self, t, x):
        w_norm_alpha = torch.norm(self.w, dim=1) ** self.alpha
        return w_norm_alpha.unsqueeze(0) * self.f(t, x)


# ============================================================================
# LOSS
# ============================================================================
def compute_loss(generator, tf, cfg: Config):
    t_final = cfg.t_final

    x0_init = sample_ic(cfg.m_initial, cfg)
    t0 = torch.zeros(cfg.m_initial, 1, device=device)
    E_initial = tf.f(t0, x0_init).mean(dim=0)

    x0_T = sample_ic(cfg.m_initial, cfg)
    r_T  = torch.randn(cfg.m_initial, cfg.d_base, device=device)
    t_T  = torch.full((cfg.m_initial, 1), t_final, device=device)
    x_T  = generator(t_T, x0_T, r_T)
    E_terminal = tf.f(t_T, x_T).mean(dim=0)

    t_int  = torch.rand(cfg.m_interior, 1, device=device) * t_final
    x0_int = sample_ic(cfg.m_interior, cfg)
    r_int  = torch.randn(cfg.m_interior, cfg.d_base, device=device)
    x_int  = generator(t_int, x0_int, r_int)

    frac_L  = tf.fractional_laplacian_f(t_int, x_int)
    grad_f  = tf.grad_f(t_int, x_int)
    b_vals  = drift_field(x_int, cfg)
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
def run_particle_benchmark(cfg: Config, snapshot_times):
    print(f"\nRunning {cfg.dim}D particle simulation...")
    particles = sample_ic_np(cfg.n_particles, cfg)   # (N, dim)
    n_steps = int(np.ceil(cfg.t_final / cfg.dt))
    trajectories = {}

    for step in range(n_steps):
        t_curr = step * cfg.dt
        for st in snapshot_times:
            if abs(t_curr - st) < cfg.dt / 2 and st not in trajectories:
                trajectories[st] = particles.copy()

        drift = drift_field_np(particles, cfg)
        if abs(cfg.alpha - 2.0) < 1e-6:
            noise = np.random.randn(cfg.n_particles, cfg.dim) * np.sqrt(2 * cfg.dt)
        else:
            noise = np.stack([
                stats.levy_stable.rvs(alpha=cfg.alpha, beta=0, loc=0,
                                      scale=cfg.dt ** (1.0 / cfg.alpha),
                                      size=cfg.n_particles)
                for _ in range(cfg.dim)
            ], axis=1)
        particles = particles + drift * cfg.dt + noise

        if (step + 1) % 20 == 0:
            pass  # silent

    trajectories[cfg.t_final] = particles.copy()
    print(f"{cfg.dim}D particle simulation complete.")
    return trajectories


# ============================================================================
# TRAINING
# ============================================================================
def train(cfg: Config):
    generator = PushforwardNetwork(cfg.d_base, cfg.dim, cfg.hidden, alpha=cfg.alpha).to(device)
    tf        = PlaneWaveTestFunctions(cfg.k_test, cfg.dim, cfg.alpha, device)

    opt_G  = torch.optim.Adam(generator.parameters(), lr=cfg.lr_generator())
    opt_A  = torch.optim.Adam(tf.parameters(), lr=cfg.lr_adversary())
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, cfg.n_epochs)

    history = {'loss': [], 'residual_norm': []}
    print(f"Training {cfg.dim}D: N_EPOCHS={cfg.n_epochs}, K_TEST={cfg.k_test}, ALPHA={cfg.alpha}")

    adv_update_freq = 1
    adv_steps = 1

    for epoch in range(cfg.n_epochs):
        opt_G.zero_grad()
        loss, residual = compute_loss(generator, tf, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_G.step()
        sched_G.step()

        if epoch % adv_update_freq == 0:
            for _ in range(adv_steps):
                opt_A.zero_grad()
                loss_adv, _ = compute_loss(generator, tf, cfg)
                (-loss_adv).backward()
                torch.nn.utils.clip_grad_norm_(tf.parameters(), 1.0)
                opt_A.step()

        history['loss'].append(loss.item())
        history['residual_norm'].append(torch.norm(residual).item())

        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1}/{cfg.n_epochs} | Loss: {loss.item():.6f}")

    return generator, tf, history


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(generator, particle_traj, history, cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    dims_to_plot = [0, 1, 2]
    n_snaps = len(SNAPSHOT_TIMES)
    n_dims  = len(dims_to_plot)

    fig, axes = plt.subplots(n_dims, n_snaps, figsize=(4 * n_snaps, 3.5 * n_dims))

    for j, t_val in enumerate(SNAPSHOT_TIMES):
        with torch.no_grad():
            x0 = sample_ic(5000, cfg)
            r  = torch.randn(5000, cfg.d_base, device=device)
            t_ = torch.full((5000, 1), t_val, device=device)
            x_nn = generator(t_, x0, r).cpu().numpy()

        t_key = min(particle_traj.keys(), key=lambda k: abs(k - t_val))
        x_part = particle_traj[t_key]

        for i, dim_idx in enumerate(dims_to_plot):
            ax = axes[i, j]
            ax.hist(x_nn[:, dim_idx], bins=50, density=True,
                    alpha=0.6, color='steelblue', label='WANPM')
            ax.hist(x_part[:, dim_idx], bins=50, density=True,
                    alpha=0.6, color='darkorange', label='Particle')
            # Robust stats
            med_nn   = np.median(x_nn[:, dim_idx])
            med_part = np.median(x_part[:, dim_idx])
            iqr_nn   = np.percentile(x_nn[:, dim_idx], 75) - np.percentile(x_nn[:, dim_idx], 25)
            iqr_part = np.percentile(x_part[:, dim_idx], 75) - np.percentile(x_part[:, dim_idx], 25)
            ax.set_title(f't={t_val}, dim{dim_idx}\nmed:NN={med_nn:.2f}/P={med_part:.2f} '
                         f'IQR:NN={iqr_nn:.2f}/P={iqr_part:.2f}', fontsize=7)
            ax.set_xlabel(f'x_{dim_idx}'); ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)

    plt.suptitle(f'Fractional {cfg.dim}D OU  (α={cfg.alpha})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, f'nd_ou_{cfg.dim}d_marginals.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history['loss'], color='steelblue', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f'Training Loss ({cfg.dim}D)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'training_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {cfg.output_dir}")


# ============================================================================
# RUN ONE CONFIG
# ============================================================================
def run_config(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Running {cfg.dim}D OU  (α={cfg.alpha})")
    print(f"{'='*70}")

    particle_traj = run_particle_benchmark(cfg, SNAPSHOT_TIMES)
    generator, tf, history = train(cfg)
    plot_results(generator, particle_traj, history, cfg)

    torch.save({
        'generator_state_dict': generator.state_dict(),
        'hyperparameters': {
            'dim': cfg.dim, 'alpha': cfg.alpha, 'd_base': cfg.d_base,
            't_final': cfg.t_final, 'mu0': cfg.mu0,
        }
    }, os.path.join(cfg.output_dir, 'model.pt'))
    print(f"{cfg.dim}D model saved.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Fractional Higher-Dimensional OU  (Section 4.9 variant)")
    print("=" * 70)
    for cfg in CONFIGS:
        run_config(cfg)
    print("\nAll configs done.")


if __name__ == '__main__':
    main()
