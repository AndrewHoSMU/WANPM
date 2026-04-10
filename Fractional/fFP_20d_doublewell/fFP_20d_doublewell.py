import matplotlib; matplotlib.use('Agg')
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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
ALPHA     = 1.5
DIM       = 20
D_BASE    = 80
T_FINAL   = 1.0
N_EPOCHS  = 90_000
ANNEAL_START = 0
ANNEAL_END   = N_EPOCHS
K_TEST    = max(5000, 30 * DIM)
M_INTERIOR = max(5000, 30 * DIM)
M_INITIAL  = 2000

V_ALPHA = 1.0   # coefficient in V(x) = V_ALPHA * ||x-c+||^2 * ||x-c-||^2

LR_GENERATOR  = 1e-3 
LR_ADVERSARY  = 1e-2
ADV_UPDATE_FREQ      = 2
ADV_STEPS_PER_UPDATE = 1

IC_SIGMA = 0.5   # N(0, 0.5^2 * I_20)

SIGMA_COEFF_INITIAL = 10.0  # starting coefficient on fractional Laplacian
SIGMA_COEFF_FINAL   = 1.0   # target coefficient (physical equation)


SNAPSHOT_TIMES = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]


# ============================================================================
# CURRICULUM SCHEDULE
# ============================================================================
def get_sigma_schedule(epoch):
    """Cosine annealing of fractional Laplacian coefficient from SIGMA_COEFF_INITIAL to SIGMA_COEFF_FINAL."""
    if epoch < ANNEAL_START:
        return SIGMA_COEFF_INITIAL
    elif epoch < ANNEAL_END:
        progress = (epoch - ANNEAL_START) / (ANNEAL_END - ANNEAL_START)
        cosine_progress = 0.5 * (1 - np.cos(np.pi * progress))
        return SIGMA_COEFF_INITIAL + cosine_progress * (SIGMA_COEFF_FINAL - SIGMA_COEFF_INITIAL)
    else:
        return SIGMA_COEFF_FINAL


# ============================================================================
# POTENTIAL & DRIFT
# V(x) = V_ALPHA * ||x-c+||^2 * ||x-c-||^2
# c+ = +1 (all ones), c- = -1 (all neg-ones)
# b(x) = -grad V = -2*V_ALPHA * [(x-c+)||x-c-||^2 + (x-c-)||x-c+||^2]
# ============================================================================
def drift_field(x):
    """x: (M, DIM) -> b: (M, DIM)"""
    c_plus  = torch.ones(1, DIM, device=x.device)
    c_minus = -torch.ones(1, DIM, device=x.device)
    diff_p = x - c_plus   # (M, DIM)
    diff_m = x - c_minus  # (M, DIM)
    norm_p2 = (diff_p ** 2).sum(dim=1, keepdim=True)   # (M, 1)
    norm_m2 = (diff_m ** 2).sum(dim=1, keepdim=True)   # (M, 1)
    b = -2.0 * V_ALPHA * (diff_p * norm_m2 + diff_m * norm_p2)
    return b


# ============================================================================
# INITIAL CONDITION
# ============================================================================
def sample_ic(n):
    return IC_SIGMA * torch.randn(n, DIM, device=device)


# ============================================================================
# NETWORK
# ============================================================================
class PushforwardNetwork(nn.Module):
    def __init__(self, d_base, d_out, hidden_dims=[256, 128]):
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
        self.w     = nn.Parameter(torch.randn(k_test, d, device=device) * 1.0)
        self.kappa = nn.Parameter(torch.randn(k_test, 1, device=device) * 1.0)
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
        w_norm_alpha = torch.norm(self.w, dim=1) ** ALPHA
        return w_norm_alpha.unsqueeze(0) * self.f(t, x)


# ============================================================================
# LOSS
# ============================================================================
def compute_loss(generator, tf, sigma=1.0):
    x0_init = sample_ic(M_INITIAL)
    t0 = torch.zeros(M_INITIAL, 1, device=device)
    E_initial = tf.f(t0, x0_init).mean(dim=0)

    x0_T = sample_ic(M_INITIAL)
    r_T  = torch.randn(M_INITIAL, D_BASE, device=device)
    t_T  = torch.full((M_INITIAL, 1), T_FINAL, device=device)
    x_T  = generator(t_T, x0_T, r_T)
    E_terminal = tf.f(t_T, x_T).mean(dim=0)

    t_int  = torch.rand(M_INTERIOR, 1, device=device) * T_FINAL
    x0_int = sample_ic(M_INTERIOR)
    r_int  = torch.randn(M_INTERIOR, D_BASE, device=device)
    x_int  = generator(t_int, x0_int, r_int)

    frac_L  = tf.fractional_laplacian_f(t_int, x_int)
    grad_f  = tf.grad_f(t_int, x_int)
    b_vals  = drift_field(x_int)
    b_dot_g = (b_vals.unsqueeze(1) * grad_f).sum(dim=2)
    Lf = -sigma * frac_L + b_dot_g

    df_dt = tf.df_dt(t_int, x_int)
    E_interior = T_FINAL * (df_dt + Lf).mean(dim=0)

    residual = E_terminal - E_initial - E_interior
    loss = (residual ** 2).mean()
    return loss, residual


# ============================================================================
# TRAINING
# ============================================================================
def train():
    generator = PushforwardNetwork(D_BASE, DIM).to(device)
    tf        = PlaneWaveTestFunctions(K_TEST, DIM, device)

    opt_G  = torch.optim.Adam(generator.parameters(), lr=LR_GENERATOR)
    opt_A   = torch.optim.SGD(tf.parameters(), lr=LR_ADVERSARY)
    sched_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size=10000, gamma=0.5)

    history = {'loss': [], 'residual_norm': []}
    print(f"Training: N_EPOCHS={N_EPOCHS}, K_TEST={K_TEST}, ALPHA={ALPHA}, DIM={DIM}")

    for epoch in range(N_EPOCHS):
        current_sigma = get_sigma_schedule(epoch)

        opt_G.zero_grad()
        loss, residual = compute_loss(generator, tf, current_sigma)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_G.step()
        # sched_G.step()

        if epoch % ADV_UPDATE_FREQ == 0:
            for _ in range(ADV_STEPS_PER_UPDATE):
                opt_A.zero_grad()
                loss_adv, _ = compute_loss(generator, tf, current_sigma)
                (-loss_adv).backward()
                torch.nn.utils.clip_grad_norm_(tf.parameters(), 1.0)
                opt_A.step()

        history['loss'].append(loss.item())
        history['residual_norm'].append(torch.norm(residual).item())

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{N_EPOCHS} | Loss: {loss.item():.6f} | sigma_coeff: {current_sigma:.4f}")

    return generator, tf, history


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(generator, history):
    # 2D projections at each snapshot time
    proj_pairs = [(0, 1), (0, DIM - 1)]    # (x1,x2) and (x1, x20)
    n_snaps = len(SNAPSHOT_TIMES)
    n_proj  = len(proj_pairs)

    fig, axes = plt.subplots(n_proj, n_snaps, figsize=(4 * n_snaps, 4 * n_proj))

    for j, t_val in enumerate(SNAPSHOT_TIMES):
        with torch.no_grad():
            if t_val == 0.0:
                x_samples = sample_ic(3000).cpu().numpy()
            else:
                x0 = sample_ic(3000)
                r  = torch.randn(3000, D_BASE, device=device)
                t_ = torch.full((3000, 1), t_val, device=device)
                x_samples = generator(t_, x0, r).cpu().numpy()

        for i, (d1, d2) in enumerate(proj_pairs):
            ax = axes[i, j]
            ax.scatter(x_samples[:, d1], x_samples[:, d2], s=2, alpha=0.4, color='steelblue')
            ax.set_title(f't={t_val}\n(x{d1+1},x{d2+1})', fontsize=9)
            ax.set_xlabel(f'x{d1+1}'); ax.set_ylabel(f'x{d2+1}')
            ax.grid(True, alpha=0.3)

    plt.suptitle(f'Fractional 20D Double-Well  (α={ALPHA})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '20d_doublewell_projections.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Training loss
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history['loss'], color='steelblue', linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss (20D Double-Well)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Plots saved.")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("Fractional 20D Double-Well  (Section 4.10 variant)")
    print("=" * 70)

    generator, tf, history = train()
    plot_results(generator, history)

    torch.save({
        'generator_state_dict': generator.state_dict(),
        'hyperparameters': {
            'alpha': ALPHA, 'dim': DIM, 'd_base': D_BASE,
            't_final': T_FINAL, 'v_alpha': V_ALPHA,
        }
    }, os.path.join(OUTPUT_DIR, 'model.pt'))
    print("Model saved.")
    print("Done.")


if __name__ == '__main__':
    main()
