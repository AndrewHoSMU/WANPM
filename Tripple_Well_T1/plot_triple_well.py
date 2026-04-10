# """
# Triple-Well Potential Visualization Script
# ===========================================
#
# Loads trained model and finite volume solution to create visualizations.
# Can be run independently after training.
# """
#
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde
# from scipy.sparse import diags
# from scipy.sparse.linalg import spsolve
# import os
#
# # Check for required files first, but don't load yet
# def check_files_exist():
#     if not os.path.exists('results_triple_well/checkpoint.pth'):
#         raise FileNotFoundError("checkpoint.pth not found. Please run train_triple_well.py first.")
#
# # We'll load the checkpoint in the main() function to avoid import-time errors
# checkpoint = None
# params = None
#
# # These will be set when checkpoint is loaded
# DIM = None
# SIGMA_FINAL = None
# T_FINAL = None
# MU_1 = None
# MU_2 = None
# SIGMA_IC = None
# MIXTURE_WEIGHT = None
# D_BASE = None
# ANNEALING_START_EPOCH = None
# ANNEALING_END_EPOCH = None
# EPSILON = 1e-3
# X_MIN, X_MAX = -2.5, 2.5
#
# # Finite volume parameters
# N_X = 600
# N_T = 5000
#
# device = None  # Will be set in main()
#
# # Model and data (loaded in main())
# pushforward_net = None
# loss_log = None
# sigma_log = None
# lr_gen_log = None
# lr_test_log = None
# x_grid_fv = None
# times_fv = None
# rho_history_fv = None
# means_fv = None
# variances_fv = None
#
#
# # ============================================================================
# # UTILITY FUNCTIONS
# # ============================================================================
#
# def potential(x):
#     """Triple-well potential: V(x) = x²(x² - 1)²"""
#     return (x**2) * ((x**2 - 1)**2)
#
#
# def drift_function(x):
#     """Drift: b(x) = -dV/dx = -2x(x² - 1)(3x² - 1)"""
#     return -2 * x * (x**2 - 1) * (3*x**2 - 1)
#
#
# def sample_mixture_gaussian(n_samples):
#     """Sample from mixture of two Gaussians at x = ±0.5"""
#     assignments = torch.rand(n_samples, 1, device=device) < MIXTURE_WEIGHT
#     samples_1 = MU_1 + SIGMA_IC * torch.randn(n_samples, DIM, device=device)
#     samples_2 = MU_2 + SIGMA_IC * torch.randn(n_samples, DIM, device=device)
#     samples = torch.where(assignments, samples_1, samples_2)
#     return samples
#
#
# def mixture_gaussian_pdf(x):
#     """PDF of the mixture Gaussian initial condition"""
#     pdf_1 = (1 / (np.sqrt(2 * np.pi) * SIGMA_IC)) * np.exp(-0.5 * ((x - MU_1) / SIGMA_IC)**2)
#     pdf_2 = (1 / (np.sqrt(2 * np.pi) * SIGMA_IC)) * np.exp(-0.5 * ((x - MU_2) / SIGMA_IC)**2)
#     return MIXTURE_WEIGHT * pdf_1 + (1 - MIXTURE_WEIGHT) * pdf_2
#
#
# # ============================================================================
# # NEURAL NETWORK COMPONENTS
# # ============================================================================
#
# class PushforwardNetwork(nn.Module):
#     def __init__(self, d_base, d_output, hidden_dims=[128, 128, 128]):
#         super().__init__()
#         layers = []
#         input_dim = 1 + d_base
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(input_dim, hidden_dim))
#             layers.append(nn.Tanh())
#             input_dim = hidden_dim
#         layers.append(nn.Linear(input_dim, d_output))
#         self.network = nn.Sequential(*layers)
#
#     def forward(self, t, x_0, r):
#         t_r = torch.cat([t, r], dim=1)
#         delta = self.network(t_r)
#         return x_0 + torch.sqrt(t) * delta
#
#
# # ============================================================================
# # FINITE VOLUME METHOD
# # ============================================================================
#
# def run_finite_volume():
#     """Solve using fully implicit finite volume method for stability"""
#     print("\nRunning fully implicit finite volume solver...")
#
#     DT = T_FINAL / N_T
#     x_grid = np.linspace(X_MIN, X_MAX, N_X)
#     dx = x_grid[1] - x_grid[0]
#
#     # Check CFL condition
#     max_drift = np.max(np.abs(drift_function(x_grid)))
#     cfl_diffusion = SIGMA_FINAL**2 * DT / dx**2
#     cfl_advection = max_drift * DT / dx
#     print(f"  CFL (diffusion): {cfl_diffusion:.4f}")
#     print(f"  CFL (advection): {cfl_advection:.4f}")
#     print(f"  Max |drift|: {max_drift:.4f}")
#
#     # Initial condition - mixture Gaussian
#     rho = mixture_gaussian_pdf(x_grid)
#     rho = rho / np.trapezoid(rho, x_grid)  # Normalize
#
#     # Store history
#     rho_history = [rho.copy()]
#     times = [0.0]
#     means = [np.trapezoid(x_grid * rho, x_grid)]
#     variances = [np.trapezoid(x_grid**2 * rho, x_grid) - means[0]**2]
#
#     # Precompute drift values
#     b_vals = drift_function(x_grid)
#
#     # Time stepping with fully implicit method
#     save_interval = max(1, N_T // 100)
#
#     for n in range(N_T):
#         # Build fully implicit system for ρ^{n+1}
#         diag_main = np.ones(N_X)
#         diag_lower = np.zeros(N_X - 1)
#         diag_upper = np.zeros(N_X - 1)
#         rhs = rho.copy()
#
#         # Interior points (fully implicit)
#         for i in range(1, N_X - 1):
#             # Diffusion contribution (implicit)
#             diff_coeff = SIGMA_FINAL**2 / 2 * DT / dx**2
#             diag_main[i] += 2 * diff_coeff
#             diag_lower[i-1] -= diff_coeff
#             diag_upper[i] -= diff_coeff
#
#             # Drift contribution (implicit upwind)
#             if b_vals[i] >= 0:
#                 drift_coeff = b_vals[i] * DT / dx
#                 diag_main[i] += drift_coeff
#                 diag_lower[i-1] -= drift_coeff
#             else:
#                 drift_coeff = -b_vals[i] * DT / dx
#                 diag_main[i] += drift_coeff
#                 diag_upper[i] -= drift_coeff
#
#         # Boundary conditions (Neumann/zero flux)
#         diag_main[0] = 1.0
#         diag_upper[0] = -1.0
#         rhs[0] = 0.0
#
#         diag_main[-1] = 1.0
#         diag_lower[-1] = -1.0
#         rhs[-1] = 0.0
#
#         # Build sparse matrix and solve
#         A = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1],
#                   shape=(N_X, N_X), format='csr')
#
#         rho = spsolve(A, rhs)
#
#         # Ensure non-negativity and normalize
#         rho = np.maximum(rho, 0)
#         rho = rho / np.trapezoid(rho, x_grid)
#
#         # Save periodically
#         if n % save_interval == 0 or n == N_T - 1:
#             rho_history.append(rho.copy())
#             times.append((n + 1) * DT)
#             means.append(np.trapezoid(x_grid * rho, x_grid))
#             variances.append(np.trapezoid(x_grid**2 * rho, x_grid) - means[-1]**2)
#
#     print(f"Finite volume solver completed: {len(times)} snapshots saved")
#     return x_grid, np.array(times), np.array(rho_history), np.array(means), np.array(variances)
#
#
# # ============================================================================
# # SAMPLING FUNCTIONS
# # ============================================================================
#
# def sample_from_neural_net(t_val, n_samples=10000):
#     """Generate samples from the neural network at time t"""
#     with torch.no_grad():
#         t = t_val * torch.ones(n_samples, 1, device=device)
#         r = torch.randn(n_samples, D_BASE, device=device)
#         x_0 = sample_mixture_gaussian(n_samples)
#         x = pushforward_net(t, x_0, r)
#     return x.cpu().numpy().flatten()
#
#
# def compute_pdf_from_samples(samples, x_grid, bandwidth=0.05):
#     """Compute PDF from samples using KDE"""
#     kde = gaussian_kde(samples, bw_method=bandwidth)
#     return kde(x_grid)
#
#
# # ============================================================================
# # VISUALIZATION FUNCTIONS
# # ============================================================================
#
# def plot_potential():
#     """Visualize triple-well potential and drift"""
#     x = np.linspace(-2, 2, 1000)
#     V = potential(x)
#     b = drift_function(x)
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#
#     # Potential
#     ax1.plot(x, V, 'b-', linewidth=2.5)
#     ax1.axvline(x=-1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Minima')
#     ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
#     ax1.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
#     ax1.axvline(x=-0.5, color='green', linestyle=':', linewidth=2, alpha=0.7, label='IC centers')
#     ax1.axvline(x=0.5, color='green', linestyle=':', linewidth=2, alpha=0.7)
#     ax1.set_xlabel('x', fontsize=12)
#     ax1.set_ylabel('V(x)', fontsize=12)
#     ax1.set_title('Triple-Well Potential: V(x) = x²(x² - 1)²', fontsize=13)
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()
#
#     # Drift
#     ax2.plot(x, b, 'r-', linewidth=2.5)
#     ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
#     ax2.axvline(x=-1, color='red', linestyle='--', linewidth=2, alpha=0.7)
#     ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
#     ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
#     ax2.set_xlabel('x', fontsize=12)
#     ax2.set_ylabel('b(x) = -dV/dx', fontsize=12)
#     ax2.set_title('Drift Function', fontsize=13)
#     ax2.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig('results_triple_well/potential_drift.png', dpi=150, bbox_inches='tight')
#     print("Saved: results_triple_well/potential_drift.png")
#     plt.close()
#
#
# def plot_initial_condition():
#     """Plot the mixture Gaussian initial condition"""
#     x = np.linspace(-2, 2, 1000)
#     pdf = mixture_gaussian_pdf(x)
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(x, pdf, 'b-', linewidth=2.5, label='Mixture Gaussian IC')
#     plt.axvline(x=-0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Component centers')
#     plt.axvline(x=0.5, color='green', linestyle='--', linewidth=2, alpha=0.7)
#     plt.axvline(x=-1, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Potential minima')
#     plt.axvline(x=0, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
#     plt.axvline(x=1, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
#     plt.xlabel('x', fontsize=12)
#     plt.ylabel('Probability Density', fontsize=12)
#     plt.title('Initial Condition: Mixture of Gaussians at x = ±0.5', fontsize=13)
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('results_triple_well/initial_condition.png', dpi=150, bbox_inches='tight')
#     print("Saved: results_triple_well/initial_condition.png")
#     plt.close()
#
#
# def plot_training_diagnostics():
#     """Plot training curves"""
#     epochs = np.arange(len(loss_log))
#
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#
#     # Loss
#     axes[0, 0].semilogy(epochs, loss_log, 'b-', linewidth=1.5)
#     axes[0, 0].axvline(x=ANNEALING_START_EPOCH, color='red', linestyle='--',
#                         alpha=0.7, label='Start annealing')
#     axes[0, 0].axvline(x=ANNEALING_END_EPOCH, color='green', linestyle='--',
#                         alpha=0.7, label='Start fine-tuning')
#     axes[0, 0].set_xlabel('Epoch', fontsize=11)
#     axes[0, 0].set_ylabel('Loss', fontsize=11)
#     axes[0, 0].set_title('Training Loss', fontsize=12)
#     axes[0, 0].grid(True, alpha=0.3)
#     axes[0, 0].legend()
#
#     # Sigma
#     axes[0, 1].plot(epochs, sigma_log, 'r-', linewidth=2)
#     axes[0, 1].axhline(y=SIGMA_FINAL, color='green', linestyle='--',
#                         alpha=0.7, label=f'Target σ = {SIGMA_FINAL}')
#     axes[0, 1].set_xlabel('Epoch', fontsize=11)
#     axes[0, 1].set_ylabel('σ', fontsize=11)
#     axes[0, 1].set_title('Diffusion Coefficient Schedule', fontsize=12)
#     axes[0, 1].grid(True, alpha=0.3)
#     axes[0, 1].legend()
#
#     # Learning rates
#     axes[1, 0].semilogy(epochs, lr_gen_log, 'b-', linewidth=1.5, label='Generator')
#     axes[1, 0].semilogy(epochs, lr_test_log, 'r-', linewidth=1.5, label='Test functions')
#     axes[1, 0].set_xlabel('Epoch', fontsize=11)
#     axes[1, 0].set_ylabel('Learning Rate', fontsize=11)
#     axes[1, 0].set_title('Learning Rate Schedules', fontsize=12)
#     axes[1, 0].grid(True, alpha=0.3)
#     axes[1, 0].legend()
#
#     # Loss (zoomed to later epochs)
#     start_zoom = ANNEALING_END_EPOCH
#     axes[1, 1].semilogy(epochs[start_zoom:], loss_log[start_zoom:], 'b-', linewidth=1.5)
#     axes[1, 1].set_xlabel('Epoch', fontsize=11)
#     axes[1, 1].set_ylabel('Loss', fontsize=11)
#     axes[1, 1].set_title(f'Loss During Fine-tuning (epochs {start_zoom}+)', fontsize=12)
#     axes[1, 1].grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig('results_triple_well/training_diagnostics.png', dpi=150, bbox_inches='tight')
#     print("Saved: results_triple_well/training_diagnostics.png")
#     plt.close()
#
#
# def plot_pdf_comparison():
#     """Compare NN and FV solutions at different times"""
#     n_samples = 20000
#     time_points = [0.3, 0.75, 1.5, 2.5, 3.0]
#
#     x_grid = np.linspace(X_MIN, X_MAX, 300)
#
#     fig, axes = plt.subplots(2, len(time_points), figsize=(5*len(time_points), 10))
#
#     comparison_results = {'times': [], 'l2_errors': []}
#
#     for idx, t_val in enumerate(time_points):
#         # Get FV solution
#         t_idx_fv = np.argmin(np.abs(times_fv - t_val))
#         rho_fv = np.interp(x_grid, x_grid_fv, rho_history_fv[t_idx_fv])
#         rho_fv = rho_fv / np.trapezoid(rho_fv, x_grid)
#
#         # Get NN solution
#         samples_nn = sample_from_neural_net(t_val, n_samples)
#         rho_nn = compute_pdf_from_samples(samples_nn, x_grid, bandwidth=0.04)
#         rho_nn = rho_nn / np.trapezoid(rho_nn, x_grid)
#
#         # Compute error
#         l2_error = np.sqrt(np.trapezoid((rho_nn - rho_fv)**2, x_grid))
#         comparison_results['times'].append(t_val)
#         comparison_results['l2_errors'].append(l2_error)
#
#         # Plot PDFs
#         axes[0, idx].hist(samples_nn, bins=80, density=True, alpha=0.5,
#                          label='Neural Net', color='blue', edgecolor='black')
#         axes[0, idx].plot(x_grid, rho_fv, 'r-', linewidth=2.5, label='FV Solution')
#         axes[0, idx].axvline(x=-1, color='gray', linestyle=':', alpha=0.5)
#         axes[0, idx].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
#         axes[0, idx].axvline(x=1, color='gray', linestyle=':', alpha=0.5)
#         axes[0, idx].set_xlabel('x', fontsize=11)
#         axes[0, idx].set_ylabel('Probability Density', fontsize=11)
#         axes[0, idx].set_title(f't = {t_val:.2f}\nL² = {l2_error:.4f}', fontsize=12)
#         axes[0, idx].legend()
#         axes[0, idx].grid(True, alpha=0.3)
#
#         # Plot absolute error
#         axes[1, idx].plot(x_grid, np.abs(rho_nn - rho_fv), 'g-', linewidth=2)
#         axes[1, idx].fill_between(x_grid, 0, np.abs(rho_nn - rho_fv), alpha=0.3, color='green')
#         axes[1, idx].set_xlabel('x', fontsize=11)
#         axes[1, idx].set_ylabel('|ρ_NN - ρ_FV|', fontsize=11)
#         axes[1, idx].set_title('Absolute Error', fontsize=12)
#         axes[1, idx].grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig('results_triple_well/pdf_comparison.png', dpi=150, bbox_inches='tight')
#     print("Saved: results_triple_well/pdf_comparison.png")
#     plt.close()
#
#     return comparison_results
#
#
# def plot_time_evolution():
#     """Plot time evolution heatmap"""
#     n_times = 30
#     times = np.linspace(EPSILON, T_FINAL, n_times)
#     x_range = np.linspace(X_MIN, X_MAX, 150)
#     density_grid = np.zeros((len(x_range), n_times))
#
#     with torch.no_grad():
#         for t_idx, t_val in enumerate(times):
#             samples = sample_from_neural_net(t_val, n_samples=5000)
#             hist, _ = np.histogram(samples, bins=x_range, density=True)
#             density_grid[:len(hist), t_idx] = hist
#
#     fig, ax = plt.subplots(figsize=(14, 7))
#     extent = [times[0], times[-1], x_range[0], x_range[-1]]
#     im = ax.imshow(density_grid, aspect='auto', origin='lower', extent=extent,
#                    cmap='viridis', interpolation='bilinear')
#
#     ax.axhline(y=-1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Potential minima')
#     ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
#     ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
#     ax.axhline(y=-0.5, color='white', linestyle=':', linewidth=1.5, alpha=0.8, label='IC centers')
#     ax.axhline(y=0.5, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
#     ax.set_xlabel('Time', fontsize=12)
#     ax.set_ylabel('x', fontsize=12)
#     ax.set_title('Probability Density Evolution: Two Peaks → Three Peaks', fontsize=14)
#     plt.colorbar(im, ax=ax, label='Density')
#     ax.legend()
#
#     plt.tight_layout()
#     plt.savefig('results_triple_well/time_evolution.png', dpi=150, bbox_inches='tight')
#     print("Saved: results_triple_well/time_evolution.png")
#     plt.close()
#
#
# # ============================================================================
# # MAIN
# # ============================================================================
#
# def main():
#     global checkpoint, params, DIM, SIGMA_FINAL, T_FINAL, MU_1, MU_2, SIGMA_IC
#     global MIXTURE_WEIGHT, D_BASE, ANNEALING_START_EPOCH, ANNEALING_END_EPOCH, device
#     global pushforward_net, loss_log, sigma_log, lr_gen_log, lr_test_log
#     global x_grid_fv, times_fv, rho_history_fv, means_fv, variances_fv
#
#     print("=" * 70)
#     print("Triple-Well Potential Visualization")
#     print("=" * 70)
#
#     # Check files exist
#     check_files_exist()
#
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"\nUsing device: {device}")
#
#     # Load checkpoint with weights_only=False (safe since we trust our own file)
#     print("Loading trained model...")
#     checkpoint = torch.load('results_triple_well/checkpoint.pth', map_location='cpu', weights_only=False)
#     params = checkpoint['hyperparameters']
#
#     # Extract parameters
#     DIM = params['DIM']
#     SIGMA_FINAL = params['SIGMA_FINAL']
#     T_FINAL = params['T_FINAL']
#     MU_1 = params['MU_1']
#     MU_2 = params['MU_2']
#     SIGMA_IC = params['SIGMA_IC']
#     MIXTURE_WEIGHT = params['MIXTURE_WEIGHT']
#     D_BASE = params['D_BASE']
#     ANNEALING_START_EPOCH = params['ANNEALING_START_EPOCH']
#     ANNEALING_END_EPOCH = params['ANNEALING_END_EPOCH']
#
#     # Load model
#     pushforward_net = PushforwardNetwork(D_BASE, DIM).to(device)
#     pushforward_net.load_state_dict(checkpoint['pushforward_net_state_dict'])
#     pushforward_net.eval()
#
#     loss_log = checkpoint['loss_log']
#     sigma_log = checkpoint['sigma_log']
#     lr_gen_log = checkpoint['lr_gen_log']
#     lr_test_log = checkpoint['lr_test_log']
#
#     # Run finite volume solver for comparison
#     x_grid_fv, times_fv, rho_history_fv, means_fv, variances_fv = run_finite_volume()
#
#     print("\n1. Plotting potential and drift...")
#     plot_potential()
#
#     print("\n2. Plotting initial condition...")
#     plot_initial_condition()
#
#     print("\n3. Plotting training diagnostics...")
#     plot_training_diagnostics()
#
#     print("\n4. Comparing NN vs FV solutions...")
#     comparison_results = plot_pdf_comparison()
#
#     print("\n5. Plotting time evolution heatmap...")
#     plot_time_evolution()
#
#     print("\n" + "=" * 70)
#     print("VISUALIZATION SUMMARY")
#     print("=" * 70)
#     print(f"\nComparison (NN vs FV):")
#     print(f"  Average L² error: {np.mean(comparison_results['l2_errors']):.6f}")
#     print(f"  L² errors at times {comparison_results['times']}:")
#     for t, err in zip(comparison_results['times'], comparison_results['l2_errors']):
#         print(f"    t = {t:.2f}: L² = {err:.6f}")
#
#     print("\n" + "=" * 70)
#     print("✓ All visualizations complete!")
#     print("\nGenerated plots in 'results_triple_well/':")
#     print("  1. potential_drift.png - Triple-well potential and drift")
#     print("  2. initial_condition.png - Mixture Gaussian IC")
#     print("  3. training_diagnostics.png - Loss, σ, and learning rates")
#     print("  4. pdf_comparison.png - NN vs FV PDFs at different times")
#     print("  5. time_evolution.png - Heatmap of density evolution")
#     print("=" * 70)
#
#
# if __name__ == '__main__':
#     main()

"""
Triple-Well Potential Visualization Script
===========================================

Loads trained model and finite volume solution to create visualizations.
Can be run independently after training.
"""

"""
Triple-Well Potential Visualization Script
===========================================

Loads trained model and finite volume solution to create visualizations.
Can be run independently after training.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import os

# Check for required files first, but don't load yet
def check_files_exist():
    if not os.path.exists('results_triple_well/checkpoint.pth'):
        raise FileNotFoundError("checkpoint.pth not found. Please run train_triple_well.py first.")

# We'll load the checkpoint in the main() function to avoid import-time errors
checkpoint = None
params = None

# These will be set when checkpoint is loaded
DIM = None
SIGMA_FINAL = None
T_FINAL = None
MU_1 = None
MU_2 = None
SIGMA_IC = None
MIXTURE_WEIGHT = None
D_BASE = None
ANNEALING_START_EPOCH = None
ANNEALING_END_EPOCH = None
EPSILON = 1e-3
X_MIN, X_MAX = -2.5, 2.5

# Finite volume parameters
N_X = 600
N_T = 5000

device = None  # Will be set in main()

# Model and data (loaded in main())
pushforward_net = None
loss_log = None
sigma_log = None
lr_gen_log = None
lr_test_log = None
x_grid_fv = None
times_fv = None
rho_history_fv = None
means_fv = None
variances_fv = None


# ============================================================================
# UTILITY FUNCTIONS
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


# ============================================================================
# FINITE VOLUME METHOD
# ============================================================================

def run_finite_volume():
    """Solve using Scharfetter-Gummel finite volume method (fully implicit)"""
    print("\nRunning Scharfetter-Gummel finite volume solver...")

    DT = T_FINAL / N_T
    x_grid = np.linspace(X_MIN, X_MAX, N_X)
    dx = x_grid[1] - x_grid[0]

    # Check CFL condition
    max_drift = np.max(np.abs(drift_function(x_grid)))
    cfl_diffusion = SIGMA_FINAL**2 * DT / dx**2
    cfl_advection = max_drift * DT / dx
    print(f"  CFL (diffusion): {cfl_diffusion:.4f}")
    print(f"  CFL (advection): {cfl_advection:.4f}")
    print(f"  Max |drift|: {max_drift:.4f}")

    # Initial condition - mixture Gaussian
    rho = mixture_gaussian_pdf(x_grid)
    rho = rho / np.trapezoid(rho, x_grid)  # Normalize

    # Store history
    rho_history = [rho.copy()]
    times = [0.0]
    means = [np.trapezoid(x_grid * rho, x_grid)]
    variances = [np.trapezoid(x_grid**2 * rho, x_grid) - means[0]**2]

    # Compute M(x) = exp(-U(x)) where U(x) = ∫(b(x)/a)dx
    # For our equation: b(x) = -dV/dx, a = σ²/2
    # U(x) = ∫(-dV/dx)/(σ²/2) dx = -2V(x)/σ²
    # Therefore: M(x) = exp(2V(x)/σ²)
    V_vals = potential(x_grid)
    log_M_vals = -2 * V_vals / SIGMA_FINAL**2  # NEGATIVE sign is critical!

    print(f"  log(M) range: [{log_M_vals.min():.2f}, {log_M_vals.max():.2f}]")

    # Compute M_{i+1/2}/M_i and M_{i+1/2}/M_{i+1} using harmonic mean in log-space
    # Harmonic mean: M_{i+1/2} = 2*M_i*M_{i+1}/(M_i + M_{i+1})
    # So: M_{i+1/2}/M_i = 2/(1 + M_i/M_{i+1}) = 2/(1 + exp(log_M_i - log_M_{i+1}))

    w_left = np.zeros(N_X - 1)   # M_{i+1/2}/M_i
    w_right = np.zeros(N_X - 1)  # M_{i+1/2}/M_{i+1}

    for i in range(N_X - 1):
        delta_log_M = log_M_vals[i] - log_M_vals[i+1]
        if abs(delta_log_M) < 1e-10:
            w_left[i] = 1.0
            w_right[i] = 1.0
        else:
            w_left[i] = 2.0 / (1.0 + np.exp(delta_log_M))
            w_right[i] = 2.0 / (1.0 + np.exp(-delta_log_M))

    # Time stepping with fully implicit method
    save_interval = max(1, N_T // 100)

    for n in range(N_T):
        # Build fully implicit system for ρ^{n+1}
        #
        # The Scharfetter-Gummel flux is:
        # F_{i+1/2} = -(σ²/2) * M_{i+1/2} * (1/Δx) * [ρ_{i+1}/M_{i+1} - ρ_i/M_i]
        #
        # This can be rewritten as:
        # F_{i+1/2} = -(σ²/2Δx) * [(M_{i+1/2}/M_i)*ρ_i - (M_{i+1/2}/M_{i+1})*ρ_{i+1}]
        #           = -(σ²/2Δx) * [w_left[i]*ρ_i - w_right[i]*ρ_{i+1}]

        diag_main = np.ones(N_X)
        diag_lower = np.zeros(N_X - 1)
        diag_upper = np.zeros(N_X - 1)
        rhs = rho.copy()

        # Interior points (1 to N_X-2)
        for i in range(1, N_X - 1):
            # Update: ρ^{n+1}_i + (Δt/Δx)[F_{i+1/2}^{n+1} - F_{i-1/2}^{n+1}] = ρ^n_i

            # F_{i+1/2} = -(σ²/2Δx)[w_left[i]*ρ_i - w_right[i]*ρ_{i+1}]
            # F_{i-1/2} = -(σ²/2Δx)[w_left[i-1]*ρ_{i-1} - w_right[i-1]*ρ_i]

            # Δt/Δx * F_{i+1/2} = -(σ²Δt/2Δx²)[w_left[i]*ρ_i - w_right[i]*ρ_{i+1}]
            # Δt/Δx * F_{i-1/2} = -(σ²Δt/2Δx²)[w_left[i-1]*ρ_{i-1} - w_right[i-1]*ρ_i]

            # ρ_i + (Δt/Δx)[F_{i+1/2} - F_{i-1/2}] = ρ^n_i
            # ρ_i - (σ²Δt/2Δx²)[w_left[i]*ρ_i - w_right[i]*ρ_{i+1}]
            #     + (σ²Δt/2Δx²)[w_left[i-1]*ρ_{i-1} - w_right[i-1]*ρ_i] = ρ^n_i
            # ρ_i[1 + (σ²Δt/2Δx²)(w_left[i] + w_right[i-1])]
            #     - (σ²Δt/2Δx²)w_right[i]*ρ_{i+1} - (σ²Δt/2Δx²)w_left[i-1]*ρ_{i-1} = ρ^n_i

            coeff = (SIGMA_FINAL**2 / 2) * DT / dx**2

            diag_main[i] += coeff * (w_left[i] + w_right[i-1])
            diag_upper[i] -= coeff * w_right[i]
            diag_lower[i-1] -= coeff * w_left[i-1]

        # Boundary conditions: Zero flux means F = 0 at boundaries
        # F_{1/2} = -(σ²/2) M_{1/2} (1/Δx)[ρ_1/M_1 - ρ_0/M_0] = 0
        # This gives: ρ_1/M_1 = ρ_0/M_0, or ρ_1 = ρ_0 * (M_1/M_0)
        #
        # In the linear system, this means we use the no-flux condition:
        # ρ_0 - (M_0/M_1)*ρ_1 = 0

        # Left boundary (i=0): enforce ρ_0/M_0 = ρ_1/M_1
        # Using: ρ_0 * M_1 - ρ_1 * M_0 = 0
        # In log-space: ρ_0 * exp(log_M_1 - log_M_0) - ρ_1 = 0
        ratio_left = np.exp(log_M_vals[1] - log_M_vals[0])
        diag_main[0] = ratio_left
        diag_upper[0] = -1.0
        rhs[0] = 0.0

        # Right boundary (i=N_X-1): enforce ρ_{N-1}/M_{N-1} = ρ_{N-2}/M_{N-2}
        # Using: ρ_{N-1} * M_{N-2} - ρ_{N-2} * M_{N-1} = 0
        ratio_right = np.exp(log_M_vals[N_X-2] - log_M_vals[N_X-1])
        diag_main[N_X-1] = 1.0
        diag_lower[N_X-2] = -ratio_right
        rhs[N_X-1] = 0.0

        # Build sparse matrix and solve
        A = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1],
                  shape=(N_X, N_X), format='csr')

        rho = spsolve(A, rhs)

        # Ensure non-negativity and normalize
        rho = np.maximum(rho, 0)
        total_mass = np.trapezoid(rho, x_grid)
        if total_mass > 1e-10:
            rho = rho / total_mass

        # Save periodically
        if n % save_interval == 0 or n == N_T - 1:
            rho_history.append(rho.copy())
            times.append((n + 1) * DT)
            means.append(np.trapezoid(x_grid * rho, x_grid))
            variances.append(np.trapezoid(x_grid**2 * rho, x_grid) - means[-1]**2)

    print(f"Scharfetter-Gummel solver completed: {len(times)} snapshots saved")
    return x_grid, np.array(times), np.array(rho_history), np.array(means), np.array(variances)


# ============================================================================
# SAMPLING FUNCTIONS
# ============================================================================

def sample_from_neural_net(t_val, n_samples=10000):
    """Generate samples from the neural network at time t"""
    with torch.no_grad():
        t = t_val * torch.ones(n_samples, 1, device=device)
        r = torch.randn(n_samples, D_BASE, device=device)
        x_0 = sample_mixture_gaussian(n_samples)
        x = pushforward_net(t, x_0, r)
    return x.cpu().numpy().flatten()


def compute_pdf_from_samples(samples, x_grid, bandwidth=0.05):
    """Compute PDF from samples using KDE"""
    kde = gaussian_kde(samples, bw_method=bandwidth)
    return kde(x_grid)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_potential():
    """Visualize triple-well potential and drift"""
    x = np.linspace(-2, 2, 1000)
    V = potential(x)
    b = drift_function(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Potential
    ax1.plot(x, V, 'b-', linewidth=2.5)
    ax1.axvline(x=-1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Minima')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(x=-0.5, color='green', linestyle=':', linewidth=2, alpha=0.7, label='IC centers')
    ax1.axvline(x=0.5, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('V(x)', fontsize=12)
    ax1.set_title('Triple-Well Potential: V(x) = x²(x² - 1)²', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Drift
    ax2.plot(x, b, 'r-', linewidth=2.5)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=-1, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('b(x) = -dV/dx', fontsize=12)
    ax2.set_title('Drift Function', fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_triple_well/potential_drift.png', dpi=150, bbox_inches='tight')
    print("Saved: results_triple_well/potential_drift.png")
    plt.close()


def plot_initial_condition():
    """Plot the mixture Gaussian initial condition"""
    x = np.linspace(-2, 2, 1000)
    pdf = mixture_gaussian_pdf(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, 'b-', linewidth=2.5, label='Mixture Gaussian IC')
    plt.axvline(x=-0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Component centers')
    plt.axvline(x=0.5, color='green', linestyle='--', linewidth=2, alpha=0.7)
    plt.axvline(x=-1, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='Potential minima')
    plt.axvline(x=0, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    plt.axvline(x=1, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Initial Condition: Mixture of Gaussians at x = ±0.5', fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results_triple_well/initial_condition.png', dpi=150, bbox_inches='tight')
    print("Saved: results_triple_well/initial_condition.png")
    plt.close()


def plot_training_diagnostics():
    """Plot training curves"""
    epochs = np.arange(len(loss_log))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].semilogy(epochs, loss_log, 'b-', linewidth=1.5)
    axes[0, 0].axvline(x=ANNEALING_START_EPOCH, color='red', linestyle='--',
                        alpha=0.7, label='Start annealing')
    axes[0, 0].axvline(x=ANNEALING_END_EPOCH, color='green', linestyle='--',
                        alpha=0.7, label='Start fine-tuning')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training Loss', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Sigma
    axes[0, 1].plot(epochs, sigma_log, 'r-', linewidth=2)
    axes[0, 1].axhline(y=SIGMA_FINAL, color='green', linestyle='--',
                        alpha=0.7, label=f'Target σ = {SIGMA_FINAL}')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('σ', fontsize=11)
    axes[0, 1].set_title('Diffusion Coefficient Schedule', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Learning rates
    axes[1, 0].semilogy(epochs, lr_gen_log, 'b-', linewidth=1.5, label='Generator')
    axes[1, 0].semilogy(epochs, lr_test_log, 'r-', linewidth=1.5, label='Test functions')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=11)
    axes[1, 0].set_title('Learning Rate Schedules', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Loss (zoomed to later epochs)
    start_zoom = ANNEALING_END_EPOCH
    axes[1, 1].semilogy(epochs[start_zoom:], loss_log[start_zoom:], 'b-', linewidth=1.5)
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Loss', fontsize=11)
    axes[1, 1].set_title(f'Loss During Fine-tuning (epochs {start_zoom}+)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_triple_well/training_diagnostics.png', dpi=150, bbox_inches='tight')
    print("Saved: results_triple_well/training_diagnostics.png")
    plt.close()


def plot_pdf_comparison():
    """Compare NN and FV solutions at different times"""
    n_samples = 20000
    time_points = [0.1 * T_FINAL, 0.2 * T_FINAL, 0.5 * T_FINAL, 0.8 * T_FINAL, 1.0 * T_FINAL]

    x_grid = np.linspace(X_MIN, X_MAX, 300)

    fig, axes = plt.subplots(2, len(time_points), figsize=(5*len(time_points), 10))

    comparison_results = {'times': [], 'l2_errors': []}

    for idx, t_val in enumerate(time_points):
        # Get FV solution
        t_idx_fv = np.argmin(np.abs(times_fv - t_val))
        rho_fv = np.interp(x_grid, x_grid_fv, rho_history_fv[t_idx_fv])
        rho_fv = rho_fv / np.trapezoid(rho_fv, x_grid)

        # Get NN solution
        samples_nn = sample_from_neural_net(t_val, n_samples)
        rho_nn = compute_pdf_from_samples(samples_nn, x_grid, bandwidth=0.04)
        rho_nn = rho_nn / np.trapezoid(rho_nn, x_grid)

        # Compute error
        l2_error = np.sqrt(np.trapezoid((rho_nn - rho_fv)**2, x_grid))
        comparison_results['times'].append(t_val)
        comparison_results['l2_errors'].append(l2_error)

        # Plot PDFs
        axes[0, idx].hist(samples_nn, bins=80, density=True, alpha=0.5,
                         label='Neural Net', color='blue', edgecolor='black')
        axes[0, idx].plot(x_grid, rho_fv, 'r-', linewidth=2.5, label='FV Solution')
        axes[0, idx].axvline(x=-1, color='gray', linestyle=':', alpha=0.5)
        axes[0, idx].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        axes[0, idx].axvline(x=1, color='gray', linestyle=':', alpha=0.5)
        axes[0, idx].set_xlabel('x', fontsize=11)
        axes[0, idx].set_ylabel('Probability Density', fontsize=11)
        axes[0, idx].set_title(f't = {t_val:.2f}\nL² = {l2_error:.4f}', fontsize=12)
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)

        # Plot absolute error
        axes[1, idx].plot(x_grid, np.abs(rho_nn - rho_fv), 'g-', linewidth=2)
        axes[1, idx].fill_between(x_grid, 0, np.abs(rho_nn - rho_fv), alpha=0.3, color='green')
        axes[1, idx].set_xlabel('x', fontsize=11)
        axes[1, idx].set_ylabel('|ρ_NN - ρ_FV|', fontsize=11)
        axes[1, idx].set_title('Absolute Error', fontsize=12)
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_triple_well/pdf_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: results_triple_well/pdf_comparison.png")
    plt.close()

    return comparison_results


def plot_time_evolution():
    """Plot time evolution heatmap"""
    n_times = 30
    times = np.linspace(EPSILON, T_FINAL, n_times)
    x_range = np.linspace(X_MIN, X_MAX, 150)
    density_grid = np.zeros((len(x_range), n_times))

    with torch.no_grad():
        for t_idx, t_val in enumerate(times):
            samples = sample_from_neural_net(t_val, n_samples=5000)
            hist, _ = np.histogram(samples, bins=x_range, density=True)
            density_grid[:len(hist), t_idx] = hist

    fig, ax = plt.subplots(figsize=(14, 7))
    extent = [times[0], times[-1], x_range[0], x_range[-1]]
    im = ax.imshow(density_grid, aspect='auto', origin='lower', extent=extent,
                   cmap='viridis', interpolation='bilinear')

    ax.axhline(y=-1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Potential minima')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(y=-0.5, color='white', linestyle=':', linewidth=1.5, alpha=0.8, label='IC centers')
    ax.axhline(y=0.5, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('x', fontsize=12)
    ax.set_title('Probability Density Evolution: Two Peaks → Three Peaks', fontsize=14)
    plt.colorbar(im, ax=ax, label='Density')
    ax.legend()

    plt.tight_layout()
    plt.savefig('results_triple_well/time_evolution.png', dpi=150, bbox_inches='tight')
    print("Saved: results_triple_well/time_evolution.png")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    global checkpoint, params, DIM, SIGMA_FINAL, T_FINAL, MU_1, MU_2, SIGMA_IC
    global MIXTURE_WEIGHT, D_BASE, ANNEALING_START_EPOCH, ANNEALING_END_EPOCH, device
    global pushforward_net, loss_log, sigma_log, lr_gen_log, lr_test_log
    global x_grid_fv, times_fv, rho_history_fv, means_fv, variances_fv

    print("=" * 70)
    print("Triple-Well Potential Visualization")
    print("=" * 70)

    # Check files exist
    check_files_exist()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load checkpoint with weights_only=False (safe since we trust our own file)
    print("Loading trained model...")
    checkpoint = torch.load('results_triple_well/checkpoint.pth', map_location='cpu', weights_only=False)
    params = checkpoint['hyperparameters']

    # Extract parameters
    DIM = params['DIM']
    SIGMA_FINAL = params['SIGMA_FINAL']
    T_FINAL = params['T_FINAL']
    MU_1 = params['MU_1']
    MU_2 = params['MU_2']
    SIGMA_IC = params['SIGMA_IC']
    MIXTURE_WEIGHT = params['MIXTURE_WEIGHT']
    D_BASE = params['D_BASE']
    ANNEALING_START_EPOCH = params['ANNEALING_START_EPOCH']
    ANNEALING_END_EPOCH = params['ANNEALING_END_EPOCH']

    # Load model
    pushforward_net = PushforwardNetwork(D_BASE, DIM).to(device)
    pushforward_net.load_state_dict(checkpoint['pushforward_net_state_dict'])
    pushforward_net.eval()

    loss_log = checkpoint['loss_log']
    sigma_log = checkpoint['sigma_log']
    lr_gen_log = checkpoint['lr_gen_log']
    lr_test_log = checkpoint['lr_test_log']

    # Run finite volume solver for comparison
    x_grid_fv, times_fv, rho_history_fv, means_fv, variances_fv = run_finite_volume()

    print("\n1. Plotting potential and drift...")
    plot_potential()

    print("\n2. Plotting initial condition...")
    plot_initial_condition()

    print("\n3. Plotting training diagnostics...")
    plot_training_diagnostics()

    print("\n4. Comparing NN vs FV solutions...")
    comparison_results = plot_pdf_comparison()

    print("\n5. Plotting time evolution heatmap...")
    plot_time_evolution()

    print("\n" + "=" * 70)
    print("VISUALIZATION SUMMARY")
    print("=" * 70)
    print(f"\nComparison (NN vs FV):")
    print(f"  Average L² error: {np.mean(comparison_results['l2_errors']):.6f}")
    print(f"  L² errors at times {comparison_results['times']}:")
    for t, err in zip(comparison_results['times'], comparison_results['l2_errors']):
        print(f"    t = {t:.2f}: L² = {err:.6f}")

    print("\n" + "=" * 70)
    print("✓ All visualizations complete!")
    print("\nGenerated plots in 'results_triple_well/':")
    print("  1. potential_drift.png - Triple-well potential and drift")
    print("  2. initial_condition.png - Mixture Gaussian IC")
    print("  3. training_diagnostics.png - Loss, σ, and learning rates")
    print("  4. pdf_comparison.png - NN vs FV PDFs at different times")
    print("  5. time_evolution.png - Heatmap of density evolution")
    print("=" * 70)


if __name__ == '__main__':
    main()



