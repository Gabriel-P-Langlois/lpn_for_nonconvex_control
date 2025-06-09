# GPL Paper Section 4.1.2 Experiment: Quadratic Hamiltonian with L1 Prior
# Approximation of J(x) using HJ PDE theory

"""
This script adapts the LPN methodology to learn the prior J(x) = ||x||_1 in the context of a 
Hamilton-Jacobi PDE with a quadratic Hamiltonian H(p) = (1/2)||p||_2^2.
The LPN learns the function ψ(y) = J*(y) - (1/2)||y||_2^2.
Training data uses samples {y_j, ψ(y_j)} where ψ(y_j) = (1/2)||y_j||_2^2 - S(y_j,1), and
S(y_j,1) = min_{x ∈ R^d} {(1/2)||x-y_j||_2^2 + ||x||_1}} (Moreau envelope of J(x)=||x||_1 at t=1).

The learned prior J_est(x) is approximated using the formula derived from the method of characteristics:
J(x) ≈ S(x + t∇_x S(x,t),t) - (t/2)||∇_x S(x,t)||^2_2, with t=1.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import seaborn as sns

# Ensure these custom modules are in your Python path or the same directory
from network import LPN
from lib.utils import cvx  # Used for evaluating learned psi
# from lib.invert import invert  # Not strictly needed for this J_approx method

# Set location
MODEL_DIR = "experiments/models/sec412_quadratic_l1_hj_approx"
os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set random seeds
np.random.seed(1)
torch.manual_seed(1)

# Plotting options
plt.rcParams.update({"text.usetex": False})
sns.set_theme()

# Parameters for this 1D Experiment
dim_exp = 8  # Dimension (e.g., 1D or 2D as in the paper for L1 prior)
data_points_exp = 500  # Example number of data points
iterations_exp_initial = 1000  # Iterations for initial training phase
iterations_exp_long = 10000  # Iterations for longer training phase 
a_exp = 4  # Grid limit for sampling y_j, e.g., y_j in [-a, a]^d
spacing = 100  # Number of points for plotting 

# LPN model parameters
beta = 10       # beta of softplus
hidden = 50     # number of hidden units 
layers = 4      # number of layers (refers to parameter in LPN class)

# Helper Functions for J(x) = |x|

def euclid_norm_sq(x_np):  # Expects numpy array
    """Calculate squared Euclidean norm."""
    if x_np.ndim == 1:
        return np.sum(x_np*x_np)
    return np.sum(x_np*x_np, axis=1)

def prox_l1_vec(y, t_lambda):  # y is N x dim numpy array
    """Element-wise proximal operator of t_lambda*||x||_1 (soft-thresholding)."""
    return np.sign(y) * np.maximum(np.abs(y) - t_lambda, 0)

def hjsol_true_quadratic_l1(y_points_np):  # y_points_np is N x dim
    """
    Computes the true Moreau envelope S(y, t=1) for J(x)=||x||_1.
    S(y, 1) = min_x { 0.5*||x-y||_2^2 + ||x||_1 }
    """
    t_moreau = 1.0
    prox_y_np = prox_l1_vec(y_points_np, t_moreau) 
    term1 = 0.5 * euclid_norm_sq(prox_y_np - y_points_np) 
    term2 = np.sum(np.abs(prox_y_np), axis=1) 
    s_values = term1 + term2
    return s_values  # N-dimensional array

def prior_true_quadratic_l1(y_points_np):  # y_points_np is N x dim
    """Computes the true prior J(y) = ||y||_1."""
    return np.sum(np.abs(y_points_np), axis=1)  # N-dimensional array

def cvx_true_quadratic_l1(y_points_np):  # y_points_np is N x dim
    """
    Computes the target function psi(y) = 0.5*||y||_2^2 - S(y, t=1) for J(x)=||x||_1.
    """
    s_y_1 = hjsol_true_quadratic_l1(y_points_np)
    psi_y = 0.5 * euclid_norm_sq(y_points_np) - s_y_1
    return psi_y  # N-dimensional array

def evaluate_learned_J_HJ_approx(x_eval_points_np, model, t_val=1.0):
    """
    Evaluates the learned prior J_est(x) using the LPN and HJ approximation.
    J(x) ≈ S(x + t*grad_S(x,t), t) - (t/2)*||grad_S(x,t)||^2
    where S(x,t) = (1/2t)||x||^2 - (1/t)psi_t(x) if LPN learns psi_t(x).
    Here, LPN learns psi(x) which is psi_1(x), so we use t=1.
    S(x,1) = 0.5*||x||^2 - psi(x)
    grad_S(x,1) = x - grad_psi(x)
    
    Args:
        x_eval_points_np (np.ndarray): N x dim array of points to estimate J(x) at.
        model: The trained LPN model object (e.g., a torch.nn.Module).
        t_val (float): Time parameter, typically 1.0 for this setup.
    
    Returns:
        np.ndarray: N-dimensional array of estimated prior values J_est(x_eval_points).
    """
    x_tensor = torch.from_numpy(x_eval_points_np).float().to(device)
    x_tensor.requires_grad_(True)  # Ensure grad can be computed
    
    # grad_psi_x = nabla_psi(x)
    # The LPN's forward() method (model(x_tensor)) should return nabla_psi(x)
    grad_psi_x_tensor = model(x_tensor) 
    
    # p_eval = nabla_S(x,1) = x - nabla_psi(x)
    p_eval_tensor = x_tensor - grad_psi_x_tensor
    
    # x_new = x + t * p_eval
    x_new_tensor = x_tensor + t_val * p_eval_tensor
    
    # psi_x_new = psi(x_new)
    psi_x_new_tensor = model.scalar(x_new_tensor)  # Should be N x 1
    
    # S(x_new, 1) = 0.5*||x_new||^2 - psi(x_new)
    norm_sq_x_new_tensor = torch.sum(x_new_tensor**2, dim=1, keepdim=True)  # N x 1
    S_x_new_t_tensor = 0.5 * norm_sq_x_new_tensor - psi_x_new_tensor  # N x 1
    
    # (t/2)*||p_eval||^2
    norm_sq_p_eval_tensor = torch.sum(p_eval_tensor**2, dim=1, keepdim=True)  # N x 1
    term2_tensor = (t_val / 2.0) * norm_sq_p_eval_tensor  # N x 1
    
    J_approx_tensor = S_x_new_t_tensor - term2_tensor  # N x 1
    
    return J_approx_tensor.detach().cpu().numpy().flatten()  # N-dimensional

def compute_1d_plot_points(a, spacing):
    """Generate 1D plot points."""
    xi = np.linspace(-a, a, spacing)
    points_1d = xi.reshape(-1, 1)
    return xi, points_1d

def compute_square_cross_sections(a, spacing, dim):
    xi = np.linspace(-a, a, spacing)
    grid = np.zeros((xi.size, dim))
    x1_0_points = np.copy(grid)
    x1_0_points[:, 0] = xi
    x2_0_points = np.copy(grid)
    if dim > 1:
        x2_0_points[:, 1] = xi
    elif dim == 1:
        x2_0_points = x1_0_points.copy()
    return xi, x1_0_points, x2_0_points

def plots_1d_l1_prior_hj_approx(model, a, spacing, dim, hamiltonian_str="H(p) = 0.5 p^2"):
    """Plot 1D L1 prior results using HJ approximation."""
    if dim != 1:
        raise ValueError("This plotting function is for 1D experiments.")
    
    xi, eval_points_np = compute_1d_plot_points(a, spacing)  # eval_points_np is N x 1

    psi_true_vals = cvx_true_quadratic_l1(eval_points_np)
    J_true_vals = prior_true_quadratic_l1(eval_points_np)

    # Learned psi(y) from LPN
    psi_est_vals = cvx(eval_points_np, model)  # Uses lib.utils.cvx which calls model.scalar
    
    # Learned J(x) using HJ approximation
    J_est_vals = evaluate_learned_J_HJ_approx(eval_points_np, model, t_val=1.0)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(xi, psi_est_vals, "-", ms=5, label="LPN (Learned $\\psi$)")
    plt.plot(xi, psi_true_vals, "--", ms=5, label="True $\\psi$ for $J(x)=|x|$")
    plt.grid(True)
    plt.title(f"$\\psi(y)$ for $J(x)=|x|$ ({hamiltonian_str})")
    plt.xlabel('$y$')
    plt.ylabel('$\\psi(y)$')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(xi, J_est_vals, "-", label="LPN (Learned $J_{HJ}$)")
    plt.plot(xi, J_true_vals, "--", label="True $J(x)=|x|$")
    plt.grid(True)
    plt.title(f"Prior $J(x)$ ({hamiltonian_str})")
    plt.xlabel('$x$')
    plt.ylabel('$J(x)$')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_all_1d_l1_prior_hj_approx(model, a, spacing, dim, hamiltonian_str="H(p) = 0.5 p^2"):
    """Plot all 1D L1 prior results."""
    print(f"\n--- Plotting for 1D L1 Prior (Dim={dim}, {hamiltonian_str}) ---")
    plots_1d_l1_prior_hj_approx(model, a, spacing, dim, hamiltonian_str)

def exp_func(x, gamma):
    """Exponential function for loss calculation."""
    if x.ndim > 1 and x.shape[1] == 1:
        x = x.squeeze(1)
    return -torch.exp(-((torch.linalg.vector_norm(x, ord=2, dim=-1) / gamma) ** 2)) + 1.0

def single_iteration(i, data_points, lpn_model, optimizer, input_tensor, cvx_samples_tensor, loss_type, gamma_loss=None):
    """Perform a single training iteration."""
    input_on_device = input_tensor.to(device)
    cvx_samples_on_device = cvx_samples_tensor.to(device)
    
    cvx_out = lpn_model.scalar(input_on_device)

    if loss_type == 2:
        loss = (cvx_out - cvx_samples_on_device).pow(2).sum() / data_points 
    elif loss_type == 1:
        loss = (cvx_out - cvx_samples_on_device).abs().sum() / data_points
    elif loss_type == 0:
        if gamma_loss is None:
            raise ValueError("gamma_loss must be provided for loss_type 0")
        diff = cvx_out.squeeze() - cvx_samples_on_device.squeeze() 
        loss = exp_func(diff, gamma_loss).mean() 
    else:
        raise ValueError("loss_type must be 0, 1, or 2")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if hasattr(lpn_model, 'wclip'):
        lpn_model.wclip()

    if not i % 500:
        print(f"iteration {i}, loss {loss.item():.6f}")

# Main Experiment Code

def main():
    """Main experiment function."""
    print("\n--- Experiment: 1D L1 Prior, H(p) = 0.5 p^2 ---")
    
    # Generate Training Data
    y_j_exp_np = np.random.uniform(-a_exp, a_exp, (data_points_exp, dim_exp))
    cvx_samples_exp_np = cvx_true_quadratic_l1(y_j_exp_np)
    cvx_samples_exp_np = cvx_samples_exp_np.reshape(-1, 1)

    y_j_exp_tensor = torch.from_numpy(y_j_exp_np).float()
    cvx_samples_exp_tensor = torch.from_numpy(cvx_samples_exp_np).float()
    print(f"Generated training data: y_j_exp_tensor shape {y_j_exp_tensor.shape}, cvx_samples_exp_tensor shape {cvx_samples_exp_tensor.shape}")

    # Train LPN for J(x)=||x||_1 with L2 loss
    print(f"--- Training LPN for J(x)=||x||_1 (Quadratic H) ---")
    lpn_model_quad_l1 = LPN(in_dim=dim_exp, hidden=hidden, layers=layers, beta=beta).to(device)
    optimizer_quad_l1 = torch.optim.Adam(lpn_model_quad_l1.parameters(), lr=1e-4)

    # Initial training phase
    for i in range(iterations_exp_initial):
        single_iteration(i, data_points_exp, lpn_model_quad_l1, optimizer_quad_l1, 
                        y_j_exp_tensor, cvx_samples_exp_tensor, loss_type=2, gamma_loss=None)

    # Longer training phase
    for g in optimizer_quad_l1.param_groups:
        g["lr"] = 1e-4 
    for i in range(iterations_exp_long):
        single_iteration(i, data_points_exp, lpn_model_quad_l1, optimizer_quad_l1, 
                        y_j_exp_tensor, cvx_samples_exp_tensor, loss_type=2, gamma_loss=None)

    # Save model
    torch.save(lpn_model_quad_l1.state_dict(), os.path.join(MODEL_DIR, "QuadraticH_L1prior_l2_HJ_approx.pth"))

    # Plot results
    plot_all_1d_l1_prior_hj_approx(lpn_model_quad_l1, a_exp, spacing, dim_exp, hamiltonian_str="H(p)=0.5p^2")

if __name__ == "__main__":
    main()