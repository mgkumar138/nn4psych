#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from sklearn.decomposition import PCA
from utils_funcs import ActorCritic
import torch

# Define the RNN dynamics
def rnn_dynamics(h, W):
    return np.tanh(np.dot(W, h))

# Function to find fixed points iteratively
def find_fixed_points(W, num_iterations=1000, tolerance=1e-6):
    fixed_points = []
    for _ in range(num_iterations):
        h = np.random.randn(W.shape[0])  # Random initial state
        for _ in range(100):  # Iterate to convergence
            h_new = rnn_dynamics(h, W)
            if np.linalg.norm(h_new - h) < tolerance:
                fixed_points.append(h_new)
                break
            h = h_new
    return np.unique(np.round(fixed_points, decimals=6), axis=0)  # Remove duplicates

# Function to compute the Jacobian at a fixed point
def compute_jacobian(h, W):
    derivative = 1 - np.tanh(np.dot(W, h)) ** 2  # Derivative of tanh
    J = W * derivative[:, np.newaxis]  # Jacobian
    return J

# Function to check for line attractors by analyzing the null space of W
def check_line_attractor(W):
    ns = null_space(W)
    if ns.shape[1] > 0:
        print("Line attractor detected! Null space dimension:", ns.shape[1])
        return ns
    else:
        print("No line attractor detected.")
        return None

# Simulate RNN dynamics
def simulate_rnn(W, h0, num_steps=100):
    h = h0
    trajectory = [h]
    for _ in range(num_steps):
        h = rnn_dynamics(h, W)
        trajectory.append(h)
    return np.array(trajectory)

# Plot trajectory in 2D PCA space
def plot_trajectory(trajectory, fixed_points=None):
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="RNN Trajectory")
    if fixed_points is not None:
        plt.scatter(fixed_points[:, 0], fixed_points[:, 1], color="red", label="Fixed Points")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("RNN Dynamics (2D PCA Projection)")
    plt.legend()
    plt.grid()
    plt.show()

def run_fp_analysis(
        model,
        rnn_info_dict, 
        model_path = "./model_params/36.0_V3_0.0ns_Nonelb_Noneub_0.95g_64n_40000e_2s.pth"
):
    """
    Main function to run the fixed point analysis.
    It initializes the model, finds fixed points, computes the Jacobian,
    checks for line attractors, and simulates the RNN dynamics.
    """
    # Parameters
    np.random.seed(42)
    N = rnn_info_dict['hidden_dim']  # Dimensionality of hidden state
    W = np.random.randn(N, N) * 0.1  # Weight matrix (scaled for stability)

    # model_path = "./model_params_gamma/12.0_V3_0.0ns_Nonelb_Noneub_0.7g_64n_40000e_2s.pth"

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print('Load Model')
    W  = np.array(model.state_dict()['rnn.weight_hh_l0'])

    # Find fixed points
    fixed_points = find_fixed_points(W)
    print("Number of fixed points found:", len(fixed_points))

    # Analyze stability of fixed points
    for i, h in enumerate(fixed_points):
        J = compute_jacobian(h, W)
        eigenvalues = np.linalg.eigvals(J)
        print(f"Fixed point {i + 1}:")
        print("Max eigenvalue magnitude:", np.max(np.abs(eigenvalues)))
        if np.max(np.abs(eigenvalues)) < 1:
            print("Stable fixed point (attractor).")
        else:
            print("Unstable fixed point.")

    # Check for line attractors
    null_space_basis = check_line_attractor(W)

    # Simulate and plot dynamics
    h0 = np.random.randn(N)  # Initial state
    trajectory = simulate_rnn(W, h0)

    # Use PCA to reduce dimensionality for visualization
    pca = PCA(n_components=2)
    trajectory_2d = pca.fit_transform(trajectory)
    fixed_points_2d = pca.transform(fixed_points) if len(fixed_points) > 0 else None

    plot_trajectory(trajectory_2d, fixed_points_2d)

    return pca, trajectory_2d, fixed_points_2d


