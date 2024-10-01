
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
import time

# Lorenz system definition
def lorenz_system(x, t, u_func, sigma=10, rho=28, beta=8/3):
    u = u_func(t)
    dx = sigma * (x[1] - x[0]) + u
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    return np.array([dx, dy, dz])

# Generate trajectory data
def generate_trajectories(num_trajectories, t_end, dt, u_func):
    t = np.linspace(0, t_end, int(t_end/dt) + 1)
    X_batch = []
    U_batch = []
    dX_batch = []

    for _ in range(num_trajectories):
        x0 = np.random.uniform(-0.5, 0.5, 3)  # Random initial conditions
        X = odeint(lorenz_system, x0, t, args=(u_func,))
        U = u_func(t)
        dX = np.array([lorenz_system(x, ti, u_func) for ti, x in zip(t, X)])
        X_batch.append(X)
        U_batch.append(U)
        dX_batch.append(dX)

    return np.array(X_batch), np.array(U_batch), np.array(dX_batch), t

# Control input function
def u(t):
    return np.sin(t)

# Main Training and Evaluation
def main():
    # Simulation parameters
    t_end = 35
    dt = 0.01
    num_train_trajectories = 70
    num_test_trajectories = 5

    # Generate training data
    print("Generating training trajectories...")
    X_train, U_train, dX_train, t_train = generate_trajectories(num_train_trajectories, t_end, dt, u)
    print("Training trajectories generated.")

    # Prepare data for GPR
    X_train_flat = X_train.reshape(-1, 4)  # [x, y, z, u]
    dX_train_flat = dX_train.reshape(-1, 3)  # [dx, dy, dz]

    # Initialize GPR models for each derivative
    kernels = [RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5) for _ in range(3)]
    gpr_models = [GaussianProcessRegressor(kernel=kernels[i], alpha=0.0, normalize_y=True, n_restarts_optimizer=5) for i in range(3)]

    # Fit GPR models
    print("Training GPR models...")
    start_time = time.time()
    for i in range(3):
        print(f"Training GPR for dX_{i+1}/dt...")
        gpr_models[i].fit(X_train_flat, dX_train_flat[:, i])
    training_time = time.time() - start_time
    print(f"GPR training completed in {training_time:.2f} seconds.")

    # Generate test data
    print("Generating test trajectories...")
    X_test, U_test, dX_test, t_test = generate_trajectories(num_test_trajectories, t_end, dt, u)
    print("Test trajectories generated.")

    # Prepare test data
    X_test_flat = X_test.reshape(-1, 4)

    # Predict using GPR models
    print("Predicting on test data using GPR...")
    dX_pred = np.zeros_like(dX_test_flat = dX_test.reshape(-1, 3))
    for i in range(3):
        dX_pred[:, i], _ = gpr_models[i].predict(X_test_flat, return_std=True)
    dX_pred = dX_pred.reshape(num_test_trajectories, -1, 3)
    dX_true = dX_test.reshape(num_test_trajectories, -1, 3)

    # Compute metrics
    mse = mean_squared_error(dX_test_flat, dX_pred.reshape(-1, 3))
    r2 = r2_score(dX_test_flat, dX_pred.reshape(-1, 3))
    print(f"Test MSE: {mse:.6f}, R2: {r2:.6f}")

    # Visualize one trajectory comparison
    traj = 0
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(dX_true[traj,:,0], label='True dx')
    plt.plot(dX_pred[traj,:,0], label='Predicted dx')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('dx')

    plt.subplot(1, 3, 2)
    plt.plot(dX_true[traj,:,1], label='True dy')
    plt.plot(dX_pred[traj,:,1], label='Predicted dy')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('dy')

    plt.subplot(1, 3, 3)
    plt.plot(dX_true[traj,:,2], label='True dz')
    plt.plot(dX_pred[traj,:,2], label='Predicted dz')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('dz')

    plt.suptitle('GPR Predictions vs True Dynamics')
    plt.show()

if __name__ == "__main__":
    main()
