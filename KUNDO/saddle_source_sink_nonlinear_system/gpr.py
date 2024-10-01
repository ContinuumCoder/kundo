import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.integrate import odeint

# Define system dynamics
def non_linear_system(x, t, a=1.0, b=-1.0, c=-1.0):
    x_val, y_val, z_val = x
    dx = a * x_val
    dy = b * y_val
    dz = c * z_val + x_val * y_val
    return np.array([dx, dy, dz])

# Generate trajectory data
def generate_trajectories(num_trajectories, t_end, dt, a=1.0, b=-1.0, c=-1.0):
    t = np.linspace(0, t_end, int(t_end/dt) + 1)
    X_batch = []
    dX_batch = []
    
    for _ in range(num_trajectories):
        x0 = np.random.uniform(-0.5, 0.5, 3)  # Random initial conditions
        X = odeint(non_linear_system, x0, t, args=(a, b, c))
        dX = np.array([non_linear_system(x, ti, a, b, c) for ti, x in zip(t, X)])
        X_batch.append(X)
        dX_batch.append(dX)
    
    return np.array(X_batch), np.array(dX_batch), t

# Calculate performance metrics
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# Visualize trajectories
def visualize_trajectories(X_true, X_pred, title):
    fig = plt.figure(figsize=(12, 6))
    
    # 3D trajectory
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(X_true[:,0], X_true[:,1], X_true[:,2], label='True')
    ax.plot(X_pred[:,0], X_pred[:,1], X_pred[:,2], label='GPR Predicted')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Trajectories')
    ax.legend()
    
    # Single variable comparison
    ax2 = fig.add_subplot(122)
    labels = ['x', 'y', 'z']
    for i in range(3):
        ax2.plot(X_true[:,i], label=f'True {labels[i]}')
        ax2.plot(X_pred[:,i], label=f'GPR Predicted {labels[i]}')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value')
    ax2.set_title('Trajectory Comparison')
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Main process
def main():
    # Simulation parameters
    t_end = 35
    dt = 0.01
    num_train_trajectories = 70
    num_test_trajectories = 5
    
    # System parameters (default Saddle system)
    a = 1.0
    b = -1.0
    c = -1.0
    
    # Generate training data
    print("Generating training trajectories...")
    X_train, dX_train, t_train = generate_trajectories(num_train_trajectories, t_end, dt, a, b, c)
    print("Training trajectories generation completed.")
    
    # Prepare training data
    X_train_flat = X_train.reshape(-1, 3)
    dX_train_flat = dX_train.reshape(-1, 3)
    
    # Fit an independent GPR model for each output dimension
    kernels = [C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) for _ in range(3)]
    gpr_models = [GaussianProcessRegressor(kernel=k, n_restarts_optimizer=10, alpha=1e-4) for k in kernels]
    
    print("Training Gaussian Process Regression models...")
    for i, model in enumerate(gpr_models):
        print(f"Training GPR model for dX/dt component {i+1}")
        model.fit(X_train_flat, dX_train_flat[:,i])
    print("GPR models training completed.")
    
    # Generate test data
    print("Generating test trajectories...")
    X_test, dX_test, t_test = generate_trajectories(num_test_trajectories, t_end, dt, a, b, c)
    print("Test trajectories generation completed.")
    
    # Make predictions using GPR
    print("Making predictions using GPR models...")
    X_pred = []
    for traj in X_test:
        traj_pred = [traj[0]]  # Initial condition
        for i in range(1, len(traj)):
            x_current = traj_pred[-1].reshape(1, -1)
            dx_pred = np.array([model.predict(x_current)[0] for model in gpr_models])
            x_next = x_current + dx_pred * dt
            traj_pred.append(x_next.flatten())
        traj_pred = np.array(traj_pred)
        X_pred.append(traj_pred)
    X_pred = np.array(X_pred)
    
    # Calculate metrics
    metrics = compute_metrics(X_test, X_pred)
    print(f"Test set performance metrics: {metrics}")
    
    # Visualize some test trajectories
    for i in range(num_test_trajectories):
        true_traj = X_test[i]
        pred_traj = X_pred[i]
        visualize_trajectories(true_traj, pred_traj, f'Test Trajectory {i+1}')

if __name__ == "__main__":
    main()
