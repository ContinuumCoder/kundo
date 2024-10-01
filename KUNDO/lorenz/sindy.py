
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pysindy as ps
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

    # Prepare data for SINDy
    X_train_flat = X_train.reshape(-1, 3)
    U_train_flat = U_train.reshape(-1, 1)
    XU_train = np.hstack([X_train_flat, U_train_flat])

    # Initialize and fit SINDy model
    print("Training SINDy model...")
    start_time = time.time()
    # Define custom library including control input
    library = ps.PolynomialLibrary(degree=3)
    feature_names = ['x', 'y', 'z', 'u']
    model = ps.SINDy(feature_names=feature_names, optimizer=ps.STLSQ(threshold=0.1), feature_library=library)
    model.fit(XU_train, t=dt)
    training_time = time.time() - start_time
    print(f"SINDy training completed in {training_time:.2f} seconds.")
    model.print()

    # Generate test data
    print("Generating test trajectories...")
    X_test, U_test, dX_test, t_test = generate_trajectories(num_test_trajectories, t_end, dt, u)
    print("Test trajectories generated.")

    # Prepare test data
    X_test_flat = X_test.reshape(-1, 3)
    U_test_flat = U_test.reshape(-1, 1)
    XU_test = np.hstack([X_test_flat, U_test_flat])

    # Predict using SINDy model
    print("Predicting on test data using SINDy...")
    dX_pred = model.predict(XU_test)

    # Compute metrics
    mse = mean_squared_error(dX_test.reshape(-1, 3), dX_pred)
    r2 = r2_score(dX_test.reshape(-1, 3), dX_pred)
    print(f"Test MSE: {mse:.6f}, R2: {r2:.6f}")

    # Reshape for visualization
    dX_pred_reshaped = dX_pred.reshape(num_test_trajectories, -1, 3)
    dX_true_reshaped = dX_test.reshape(num_test_trajectories, -1, 3)

    # Visualize one trajectory comparison
    traj = 0
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(dX_true_reshaped[traj,:,0], label='True dx')
    plt.plot(dX_pred_reshaped[traj,:,0], label='Predicted dx')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('dx')

    plt.subplot(1, 3, 2)
    plt.plot(dX_true_reshaped[traj,:,1], label='True dy')
    plt.plot(dX_pred_reshaped[traj,:,1], label='Predicted dy')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('dy')

    plt.subplot(1, 3, 3)
    plt.plot(dX_true_reshaped[traj,:,2], label='True dz')
    plt.plot(dX_pred_reshaped[traj,:,2], label='Predicted dz')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('dz')

    plt.suptitle('SINDy Predictions vs True Dynamics')
    plt.show()

if __name__ == "__main__":
    main()# sindy_system_identification.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pysindy as ps
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

    # Prepare data for SINDy
    X_train_flat = X_train.reshape(-1, 3)
    U_train_flat = U_train.reshape(-1, 1)
    XU_train = np.hstack([X_train_flat, U_train_flat])

    # Initialize and fit SINDy model
    print("Training SINDy model...")
    start_time = time.time()
    # Define custom library including control input
    library = ps.PolynomialLibrary(degree=3)
    feature_names = ['x', 'y', 'z', 'u']
    model = ps.SINDy(feature_names=feature_names, optimizer=ps.STLSQ(threshold=0.1), feature_library=library)
    model.fit(XU_train, t=dt)
    training_time = time.time() - start_time
    print(f"SINDy training completed in {training_time:.2f} seconds.")
    model.print()

    # Generate test data
    print("Generating test trajectories...")
    X_test, U_test, dX_test, t_test = generate_trajectories(num_test_trajectories, t_end, dt, u)
    print("Test trajectories generated.")

    # Prepare test data
    X_test_flat = X_test.reshape(-1, 3)
    U_test_flat = U_test.reshape(-1, 1)
    XU_test = np.hstack([X_test_flat, U_test_flat])

    # Predict using SINDy model
    print("Predicting on test data using SINDy...")
    dX_pred = model.predict(XU_test)

    # Compute metrics
    mse = mean_squared_error(dX_test.reshape(-1, 3), dX_pred)
    r2 = r2_score(dX_test.reshape(-1, 3), dX_pred)
    print(f"Test MSE: {mse:.6f}, R2: {r2:.6f}")

    # Reshape for visualization
    dX_pred_reshaped = dX_pred.reshape(num_test_trajectories, -1, 3)
    dX_true_reshaped = dX_test.reshape(num_test_trajectories, -1, 3)

    # Visualize one trajectory comparison
    traj = 0
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(dX_true_reshaped[traj,:,0], label='True dx')
    plt.plot(dX_pred_reshaped[traj,:,0], label='Predicted dx')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('dx')

    plt.subplot(1, 3, 2)
    plt.plot(dX_true_reshaped[traj,:,1], label='True dy')
    plt.plot(dX_pred_reshaped[traj,:,1], label='Predicted dy')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('dy')

    plt.subplot(1, 3, 3)
    plt.plot(dX_true_reshaped[traj,:,2], label='True dz')
    plt.plot(dX_pred_reshaped[traj,:,2], label='Predicted dz')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('dz')

    plt.suptitle('SINDy Predictions vs True Dynamics')
    plt.show()

if __name__ == "__main__":
    main()
