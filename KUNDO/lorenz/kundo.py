import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torchdiffeq import odeint as neural_odeint
from scipy.integrate import odeint  # for generating training data
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define neural network modules
class NeuralODE(nn.Module):
    def __init__(self, num_basis, hidden_dim):
        super(NeuralODE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_basis, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_basis)
        )

    def forward(self, t, x):
        return self.net(x)

class ODEFunc(nn.Module):
    def __init__(self, num_basis, hidden_dim):
        super(ODEFunc, self).__init__()
        self.node = NeuralODE(num_basis, hidden_dim)

    def forward(self, t, x):
        return self.node(t, x)

class NeuralSBF(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_basis):
        super(NeuralSBF, self).__init__()
        self.input_map = nn.Linear(input_dim, num_basis)
        self.ode_func = ODEFunc(num_basis, hidden_dim)
        self.num_basis = num_basis

    def forward(self, x):
        x = self.input_map(x)
        t = torch.linspace(0, 1, 2).float().to(x.device)  # t from 0 to 1 with two points
        out = neural_odeint(self.ode_func, x, t, method='euler')
        return out[-1]

# Define explicit system class
class ExplicitSystem:
    def __init__(self, Gamma, basis_functions):
        self.Gamma = Gamma
        self.basis_functions = basis_functions

    def F(self, x, u):
        xu = np.concatenate([x, [u]])
        G = np.array([f(xu) for f in self.basis_functions]).reshape(1, -1)
        return np.dot(G, self.Gamma).flatten()

# Helper function to compute metrics
def compute_metrics(y_true, y_pred):        
    original_shape = y_true.shape
    y_true = y_true.reshape(-1, original_shape[-1])
    y_pred = y_pred.reshape(-1, original_shape[-1])
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # To avoid division by zero in MAPE
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

# Train neural SBF
def train_neural_sbf(X_batch, U_batch, dX_batch, num_basis, hidden_dim=256, num_epochs=1000, lr=0.001, batch_size=32):
    X = torch.tensor(X_batch, dtype=torch.float32).to(device)
    U = torch.tensor(U_batch, dtype=torch.float32).unsqueeze(2).to(device)
    dX = torch.tensor(dX_batch, dtype=torch.float32).to(device)
    XU = torch.cat([X, U], dim=2)

    dataset = torch.utils.data.TensorDataset(XU, dX)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    neural_sbf = NeuralSBF(XU.shape[2], hidden_dim, num_basis).to(device)
    optimizer = optim.Adam(neural_sbf.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        neural_sbf.train()
        total_loss = 0
        for batch_XU, batch_dX in dataloader:
            batch_XU = batch_XU.to(device)
            batch_dX = batch_dX.to(device)
            optimizer.zero_grad()
            G = neural_sbf(batch_XU)  # (batch_size, num_basis)
            Gamma = torch.linalg.lstsq(G, batch_dX).solution  # (num_basis, dX_dim)
            X_pred = torch.matmul(G, Gamma)  # (batch_size, dX_dim)
            loss = criterion(X_pred, batch_dX)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        all_pred = []
        all_true = []
        if (epoch + 1) % 100 == 0 or epoch == 0:
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
            all_pred.append(X_pred.cpu().detach().numpy())
            all_true.append(batch_dX.cpu().detach().numpy())
            all_pred = np.concatenate(all_pred)
            all_true = np.concatenate(all_true)
            
            metrics = compute_metrics(all_true, all_pred)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, MAE: {metrics["MAE"]:.6f}, RMSE: {metrics["RMSE"]:.6f}, R2: {metrics["R2"]:.6f}, MAPE: {metrics["MAPE"]:.2f}%')

    # After training, compute the final Gamma using all data
    with torch.no_grad():
        G = neural_sbf(XU)  # (N, num_basis)
        Gamma = torch.linalg.lstsq(G, dX).solution  # (num_basis, dX_dim)

    return neural_sbf, Gamma.cpu().numpy()

# Extract basis functions (return values of all basis functions for given inputs)
def extract_basis_functions(neural_sbf, input_data):
    # Prepare input tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

    # Process in batches to prevent memory issues
    batch_size = 10000
    G = []
    neural_sbf.eval()
    with torch.no_grad():
        for i in range(0, input_tensor.shape[0], batch_size):
            batch = input_tensor[i:i+batch_size]
            G_batch = neural_sbf(batch).cpu().numpy()  # (batch_size, num_basis)
            G.append(G_batch)
    G = np.vstack(G)  # (N, num_basis)
    return G

# Fit and print basis functions using polynomial features
def fit_and_print_basis_functions(input_data, G, num_basis, degree=2):
    scaler = StandardScaler()
    scaler.fit(input_data)  # Fit scaler on input data

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_scaled = scaler.transform(input_data)
    X_poly = poly.fit_transform(X_scaled)
    feature_names = poly.get_feature_names(['x1', 'x2', 'z', 'u'])  # Adjust based on input dimensions

    print("\n--- Basis Functions Approximations ---")
    fitted_basis_functions = []
    for i in range(num_basis):
        reg = LinearRegression().fit(X_poly, G[:, i])
        coeffs = reg.coef_
        intercept = reg.intercept_

        expr = f"f{i+1} = {intercept:.4f}"
        for coef, name in zip(coeffs, feature_names):
            if abs(coef) > 1e-4:
                expr += f" + {coef:.4f}*{name}"
        print(expr)
        print(f"R² score: {reg.score(X_poly, G[:, i]):.4f}\n")

        # Define fitted basis function
        def fitted_f(xu, coef=coeffs, intercept=intercept, poly=poly, scaler=scaler):
            xu = np.array(xu).reshape(1, -1)
            xu_scaled = scaler.transform(xu)
            xu_poly = poly.transform(xu_scaled)
            return np.dot(xu_poly, coef) + intercept

        fitted_basis_functions.append(fitted_f)
    return fitted_basis_functions

# Compute Gamma using explicit polynomial basis functions
def compute_explicit_gamma(X_batch, U_batch, dX_batch, fitted_basis_functions):
    XU = np.concatenate([X_batch, U_batch.reshape(X_batch.shape[0], X_batch.shape[1], 1)], axis=2)
    num_trajectories, num_time_steps, input_dim = XU.shape

    G_list = []
    for traj in XU:
        for xu in traj:
            G = np.array([f(xu) for f in fitted_basis_functions]).reshape(-1)
            G_list.append(G)

    G_matrix = np.array(G_list)  # (N, num_basis)
    dX_flat = dX_batch.reshape(-1, dX_batch.shape[-1])  # (N, dX_dim)

    # Compute Gamma matrix
    Gamma, _, _, _ = np.linalg.lstsq(G_matrix, dX_flat, rcond=None)
    return Gamma

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

# Visualize trajectories
def visualize_trajectories(X_batch, X_identified_batch, title):
    fig = plt.figure(figsize=(18, 8))
    
    # True trajectories subplot
    ax1 = fig.add_subplot(121, projection='3d')
    for X in X_batch:
        ax1.plot(X[:, 0], X[:, 1], X[:, 2], 'b', linewidth=1)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_zlabel('z', fontsize=12)
    ax1.set_title('True Trajectories', fontsize=14, fontweight='bold')

    # Identified trajectories subplot
    ax2 = fig.add_subplot(122, projection='3d')
    for X_identified in X_identified_batch:
        ax2.plot(X_identified[:, 0], X_identified[:, 1], X_identified[:, 2], 'r--', linewidth=1)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_zlabel('z', fontsize=12)
    ax2.set_title('Identified Trajectories', fontsize=14, fontweight='bold')

    # Main title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# Print results
def print_results(identified_system, rmse, r_squared):
    print("\n--- Results ---")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Goodness of fit (R-squared): {r_squared:.4f}")

    print("\n--- Identified System Parameters ---")
    print("Gamma matrix:")
    print(identified_system.Gamma)

# Main workflow
def main():
    # Simulation parameters
    t_end = 35
    dt = 0.01
    num_train_trajectories = 70
    num_test_trajectories = 5

    # Input function (control input u)
    def u(t):
        return np.sin(t)

    # Generate training data
    print("Generating training trajectories...")
    X_train, U_train, dX_train, t = generate_trajectories(num_train_trajectories, t_end, dt, u)
    print("Training trajectories generated.")

    # Train neural SBF
    num_basis = 21
    print(f"Training Neural SBF with {num_basis} basis functions...")
    start_time = time.time()
    trained_neural_sbf, Gamma_neural = train_neural_sbf(X_train, U_train, dX_train, num_basis)
    training_time = time.time() - start_time
    print(f"Neural SBF training completed in {training_time:.2f} seconds.")

    # Extract basis functions
    print("Extracting basis functions...")
    start_time = time.time()

    # To avoid memory issues, use a smaller num_samples, e.g., 20
    num_samples = 20  # Adjust this value to fit your computational resources
    x1 = np.linspace(-5, 5, num_samples)
    x2 = np.linspace(-5, 5, num_samples)
    x3 = np.linspace(-5, 5, num_samples)
    u_vals = np.linspace(-1, 1, num_samples)

    X, Y, Z, U_vals = np.meshgrid(x1, x2, x3, u_vals)
    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)
    Z_flat = Z.reshape(-1, 1)
    U_flat = U_vals.reshape(-1, 1)

    input_data = np.hstack([X_flat, Y_flat, Z_flat, U_flat])  # 4D input (x1, x2, z, u)

    G = extract_basis_functions(trained_neural_sbf, input_data)
    basis_extraction_time = time.time() - start_time
    print(f"Basis functions extracted in {basis_extraction_time:.2f} seconds.")

    # Fit with polynomial basis functions
    print("Fitting basis functions with polynomial features...")
    fitted_basis_functions = fit_and_print_basis_functions(input_data, G, num_basis, degree=2)

    # Identify system using explicit polynomial basis functions
    print("Identifying system using explicit polynomial basis functions...")
    start_time = time.time()
    Gamma_explicit = compute_explicit_gamma(X_train, U_train, dX_train, fitted_basis_functions)
    system_identification_time = time.time() - start_time
    print(f"System identified in {system_identification_time:.2f} seconds.")

    identified_system = ExplicitSystem(Gamma_explicit, fitted_basis_functions)

    # Generate test data
    print("Generating test trajectories...")
    X_test, U_test, dX_test, t_test = generate_trajectories(num_test_trajectories, t_end, dt, u)
    print("Test trajectories generated.")

    # Predict using explicit system
    print("Simulating identified system on test data...")
    start_time = time.time()
    X_identified_test = np.zeros_like(X_test)
    for i in range(num_test_trajectories):
        X_identified_test[i, 0] = X_test[i, 0]  # Use true initial conditions
        for j in range(1, len(t_test)):
            x_prev = X_identified_test[i, j-1]
            u_prev = U_test[i, j-1]
            dx = identified_system.F(x_prev, u_prev)
            X_identified_test[i, j] = x_prev + dx * (t_test[j] - t_test[j-1])
    simulation_time = time.time() - start_time
    print(f"Simulation on test data completed in {simulation_time:.2f} seconds.")

    print(compute_metrics(X_test, X_identified_test))

    # Calculate results
    # Calculate RMSE per trajectory and then average
    rmse = np.sqrt(np.mean((X_test - X_identified_test)**2, axis=(1,2)))
    avg_rmse = np.mean(rmse)

    # Compute R-squared per trajectory and then average
    r_squared_list = []
    for i in range(num_test_trajectories):
        ss_res = np.sum((X_test[i] - X_identified_test[i])**2, axis=0)
        ss_tot = np.sum((X_test[i] - np.mean(X_test[i], axis=0))**2, axis=0)
        r2 = 1 - ss_res / ss_tot
        # Average R2 over all state variables
        r_squared_list.append(np.mean(r2))
    avg_r_squared = np.mean(r_squared_list)

    # Print results and visualize
    print_results(identified_system, avg_rmse, avg_r_squared)

    # Visualize trajectories
    print("Visualizing test trajectories...")
    visualize_trajectories(X_test, X_identified_test, 'Comparison of True and Explicitly Identified Trajectories on Test Data')

    # ============================
    # EXTRAPOLATION PART
    # ============================

    print("Generating EXTRAPOLATION test trajectories...")
    X_test, U_test, dX_test, t_test = generate_trajectories(num_test_trajectories, t_end * 1.5, dt, u)
    print("EXTRAPOLATION Test trajectories generated.")

    # Predict using explicit system
    print("Simulating identified system on test data...")
    start_time = time.time()
    X_identified_test = np.zeros_like(X_test)
    for i in range(num_test_trajectories):
        X_identified_test[i, 0] = X_test[i, 0]  # Use true initial conditions
        for j in range(1, len(t_test)):
            x_prev = X_identified_test[i, j-1]
            u_prev = U_test[i, j-1]
            dx = identified_system.F(x_prev, u_prev)
            X_identified_test[i, j] = x_prev + dx * (t_test[j] - t_test[j-1])
    simulation_time = time.time() - start_time
    print(f"Simulation on test data completed in {simulation_time:.2f} seconds.")

    print(compute_metrics(X_test, X_identified_test))

    # Calculate results
    # Calculate RMSE per trajectory and then average
    rmse = np.sqrt(np.mean((X_test - X_identified_test)**2, axis=(1,2)))
    avg_rmse = np.mean(rmse)

    # Compute R-squared per trajectory and then average
    r_squared_list = []
    for i in range(num_test_trajectories):
        ss_res = np.sum((X_test[i] - X_identified_test[i])**2, axis=0)
        ss_tot = np.sum((X_test[i] - np.mean(X_test[i], axis=0))**2, axis=0)
        r2 = 1 - ss_res / ss_tot
        # Average R2 over all state variables
        r_squared_list.append(np.mean(r2))
    avg_r_squared = np.mean(r_squared_list)

    # Print results and visualize
    print_results(identified_system, avg_rmse, avg_r_squared)

    # Visualize trajectories
    print("Visualizing test trajectories...")
    visualize_trajectories(X_test, X_identified_test, 'Comparison of True and Explicitly Identified Trajectories on Extrapolation Data')

if __name__ == "__main__":
    main()
