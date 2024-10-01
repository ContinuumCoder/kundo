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

    def F(self, x):
        """
        Compute the derivative based on the identified system.
        
        Args:
            x (array): Current state [x, y, z].
            
        Returns:
            array: Derivatives [dx/dt, dy/dt, dz/dt].
        """
        G = np.array([f(x) for f in self.basis_functions]).reshape(-1)
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
def train_neural_sbf(X_batch, dX_batch, num_basis, hidden_dim=256, num_epochs=1000, lr=0.001, batch_size=32):
    X = torch.tensor(X_batch, dtype=torch.float32).to(device)
    dX = torch.tensor(dX_batch, dtype=torch.float32).to(device)
    X_flat = X.view(-1, X.shape[2])  # Flatten to (N, state_dim)

    dataset = torch.utils.data.TensorDataset(X_flat, dX.view(-1, dX.shape[2]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    neural_sbf = NeuralSBF(X_flat.shape[1], hidden_dim, num_basis).to(device)
    optimizer = optim.Adam(neural_sbf.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        neural_sbf.train()
        total_loss = 0
        for batch_X, batch_dX in dataloader:
            batch_X = batch_X.to(device)
            batch_dX = batch_dX.to(device)
            optimizer.zero_grad()
            G = neural_sbf(batch_X)  # (batch_size, num_basis)
            Gamma = torch.linalg.lstsq(G, batch_dX).solution  # (num_basis, state_dim)
            X_pred = torch.matmul(G, Gamma)  # (batch_size, state_dim)
            loss = criterion(X_pred, batch_dX)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

    # After training, compute the final Gamma using all data
    with torch.no_grad():
        G = neural_sbf(X_flat)  # (N, num_basis)
        Gamma = torch.linalg.lstsq(G, dX.view(-1, dX.shape[2])).solution  # (num_basis, state_dim)

    return neural_sbf, Gamma.cpu().numpy()

# Extract basis functions
def extract_basis_functions(neural_sbf, input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
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
    feature_names = poly.get_feature_names_out(['x', 'y', 'z'])  # Updated for 3D

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
        def fitted_f(x, coef=coeffs, intercept=intercept, poly=poly, scaler=scaler):
            x = np.array(x).reshape(1, -1)
            x_scaled = scaler.transform(x)
            x_poly = poly.transform(x_scaled)
            return np.dot(x_poly, coef) + intercept

        fitted_basis_functions.append(fitted_f)
    return fitted_basis_functions

# Compute Gamma using explicit polynomial basis functions
def compute_explicit_gamma(X_batch, dX_batch, fitted_basis_functions):
    num_trajectories, num_time_steps, input_dim = X_batch.shape
    G_list = []
    for traj in X_batch:
        for x in traj:
            G = np.array([f(x) for f in fitted_basis_functions]).reshape(-1)
            G_list.append(G)

    G_matrix = np.array(G_list)  # (N, num_basis)
    dX_flat = dX_batch.reshape(-1, dX_batch.shape[-1])  # (N, state_dim)

    # Compute Gamma matrix using least squares
    Gamma, _, _, _ = np.linalg.lstsq(G_matrix, dX_flat, rcond=None)
    return Gamma

def non_linear_system(x, t, a=1.0, b=-1.0, c=-1.0):
    x_val, y_val, z_val = x
    dx = a * x_val
    dy = b * y_val
    dz = c * z_val + x_val * y_val
    return np.array([dx, dy, dz])


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
    case = 0 # 0:saddle, 1:sink; 2:source


    # System parameters for Saddle system
    a = 1.0
    b = -1.0
    c = -1.0

    if case == 1:
        # System parameters for Sink system
        a = -1.0
        b = -1.0
        c = -1.0

    if case == 2:
        # System parameters for Source system
        a = 1.0
        b = 1.0
        c = 1.0

    # Generate training data
    print("Generating training trajectories...")
    X_train, dX_train, t = generate_trajectories(num_train_trajectories, t_end, dt, a, b, c)
    print("Training trajectories generated.")

    # Train neural SBF
    num_basis = 21
    print(f"Training Neural SBF with {num_basis} basis functions...")
    start_time = time.time()
    trained_neural_sbf, Gamma_neural = train_neural_sbf(X_train, dX_train, num_basis)
    training_time = time.time() - start_time
    print(f"Neural SBF training completed in {training_time:.2f} seconds.")

    # Extract basis functions
    print("Extracting basis functions...")
    start_time = time.time()

    # Define the input data grid for basis function extraction
    # For 3D, x, y, z
    num_samples = 20  # Adjust based on computational resources
    x1 = np.linspace(-5, 5, num_samples)
    x2 = np.linspace(-5, 5, num_samples)
    x3 = np.linspace(-5, 5, num_samples)

    X, Y, Z = np.meshgrid(x1, x2, x3)
    input_data = np.hstack([
        X.reshape(-1, 1),
        Y.reshape(-1, 1),
        Z.reshape(-1, 1)
    ])  # 3D input (x, y, z)

    G = extract_basis_functions(trained_neural_sbf, input_data)
    basis_extraction_time = time.time() - start_time
    print(f"Basis functions extracted in {basis_extraction_time:.2f} seconds.")

    # Fit with polynomial basis functions
    print("Fitting basis functions with polynomial features...")
    fitted_basis_functions = fit_and_print_basis_functions(input_data, G, num_basis, degree=2)

    # Identify system using explicit polynomial basis functions
    print("Identifying system using explicit polynomial basis functions...")
    start_time = time.time()
    Gamma_explicit = compute_explicit_gamma(X_train, dX_train, fitted_basis_functions)
    system_identification_time = time.time() - start_time
    print(f"System identified in {system_identification_time:.2f} seconds.")

    identified_system = ExplicitSystem(Gamma_explicit, fitted_basis_functions)

    # Generate test data
    print("Generating test trajectories...")
    X_test, dX_test, t_test = generate_trajectories(num_test_trajectories, t_end, dt, a, b, c)
    print("Test trajectories generated.")

    # Predict using explicit system
    print("Simulating identified system on test data...")
    start_time = time.time()
    X_identified_test = np.zeros_like(X_test)
    for i in range(num_test_trajectories):
        X_identified_test[i, 0] = X_test[i, 0]  # Use true initial conditions
        for j in range(1, len(t_test)):
            x_prev = X_identified_test[i, j-1]
            dx = identified_system.F(x_prev)
            X_identified_test[i, j] = x_prev + dx * dt
    simulation_time = time.time() - start_time
    print(f"Simulation on test data completed in {simulation_time:.2f} seconds.")

    # Compute metrics
    metrics = compute_metrics(X_test, X_identified_test)
    print(metrics)

    # Calculate results
    rmse = np.sqrt(np.mean((X_test - X_identified_test)**2, axis=(1,2)))
    avg_rmse = np.mean(rmse)

    r_squared_list = []
    for i in range(num_test_trajectories):
        ss_res = np.sum((X_test[i] - X_identified_test[i])**2, axis=0)
        ss_tot = np.sum((X_test[i] - np.mean(X_test[i], axis=0))**2, axis=0)
        r2 = 1 - ss_res / ss_tot
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
    X_extrap, dX_extrap, t_extrap = generate_trajectories(num_test_trajectories, t_end * 1.5, dt, a, b, c)
    print("EXTRAPOLATION Test trajectories generated.")

    # Predict using explicit system
    print("Simulating identified system on extrapolation test data...")
    start_time = time.time()
    X_identified_extrap = np.zeros_like(X_extrap)
    for i in range(num_test_trajectories):
        X_identified_extrap[i, 0] = X_extrap[i, 0]  # Use true initial conditions
        for j in range(1, len(t_extrap)):
            x_prev = X_identified_extrap[i, j-1]
            dx = identified_system.F(x_prev)
            X_identified_extrap[i, j] = x_prev + dx * dt
    simulation_time = time.time() - start_time
    print(f"Simulation on extrapolation test data completed in {simulation_time:.2f} seconds.")

    # Compute metrics
    metrics_extrap = compute_metrics(X_extrap, X_identified_extrap)
    print(metrics_extrap)

    # Calculate results
    rmse_extrap = np.sqrt(np.mean((X_extrap - X_identified_extrap)**2, axis=(1,2)))
    avg_rmse_extrap = np.mean(rmse_extrap)

    r_squared_extrap_list = []
    for i in range(num_test_trajectories):
        ss_res = np.sum((X_extrap[i] - X_identified_extrap[i])**2, axis=0)
        ss_tot = np.sum((X_extrap[i] - np.mean(X_extrap[i], axis=0))**2, axis=0)
        r2 = 1 - ss_res / ss_tot
        r_squared_extrap_list.append(np.mean(r2))
    avg_r_squared_extrap = np.mean(r_squared_extrap_list)

    # Print results and visualize
    print_results(identified_system, avg_rmse_extrap, avg_r_squared_extrap)

    # Visualize trajectories
    print("Visualizing extrapolation test trajectories...")
    visualize_trajectories(X_extrap, X_identified_extrap, 'Comparison of True and Explicitly Identified Trajectories on Extrapolation Data')

if __name__ == "__main__":
    main()
