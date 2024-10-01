
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from scipy.integrate import odeint as scipy_odeint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lorenz system definition for data generation
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
        X = scipy_odeint(lorenz_system, x0, t, args=(u_func,))
        U = u_func(t)
        dX = np.array([lorenz_system(x, ti, u_func) for ti, x in zip(t, X)])
        X_batch.append(X)
        U_batch.append(U)
        dX_batch.append(dX)

    return np.array(X_batch), np.array(U_batch), np.array(dX_batch), t

# Control input function
def u(t):
    return np.sin(t)

# Encoder for Latent Neural ODE
class Encoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, z_dim=64):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, z_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ODE function for latent space
class ODEFunc(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=128):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, z_dim)
        )
    
    def forward(self, t, z):
        return self.net(z)

# Decoder from latent space
class Decoder(nn.Module):
    def __init__(self, z_dim=64, hidden_dim=128, output_dim=3):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z):
        return self.net(z)

# Full Latent Neural ODE Model
class LatentNeuralODE(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, z_dim=64):
        super(LatentNeuralODE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.odefunc = ODEFunc(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim, output_dim=3)
    
    def forward(self, x, t):
        z0 = self.encoder(x)
        z = odeint(self.odefunc, z0, t)
        x_pred = self.decoder(z)
        return x_pred

# Compute Metrics
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'R2': r2}

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

    # Prepare data for Neural ODE
    X_train_flat = X_train.reshape(-1, 3)
    U_train_flat = U_train.reshape(-1, 1)
    inputs_train = np.hstack([X_train_flat, U_train_flat])
    inputs_train = torch.tensor(inputs_train, dtype=torch.float32).to(device)
    targets_train = torch.tensor(dX_train.reshape(-1, 3), dtype=torch.float32).to(device)
    t_train_tensor = torch.linspace(0, t_end, int(t_end/dt)+1).to(device)

    # Initialize model, optimizer, and loss function
    model = LatentNeuralODE(input_dim=4, hidden_dim=128, z_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 100
    print("Starting training...")
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        output = model(inputs_train, t_train_tensor)
        loss = criterion(output, targets_train)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # Generate test data
    print("Generating test trajectories...")
    X_test, U_test, dX_test, t_test = generate_trajectories(num_test_trajectories, t_end, dt, u)
    print("Test trajectories generated.")

    # Prepare test data
    X_test_flat = X_test.reshape(-1, 3)
    U_test_flat = U_test.reshape(-1, 1)
    inputs_test = np.hstack([X_test_flat, U_test_flat])
    inputs_test = torch.tensor(inputs_test, dtype=torch.float32).to(device)
    targets_test = dX_test.reshape(-1, 3)

    # Predict on test data
    print("Evaluating on test data...")
    model.eval()
    with torch.no_grad():
        output_test = model(inputs_test, t_train_tensor).cpu().numpy()
        y_true = targets_test
        y_pred = output_test
        metrics = compute_metrics(y_true, y_pred)
    print(f"Test MSE: {metrics['MSE']:.6f}, R2: {metrics['R2']:.6f}")

    # Reshape for visualization
    X_identified_test = y_pred.reshape(num_test_trajectories, -1, 3)

    # Visualize trajectories
    for i in range(num_test_trajectories):
        plt.figure(figsize=(12, 6))
        plt.plot(X_test[i,:,0], X_test[i,:,1], label='True Trajectory')
        plt.plot(X_identified_test[i,:,0], X_identified_test[i,:,1], label='Identified Trajectory')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'Trajectory {i+1}')
        plt.show()

if __name__ == "__main__":
    main()
