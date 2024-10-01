import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define simplified Neural Turing Machine
class NTMController(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, output_size=3, memory_units=128, memory_dim=20):
        super(NTMController, self).__init__()
        self.hidden_size = hidden_size
        self.memory_units = memory_units
        self.memory_dim = memory_dim
        
        self.memory = torch.randn(memory_units, memory_dim).to(device)
        self.memory.requires_grad = False  # Fixed memory
        
        self.fc = nn.Linear(input_size + memory_dim, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Simple read mechanism: find the most similar memory unit
        similarity = torch.matmul(self.memory, x.unsqueeze(1)).squeeze(1)  # (memory_units)
        attn_weights = torch.softmax(similarity, dim=0).unsqueeze(1)  # (memory_units, 1)
        read_vector = torch.matmul(self.memory.t(), attn_weights).squeeze(1)  # (memory_dim)
        
        # Combine input and read vector
        combined = torch.cat([x, read_vector], dim=0)
        hidden = self.relu(self.fc(combined))
        output = self.output_layer(hidden)
        return output


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
    
    for _ in range(num_trajectories):
        x0 = np.random.uniform(-0.5, 0.5, 3)  # Random initial conditions
        X = odeint(non_linear_system, x0, t, args=(a, b, c))
        X_batch.append(X)
    
    return np.array(X_batch), t


# Data preprocessing
def prepare_data(X, t):
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    t_tensor = torch.tensor(t, dtype=torch.float32).to(device)
    return X_tensor, t_tensor


# Compute performance metrics
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
    ax.plot(X_pred[:,0], X_pred[:,1], X_pred[:,2], label='Predicted')
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
        ax2.plot(X_pred[:,i], label=f'Predicted {labels[i]}')
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
    X_train, t_train = generate_trajectories(num_train_trajectories, t_end, dt, a, b, c)
    print("Training trajectories generated.")
    
    # Prepare training data
    X_train_tensor, t_train_tensor = prepare_data(X_train, t_train)
    
    # Initialize model, loss function, and optimizer
    model = NTMController().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train model
    num_epochs = 1000
    print("Starting Neural Turing Machine model training...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        outputs = model(X_train_tensor.view(-1, 3))
        # Compute loss
        loss = criterion(outputs, X_train_tensor.view(-1, 3))
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
    
    print("Neural Turing Machine model training completed.")
    
    # Generate test data
    print("Generating test trajectories...")
    X_test, t_test = generate_trajectories(num_test_trajectories, t_end, dt, a, b, c)
    print("Test trajectories generated.")
    
    # Prepare test data
    X_test_tensor, t_test_tensor = prepare_data(X_test, t_test)
    
    # Model prediction
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor.view(-1, 3)).cpu().numpy()
    y_true = X_test.reshape(-1, 3)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    print(f"Test set performance metrics: {metrics}")
    
    # Visualize some test trajectories
    for i in range(num_test_trajectories):
        true_traj = X_test[i]
        pred_traj = y_pred[i*(len(t_test)): (i+1)*(len(t_test))]
        visualize_trajectories(true_traj, pred_traj, f'Test Trajectory {i+1}')


if __name__ == "__main__":
    main()
