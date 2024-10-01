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


# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out


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
def prepare_data(X, t, seq_length=10):
    X_seq = []
    y_seq = []
    for traj in X:
        for i in range(len(t) - seq_length):
            X_seq.append(traj[i:i+seq_length])
            y_seq.append(traj[i+seq_length])
    return np.array(X_seq), np.array(y_seq)


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
    seq_length = 10  # Sequence length
    
    # System parameters (default Saddle system)
    a = 1.0
    b = -1.0
    c = -1.0
    
    # Generate training data
    print("Generating training trajectories...")
    X_train, t_train = generate_trajectories(num_train_trajectories, t_end, dt, a, b, c)
    print("Training trajectories generation complete.")
    
    # Prepare training data
    X_train_seq, y_train = prepare_data(X_train, t_train, seq_length)
    
    # Convert to Tensor
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    # Define model, loss function and optimizer
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    num_epochs = 1000
    print("Starting LSTM model training...")
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
    
    print("LSTM model training complete.")
    
    # Generate test data
    print("Generating test trajectories...")
    X_test, t_test = generate_trajectories(num_test_trajectories, t_end, dt, a, b, c)
    print("Test trajectories generation complete.")
    
    # Prepare test data
    X_test_seq, y_test = prepare_data(X_test, t_test, seq_length)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Model prediction
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()
    y_true = y_test
    
    # Calculate metrics
    metrics = compute_metrics(y_true, y_pred)
    print(f"Test set performance metrics: {metrics}")
    
    # Visualize partial test trajectories
    for i in range(num_test_trajectories):
        true_traj = X_test[i, seq_length:]
        pred_traj = y_pred[i* (len(t_test) - seq_length): (i+1)* (len(t_test) - seq_length)]
        visualize_trajectories(true_traj, pred_traj, f'Test Trajectory {i+1}')
    

if __name__ == "__main__":
    main()
