

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=3):
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
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch, seq_length, hidden_size)
        # Decode the last time step
        out = self.fc(out[:, -1, :])
        return out

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
    sequence_length = 10  # Number of past steps to use for prediction

    # Control input function
    def u(t):
        return np.sin(t)

    # Generate training data
    print("Generating training trajectories...")
    X_train, U_train, dX_train, t_train = generate_trajectories(num_train_trajectories, t_end, dt, u)
    print("Training trajectories generated.")

    # Prepare dataset for LSTM
    def create_sequences(X, U, dX, seq_length):
        sequences = []
        targets = []
        for traj, u_traj, dX_traj in zip(X, U, dX):
            for i in range(len(t_train) - seq_length):
                seq_x = traj[i:i+seq_length]
                seq_u = u_traj[i:i+seq_length].reshape(-1, 1)
                seq_input = np.hstack([seq_x, seq_u])
                sequences.append(seq_input)
                targets.append(dX_traj[i+seq_length])
        return np.array(sequences), np.array(targets)

    print("Preparing training sequences...")
    sequences, targets = create_sequences(X_train, U_train, dX_train, sequence_length)
    X_tensor = torch.tensor(sequences, dtype=torch.float32)
    y_tensor = torch.tensor(targets, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize model, loss function, optimizer
    model = LSTMModel(input_size=4, hidden_size=128, num_layers=2, output_size=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    print("Starting training...")
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    # Generate test data
    print("Generating test trajectories...")
    X_test, U_test, dX_test, t_test = generate_trajectories(num_test_trajectories, t_end, dt, u)
    print("Test trajectories generated.")

    # Prepare test sequences
    print("Preparing test sequences...")
    test_sequences, test_targets = create_sequences(X_test, U_test, dX_test, sequence_length)
    X_test_tensor = torch.tensor(test_sequences, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(test_targets, dtype=torch.float32).to(device)

    # Predict on test data
    print("Evaluating on test data...")
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()
        y_true = test_targets
        metrics = compute_metrics(y_true, predictions)
    print(f"Test MSE: {metrics['MSE']:.6f}, R2: {metrics['R2']:.6f}")

    # Visualize some predictions
    num_plots = 3  # Number of trajectories to plot
    plt.figure(figsize=(15, 5))
    for i in range(num_plots):
        plt.subplot(1, 3, i+1)
        plt.plot(y_true[i*len(t_test)//num_plots:(i+1)*len(t_test)//num_plots, 0], label='True dx')
        plt.plot(predictions[i*len(t_test)//num_plots:(i+1)*len(t_test)//num_plots, 0], label='Predicted dx')
        plt.legend()
        plt.xlabel('Time Step')
        plt.ylabel('dx')
    plt.suptitle('LSTM Predictions vs True Dynamics')
    plt.show()

if __name__ == "__main__":
    main()
