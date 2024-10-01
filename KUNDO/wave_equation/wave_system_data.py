import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 1.0  # wave speed
L = 10.0  # spatial domain length
T = 50.0  # total simulation time
dx = 0.1  # spatial step
dt = 0.01  # time step
x = np.arange(0, L + dx, dx)  # spatial grid
t = np.arange(0, T + dt, dt)  # time grid
Nx = len(x)
Nt = len(t)
num_series = 100  # number of time series to generate

# Calculate dx/dt sequence
dx_dt_sequence = np.full(Nt, dx/dt)

# Check stability condition (CFL condition)
if c * dt / dx > 1:
    raise ValueError("Stability condition not met. Decrease dt or increase dx.")

# Initialize solution array for all time series
all_series = np.zeros((num_series, Nt, Nx))

# Generate multiple time series
for series in range(num_series):
    # Initialize solution array for this series
    u = np.zeros((Nt, Nx))

    # Set random initial conditions for this series
    u[0, :] = np.random.rand(Nx)  # random initial displacement
    u[1, :] = u[0, :] + dt * np.random.rand(Nx)  # random initial velocity

    # Solve wave equation using finite difference method
    for n in range(1, Nt - 1):
        u[n+1, 1:-1] = 2*u[n, 1:-1] - u[n-1, 1:-1] + \
                       (c*dt/dx)**2 * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
        
        # Apply periodic boundary conditions
        u[n+1, 0] = u[n+1, -2]
        u[n+1, -1] = u[n+1, 1]
    
    # Store this series in the all_series array
    all_series[series] = u

# Plot a few snapshots of the first time series
plt.figure(figsize=(12, 8))
for i, t_val in enumerate([0, 10, 25, 49]):
    plt.subplot(2, 2, i+1)
    plt.plot(x, all_series[0, int(t_val/dt), :])
    plt.title(f't = {t_val}')
    plt.xlabel('x')
    plt.ylabel('u')
plt.tight_layout()
plt.show()

# Save data
np.savez('wave_equation_data_100series.npz', 
         series_data=all_series, 
         dx_dt_sequence=dx_dt_sequence,
         x=x, 
         t=t, 
         parameters={'c': c, 'L': L, 'T': T, 'dx': dx, 'dt': dt})

print(f"Data shape: {all_series.shape}")
print("dx/dt sequence shape:", dx_dt_sequence.shape)
print("Spatial grid shape:", x.shape)
print("Time grid shape:", t.shape)
print("Data saved as 'wave_equation_data_100series.npz'")
