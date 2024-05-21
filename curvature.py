import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant, m^3 kg^-1 s^-2
c = 3.0e8  # Speed of light, m/s
M = 5e30  # Mass of the black hole, kg

# Schwarzschild radius
Rs = 2 * G * M / c**2

# Create a grid of points in the x-y plane
N = 500
x = np.linspace(-10 * Rs, 10 * Rs, N)
y = np.linspace(-10 * Rs, 10 * Rs, N)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Calculate the gravitational time dilation factor
Z = np.log10(1 - Rs / R)
Z[R <= Rs] = np.nan  # Mask out the region inside the Schwarzschild radius

# Plot the curvature
fig, ax = plt.subplots(figsize=(10, 10))
c = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Curvature of Spacetime around a Black Hole')
fig.colorbar(c, label='log(1 - Rs/R)')
ax.plot(0, 0, 'ro', label='Black Hole')
ax.legend()

# Save the plot to a file
plt.savefig("spacetime_curvature.png")

# Display the plot
plt.show()
