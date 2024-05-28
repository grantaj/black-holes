import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
num_epicycles = 50  # Number of epicycles
time_steps = 2000  # Number of time steps for the animation

# Generate a more interesting complex curve using filtered noise
def generate_filtered_noise(length, scale=10):
    noise = np.cumsum(np.random.randn(length))  # Cumulative sum to create a random walk
    noise = noise - np.mean(noise)  # Remove DC component
    noise = noise / np.max(np.abs(noise))  # Normalize
    noise = np.convolve(noise, np.ones(scale)/scale, mode='same')  # Apply a simple moving average filter
    return noise

t = np.linspace(0, 2 * np.pi, time_steps)
x = generate_filtered_noise(time_steps, scale=20)
y = generate_filtered_noise(time_steps, scale=20)

# Fourier coefficients
z = x + 1j * y
fourier_coefficients = np.fft.fft(z) / len(t)
frequencies = np.fft.fftfreq(len(t), d=t[1] - t[0])

# Sort coefficients by magnitude for better visualization
indices = np.argsort(-np.abs(fourier_coefficients))
fourier_coefficients = fourier_coefficients[indices]
frequencies = frequencies[indices]

# Function to compute the position of each epicycle at a given time
def compute_epicycles(t, coeffs, freqs, num_cycles):
    x, y = 0, 0
    positions = [(x, y)]
    for i in range(num_cycles):
        coeff = coeffs[i]
        freq = freqs[i]
        x += np.real(coeff) * np.cos(freq * t) - np.imag(coeff) * np.sin(freq * t)
        y += np.real(coeff) * np.sin(freq * t) + np.imag(coeff) * np.cos(freq * t)
        positions.append((x, y))
    return positions

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')

# Plot elements
lines = [ax.plot([], [], 'r-')[0] for _ in range(num_epicycles)]
points, = ax.plot([], [], 'bo')
trace, = ax.plot([], [], 'g-')

# Initialize the animation
def init():
    for line in lines:
        line.set_data([], [])
    points.set_data([], [])
    trace.set_data([], [])
    return lines + [points, trace]

# Update function for the animation
def update(frame):
    positions = compute_epicycles(frame * 2 * np.pi / time_steps, fourier_coefficients, frequencies, num_epicycles)
    trace.set_data(x[:frame], y[:frame])
    for i, line in enumerate(lines):
        if i == 0:
            line.set_data([0, positions[i][0]], [0, positions[i][1]])
        else:
            line.set_data([positions[i-1][0], positions[i][0]], [positions[i-1][1], positions[i][1]])
    points.set_data([positions[-1][0]], [positions[-1][1]])
    return lines + [points, trace]

# Create the animation
ani = FuncAnimation(fig, update, frames=time_steps, init_func=init, blit=True, interval=20)

# Display the animation
plt.show()





