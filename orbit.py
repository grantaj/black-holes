import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Constants
G = 6.67430e-11  # Gravitational constant, m^3 kg^-1 s^-2
M1 = 5e30  # Mass of the first black hole, kg
M2 = 5e30  # Mass of the second black hole, kg
M3 = 5e30  # Mass of the third black hole, kg

# Initial conditions: positions (x1, y1, x2, y2, x3, y3) and velocities (vx1, vy1, vx2, vy2, vx3, vy3)
initial_conditions = [
    -1e11, 0, 1e11, 0, 0, 1e11,  # positions: x1, y1, x2, y2, x3, y3
    0, 1e4, 0, -1e4, -1e4, 0  # velocities: vx1, vy1, vx2, vy2, vx3, vy3
]

def equations(t, y):
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = y
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    
    # Accelerations due to gravity
    ax1 = G * M2 * (x2 - x1) / r12**3 + G * M3 * (x3 - x1) / r13**3
    ay1 = G * M2 * (y2 - y1) / r12**3 + G * M3 * (y3 - y1) / r13**3
    ax2 = G * M1 * (x1 - x2) / r12**3 + G * M3 * (x3 - x2) / r23**3
    ay2 = G * M1 * (y1 - y2) / r12**3 + G * M3 * (y3 - y2) / r23**3
    ax3 = G * M1 * (x1 - x3) / r13**3 + G * M2 * (x2 - x3) / r23**3
    ay3 = G * M1 * (y1 - y3) / r13**3 + G * M2 * (y2 - y3) / r23**3
    
    return [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]

# Time span for the simulation
t_span = (0, 1e8)  # seconds
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Reduced the number of frames

# Solve the differential equations
solution = solve_ivp(equations, t_span, initial_conditions, t_eval=t_eval, rtol=1e-8)

# Extract solutions
x1, y1, x2, y2, x3, y3 = solution.y[0], solution.y[1], solution.y[2], solution.y[3], solution.y[4], solution.y[5]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(6, 6))  # Reduced figure size
x_range = np.linspace(-2e11, 2e11, 200)  # Reduced grid resolution
y_range = np.linspace(-2e11, 2e11, 200)
X, Y = np.meshgrid(x_range, y_range)

# Initialize the color plot
wave_amplitude = np.zeros_like(X)

# Function to initialize the animation
def init():
    ax.clear()
    ax.set_xlim(-2e11, 2e11)
    ax.set_ylim(-2e11, 2e11)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Gravitational Waves from Three Black Holes')
    return ax

# Function to update the animation
def update(frame):
    ax.clear()
    
    # Current positions of the black holes
    pos1 = (x1[frame], y1[frame])
    pos2 = (x2[frame], y2[frame])
    pos3 = (x3[frame], y3[frame])
    
    # Accelerations at current positions
    acc1 = np.sqrt(solution.y[6, frame]**2 + solution.y[7, frame]**2)
    acc2 = np.sqrt(solution.y[8, frame]**2 + solution.y[9, frame]**2)
    acc3 = np.sqrt(solution.y[10, frame]**2 + solution.y[11, frame]**2)
    
    # Update the wave amplitude
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            r1 = np.sqrt((X[i, j] - pos1[0])**2 + (Y[i, j] - pos1[1])**2)
            r2 = np.sqrt((X[i, j] - pos2[0])**2 + (Y[i, j] - pos2[1])**2)
            r3 = np.sqrt((X[i, j] - pos3[0])**2 + (Y[i, j] - pos3[1])**2)
            
            # Simplified wave amplitude model
            wave_amplitude[i, j] = (acc1 / (r1 + 1e-10) + acc2 / (r2 + 1e-10) + acc3 / (r3 + 1e-10))
    
    ax.imshow(wave_amplitude, extent=[-2e11, 2e11, -2e11, 2e11], origin='lower', cmap='viridis')
    
    # Plot the black holes
    ax.plot(x1[:frame], y1[:frame], 'w-', label='Black Hole 1')
    ax.plot(x2[:frame], y2[:frame], 'r-', label='Black Hole 2')
    ax.plot(x3[:frame], y3[:frame], 'b-', label='Black Hole 3')
    ax.plot(pos1[0], pos1[1], 'wo')
    ax.plot(pos2[0], pos2[1], 'ro')
    ax.plot(pos3[0], pos3[1], 'bo')
    
    ax.legend()
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Gravitational Waves from Three Black Holes')
    ax.grid()
    
    return ax

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=False)

# Save the animation to a file
writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)  # Reduced fps
ani.save("gravitational_waves_black_holes.mp4", writer=writer)

# Display the plot (if you want to see it in a window)
plt.show()
