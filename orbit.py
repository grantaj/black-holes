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
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solve the differential equations
solution = solve_ivp(equations, t_span, initial_conditions, t_eval=t_eval, rtol=1e-8)

# Extract solutions
x1, y1, x2, y2, x3, y3 = solution.y[0], solution.y[1], solution.y[2], solution.y[3], solution.y[4], solution.y[5]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(min(x1.min(), x2.min(), x3.min()), max(x1.max(), x2.max(), x3.max()))
ax.set_ylim(min(y1.min(), y2.min(), y3.min()), max(y1.max(), y2.max(), y3.max()))
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Orbital Dynamics of Three Black Holes')
ax.grid()

# Initialize the lines for the three black holes
line1, = ax.plot([], [], label='Black Hole 1')
line2, = ax.plot([], [], label='Black Hole 2')
line3, = ax.plot([], [], label='Black Hole 3')
ax.legend()

# Function to initialize the animation
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3

# Function to update the animation
def update(frame):
    line1.set_data(x1[:frame], y1[:frame])
    line2.set_data(x2[:frame], y2[:frame])
    line3.set_data(x3[:frame], y3[:frame])
    return line1, line2, line3

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True)

# Save the animation to a file
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
ani.save("three_black_holes_orbits.mp4", writer=writer)

# Display the plot (if you want to see it in a window)
plt.show()

