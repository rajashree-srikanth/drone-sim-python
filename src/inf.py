import numpy as np
import matplotlib.pyplot as plt

# Define the parametric equations for the infinity shape
t = np.linspace(0, 2 * np.pi, 1000)
x = 12.5 * np.cos(t) + 87.5
y = 10 * np.sin(2 * t) + 30

# Define the extreme end points
extreme_points = [(75, 40), (100, 20)]

# Plotting the lemniscate (infinity shape)
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Infinity Shape', color='blue')
plt.scatter(*zip(*extreme_points), color='red', zorder=5, label='Extreme End Points')

# Mark the extreme points with coordinates
for point in extreme_points:
    plt.text(point[0], point[1], f'{point}', fontsize=10, ha='right')

plt.title('Visualization of Infinity Shape with Specified Endpoints')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.axis('equal')
plt.show()