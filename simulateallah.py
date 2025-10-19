import fastf1
import os
from fastf1 import plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

fastf1.Cache.enable_cache('cache')  # Enable cache to speed up data loading

# Load the race session (e.g., 2023 Monaco GP)
session = fastf1.get_session(2023, 'Monaco', 'R')
session.load()

drivers = ['VER', 'HAM']  # Example drivers
telemetry_data = {}

for driver in drivers:
    lap = session.laps.pick_driver(driver).pick_fastest()
    telemetry = lap.get_telemetry()
    telemetry_data[driver] = telemetry

fig, ax = plt.subplots()
ax.set_title('F1 Race Simulation')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Plot the track layout using one driver's telemetry
track_x = telemetry_data[drivers[0]]['X']
track_y = telemetry_data[drivers[0]]['Y']
ax.plot(track_x, track_y, color='gray', linestyle='--', linewidth=1)

# Initialize scatter plots for each driver
scatters = {}
colors = ['red', 'blue']  # Assign colors to drivers

for driver, color in zip(drivers, colors):
    scatters[driver] = ax.scatter([], [], color=color, label=driver)

ax.legend()

# Determine the number of frames based on the shortest telemetry length
num_frames = min(len(data) for data in telemetry_data.values())

def update(frame):
    for driver in drivers:
        data = telemetry_data[driver]
        if frame < len(data):
            x = data['X'].iloc[frame]
            y = data['Y'].iloc[frame]
            scatters[driver].set_offsets([x, y])
    return scatters.values()
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
plt.show()
