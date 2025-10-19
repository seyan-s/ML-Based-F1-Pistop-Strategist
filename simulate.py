import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import joblib
import os

# Enable cache
cache_dir = 'cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# --- Setup ---
year = 2023
track_name = 'Silverstone'
driver_code = 'LEC'  # Charles Leclerc short code

# Load session
session = fastf1.get_session(year, track_name, 'R')
session.load()

laps_all = session.laps.pick_driver(driver_code)
actual_laps = laps_all.copy()
telemetry_base = actual_laps.pick_fastest().get_telemetry()

# --- Get actual pit stops ---
actual_pit_laps = actual_laps[actual_laps['PitInTime'].notnull()]['LapNumber'].values

# --- Predict pitstops ---
sample = pd.DataFrame([{
    'Track': 'Silverstone',
    'Driver': 'Charles Leclerc',
    'Compounds': 'Medium,Hard',
    'TrackType': 'Balanced',
    'Rainfall_max': False,
    'Year': 2023,
    'TyreLife_max': 22,
    'AirTemp_mean': 21.2,
    'TrackTemp_mean': 35.5,
    'Humidity_mean': 60.0,
    'TotalLaps_first': 52
}])

clf = joblib.load('pitstop_classifier.pkl')
reg1 = joblib.load('pitlap1_regressor.pkl')
reg2 = joblib.load('pitlap2_regressor.pkl')

n_stops = clf.predict(sample)[0]
pit1 = round(reg1.predict(sample)[0])
pit2 = round(reg2.predict(sample)[0]) if n_stops == 2 else None
predicted_pits = [pit1] + ([pit2] if pit2 else [])

# --- Simulate both strategies ---
actual_times = actual_laps['LapTime'].dt.total_seconds().dropna().values
predicted_strategy = actual_laps.copy()
for lap in predicted_strategy.itertuples():
    if lap.LapNumber in predicted_pits and lap.LapTime is not pd.NaT:
        predicted_strategy.at[lap.Index, 'LapTime'] += pd.to_timedelta(20, unit='s')  # add 20s as timedelta


predicted_times = predicted_strategy['LapTime'].dt.total_seconds().dropna().values

# --- Plot lap time comparison ---
plt.figure(figsize=(10, 5))
plt.plot(actual_laps['LapNumber'], actual_times, label='Actual Strategy', marker='o')
plt.plot(predicted_strategy['LapNumber'], predicted_times, label='Predicted Strategy', marker='x')

for lap in actual_pit_laps:
    plt.axvline(lap, color='green', linestyle='--', label='Actual Pit' if lap == actual_pit_laps[0] else "")
for lap in predicted_pits:
    plt.axvline(lap, color='blue', linestyle=':', label='Predicted Pit' if lap == predicted_pits[0] else "")

plt.title(f"{driver_code} – Lap Time Comparison")
plt.xlabel('Lap Number')
plt.ylabel('Lap Time (s)')
plt.legend()
plt.tight_layout()
plt.show()

# --- Track animation for both strategies ---
fig, ax = plt.subplots()
ax.set_title(f"{driver_code} – Track Simulation (Actual vs Predicted)")
ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.plot(telemetry_base['X'], telemetry_base['Y'], color='gray', linestyle='--', linewidth=1)

# Use fastest lap as base for animation (loop over it)
telemetry = telemetry_base
frames = len(telemetry)

scatter_actual = ax.scatter([], [], color='red', label='Actual Strategy')
scatter_pred = ax.scatter([], [], color='blue', label='Predicted Strategy')

# Mark pit stops
for lap in actual_pit_laps:
    lap_data = actual_laps[actual_laps['LapNumber'] == lap]
    if not lap_data.empty:
        try:
            tel = lap_data.iloc[0].get_telemetry()
            ax.plot(tel['X'].iloc[0], tel['Y'].iloc[0], 'go', label='Actual Pit' if lap == actual_pit_laps[0] else "")
        except:
            continue

for lap in predicted_pits:
    lap_data = actual_laps[actual_laps['LapNumber'] == lap]
    if not lap_data.empty:
        try:
            tel = lap_data.iloc[0].get_telemetry()
            ax.plot(tel['X'].iloc[0], tel['Y'].iloc[0], 'bo', label='Predicted Pit' if lap == predicted_pits[0] else "")
        except:
            continue

def update(i):
    x = telemetry['X'].iloc[i]
    y = telemetry['Y'].iloc[i]
    scatter_actual.set_offsets([x, y])
    scatter_pred.set_offsets([x + 2, y])  # offset slightly to show both
    return scatter_actual, scatter_pred

ax.legend()
ani = animation.FuncAnimation(fig, update, frames=frames, interval=1, blit=True)
plt.show()

# --- Final Comparison ---
total_actual_time = np.sum(actual_times)
total_predicted_time = np.sum(predicted_times)

print(f"🟢 Actual Strategy Total Time: {total_actual_time:.1f} seconds")
print(f"🔵 Predicted Strategy Total Time: {total_predicted_time:.1f} seconds")

if total_predicted_time < total_actual_time:
    print("✅ Predicted strategy is faster!")
else:
    print("❌ Actual strategy was better.")
