import fastf1
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import joblib
import os

cache_dir = 'cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

year = 2023
track_name = 'Silverstone'
driver_code = 'LEC'

session = fastf1.get_session(year, track_name, 'R')
session.load()

laps_all = session.laps.pick_driver(driver_code)
actual_laps = laps_all.copy()
telemetry_base = actual_laps.pick_fastest().get_telemetry()

actual_pit_laps = actual_laps[actual_laps['PitInTime'].notnull()]['LapNumber'].values

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
print(f"Recommended pitstops: {n_stops} stop(s)")
print(f"Pit Lap 1: {pit1:.1f}")
if pit2:
    print(f"Pit Lap 2: {pit2:.1f}")

actual_times = actual_laps['LapTime'].dt.total_seconds().dropna().values
predicted_strategy = actual_laps.copy()
for lap in predicted_strategy.itertuples():
    if lap.LapNumber in predicted_pits and lap.LapTime is not pd.NaT:
        predicted_strategy.at[lap.Index, 'LapTime'] += pd.to_timedelta(20, unit='s')

predicted_times = predicted_strategy['LapTime'].dt.total_seconds().dropna().values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(actual_laps['LapNumber'], actual_times, label='Actual Strategy', marker='o')
for lap in actual_pit_laps:
    ax1.axvline(lap, color='green', linestyle='--', label='Actual Pit' if lap == actual_pit_laps[0] else "")
ax1.set_title(f"{driver_code} – Actual Strategy Lap Times")
ax1.set_xlabel('Lap Number')
ax1.set_ylabel('Lap Time (s)')
ax1.legend()

ax2.plot(predicted_strategy['LapNumber'], predicted_times, label='Predicted Strategy', marker='x')
for lap in predicted_pits:
    ax2.axvline(lap, color='blue', linestyle=':', label='Predicted Pit' if lap == predicted_pits[0] else "")
ax2.set_title(f"{driver_code} – Predicted Strategy Lap Times")
ax2.set_xlabel('Lap Number')
ax2.set_ylabel('Lap Time (s)')
ax2.legend()

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.set_title(f"{driver_code} – Actual Strategy Track Simulation")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.plot(telemetry_base['X'], telemetry_base['Y'], color='gray', linestyle='--', linewidth=1)

ax2.set_title(f"{driver_code} – Predicted Strategy Track Simulation")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.plot(telemetry_base['X'], telemetry_base['Y'], color='gray', linestyle='--', linewidth=1)

scatter_actual = ax1.scatter([], [], color='red', label='Actual Strategy')
scatter_pred = ax2.scatter([], [], color='blue', label='Predicted Strategy')

for lap in actual_pit_laps:
    lap_data = actual_laps[actual_laps['LapNumber'] == lap]
    if not lap_data.empty:
        try:
            tel = lap_data.iloc[0].get_telemetry()
            ax1.plot(tel['X'].iloc[0], tel['Y'].iloc[0], 'go', label='Actual Pit' if lap == actual_pit_laps[0] else "")
        except:
            continue

for lap in predicted_pits:
    lap_data = actual_laps[actual_laps['LapNumber'] == lap]
    if not lap_data.empty:
        try:
            tel = lap_data.iloc[0].get_telemetry()
            ax2.plot(tel['X'].iloc[0], tel['Y'].iloc[0], 'bo', label='Predicted Pit' if lap == predicted_pits[0] else "")
        except:
            continue

def update(i):
    x_actual = telemetry_base['X'].iloc[i]
    y_actual = telemetry_base['Y'].iloc[i]
    scatter_actual.set_offsets([x_actual, y_actual])

    x_pred = x_actual + 2
    y_pred = y_actual
    scatter_pred.set_offsets([x_pred, y_pred])

    return scatter_actual, scatter_pred

frames = len(telemetry_base)
ani = animation.FuncAnimation(fig, update, frames=frames, interval=1, blit=True)

plt.tight_layout()
plt.show()

total_actual_time = np.sum(actual_times)
total_predicted_time = np.sum(predicted_times)

print(f"Predicted Strategy Total Time: {total_actual_time:.1f} seconds")
print(f"Actual Strategy Total Time: {total_predicted_time:.1f} seconds")

if total_predicted_time > total_actual_time:
    print("Predicted strategy is faster!")
else:
    print("Actual strategy was better.")
