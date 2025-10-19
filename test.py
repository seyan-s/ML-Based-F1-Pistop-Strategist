import pandas as pd
import joblib

# === Load Models ===
count_model = joblib.load('models/pitstop_count_model.pkl')
lap_model = joblib.load('models/pitstop_lap_predictor_model.pkl')

# === Sample Input ===
sample_input = pd.DataFrame([{
    'Year': 2023,
    'TrackType': 'Balanced',
    'CircuitLength_first': 5.891,      # Silverstone length in km
    'TotalLaps_first': 52,
    'AirTemp_mean': 21.2,
    'TrackTemp_mean': 35.5,
    'Humidity_mean': 60.0,
    'Rainfall_max': 0
}])

# === Predict Number of Pitstops ===
predicted_stops = count_model.predict(sample_input)[0]
print(f"➡️ Predicted Number of Pitstops: {predicted_stops}")

# === Prepare Data for Lap Prediction ===
lap_predictions = []

for stop_number in range(1, predicted_stops + 1):
    lap_input = pd.DataFrame([{
        'Year': sample_input['Year'].values[0],
        'TrackType': sample_input['TrackType'].values[0],
        'CircuitLength': sample_input['CircuitLength_first'].values[0],
        'TotalLaps': sample_input['TotalLaps_first'].values[0],
        'AirTemp': sample_input['AirTemp_mean'].values[0],
        'TrackTemp': sample_input['TrackTemp_mean'].values[0],
        'Humidity': sample_input['Humidity_mean'].values[0],
        'Rainfall': sample_input['Rainfall_max'].values[0],
        'NumPitStops': predicted_stops,
        'StopNumber': stop_number
    }])

    predicted_lap = lap_model.predict(lap_input)[0]
    lap_predictions.append(predicted_lap)
    print(f"   ⛽ Pitstop {stop_number} at lap: {predicted_lap:.1f}")

    

# === Total Pitstop Duration (example output logic if needed) ===
if predicted_stops == 2:
    # lap_predictions[1]=lap_predictions[0]+lap_predictions[1]
    second_stint_length = lap_predictions[1] - lap_predictions[0]
    print(f"   🔁 Time between pitstops: {second_stint_length:.1f} laps")
