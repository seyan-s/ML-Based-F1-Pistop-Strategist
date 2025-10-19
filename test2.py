import pandas as pd
import joblib

sample = pd.DataFrame([{
    'Track': 'Silverstone',
    'Driver': 'Charles Leclerc',
    'Compounds': 'Medium,Hard',
    'TrackType': 'Balanced',
    'Rainfall_max': False,
    'Year': 2023,
    'TyreLife_max': 22,          # medium/softs degrade faster
    'AirTemp_mean': 21.2,        # typical UK summer
    'TrackTemp_mean': 35.5,      # moderate
    'Humidity_mean': 60.0,       # average
    'TotalLaps_first': 52        # standard full race
}])


# Load models
clf = joblib.load('pitstop_classifier.pkl')
reg1 = joblib.load('pitlap1_regressor.pkl')
reg2 = joblib.load('pitlap2_regressor.pkl')

# Predict number of pitstops
n_stops = clf.predict(sample)[0]

# Predict first pit stop
pit1 = reg1.predict(sample)[0]

# Predict second pit stop if needed
if n_stops == 2:
    pit2 = reg2.predict(sample)[0]
else:
    pit2 = None


print(f"Recommended pitstops: {n_stops} stop(s)")
print(f"   Pit Lap 1: {pit1:.1f}")
if pit2:
    print(f"   Pit Lap 2: {pit2:.1f}")
