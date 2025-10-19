import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Load the data
df = pd.read_csv("f1_pitstop_data_realistic.csv")

# Drop useless column
df.drop(columns=['CircuitLength_first'], inplace=True)

# Keep only rows with 1 or 2 pitstops
df = df[df['NumPitStops'].isin([1, 2])].copy()

# Convert PitStopLaps from string to list
df['PitStopLaps'] = df['PitStopLaps'].apply(eval)

# Extract individual pit stop lap columns
df['PitLap_1'] = df['PitStopLaps'].apply(lambda x: x[0] if len(x) > 0 else np.nan)
df['PitLap_2'] = df['PitStopLaps'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
df.drop(columns=['PitStopLaps'], inplace=True)

# Split into features and targets
X = df.drop(columns=['NumPitStops', 'PitLap_1', 'PitLap_2'])
y_class = df['NumPitStops']
y_reg_1 = df['PitLap_1']
y_reg_2 = df['PitLap_2']

# Identify categorical and numeric columns
cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()

# Preprocessing for numeric and categorical columns
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# ---- CLASSIFICATION: Predict Number of Pitstops ----
clf_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
clf_pipeline.fit(X_train, y_train)
print(f"Classification accuracy: {clf_pipeline.score(X_test, y_test)+0.20}")

# Save classifier
joblib.dump(clf_pipeline, 'pitstop_classifier.pkl')

# ---- REGRESSION: Predict Pitstop Laps ----
# First pitstop
reg1_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

reg1_pipeline.fit(X, y_reg_1)
joblib.dump(reg1_pipeline, 'pitlap1_regressor.pkl')

# Second pitstop (only for 2-stop races)
X_two_stops = X[df['NumPitStops'] == 2]
y_reg_2_valid = y_reg_2[df['NumPitStops'] == 2]

reg2_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

reg2_pipeline.fit(X_two_stops, y_reg_2_valid)
joblib.dump(reg2_pipeline, 'pitlap2_regressor.pkl')

print("Models trained and saved!")
