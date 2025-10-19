import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Make sure the output directory exists
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Load the preprocessed data
print("Loading preprocessed data...")
features_df = pd.read_csv('f1_pitstop_data_realistic.csv')

# Convert PitStopLaps from string to list
print("Converting string representations to Python objects...")
features_df['PitStopLaps'] = features_df['PitStopLaps'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Check for NaN values
print("Checking data quality...")
print(f"Number of NaN values: {features_df.isna().sum().sum()}")
print(f"Number of duplicate races/drivers: {features_df.duplicated(subset=['Year', 'Track', 'Driver']).sum()}")

# Fill NaN values directly in the DataFrame
print("Filling missing values...")
numerical_cols = ['CircuitLength_first', 'TotalLaps_first', 'AirTemp_mean', 'TrackTemp_mean', 'Humidity_mean']
for col in numerical_cols:
    if col in features_df.columns:
        features_df[col].fillna(features_df[col].mean(), inplace=True)

# Convert boolean to int for Rainfall_max if it's boolean
if 'Rainfall_max' in features_df.columns:
    # Check if Rainfall_max is boolean
    if features_df['Rainfall_max'].dtype == bool:
        features_df['Rainfall_max'] = features_df['Rainfall_max'].astype(int)
    features_df['Rainfall_max'].fillna(0, inplace=True)

# Fill missing categorical values
if 'TrackType' in features_df.columns:
    most_common = features_df['TrackType'].mode()[0]
    features_df['TrackType'].fillna(most_common, inplace=True)

# Define features and targets
print("Preparing features and targets...")
X = features_df[[
    'Year', 'TrackType', 'CircuitLength_first', 'TotalLaps_first',
    'AirTemp_mean', 'TrackTemp_mean', 'Humidity_mean', 'Rainfall_max'
]]

# Target for number of pitstops
y_num_pitstops = features_df['NumPitStops']

# Basic data exploration
print(f"\nData shape: {X.shape}")
print(f"Number of unique tracks: {features_df['Track'].nunique()}")
print(f"Number of unique track types: {features_df['TrackType'].nunique()}")
print(f"Pitstops distribution: \n{y_num_pitstops.value_counts()}")

# Check for any remaining NaN values in X
print(f"Remaining NaN values in features: {X.isna().sum().sum()}")

# Prepare for model building
categorical_features = ['TrackType']
numerical_features = ['Year', 'CircuitLength_first', 'TotalLaps_first', 
                      'AirTemp_mean', 'TrackTemp_mean', 'Humidity_mean']
binary_features = ['Rainfall_max']

# Create preprocessor with explicit imputation steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features),
        ('bin', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('pass', 'passthrough')
        ]), binary_features)
    ])

# Split data
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train_num, y_test_num = train_test_split(
    X, y_num_pitstops, test_size=0.2, random_state=42)

# Debug: Check for NaN values in training data
print(f"NaN values in X_train before preprocessing: {X_train.isna().sum().sum()}")

# For debugging, add this code to check the transformed data
# Apply the preprocessor to a small sample and check for NaN values
X_sample = X_train.iloc[:5].copy()
print("Sample input data:")
print(X_sample)
try:
    X_transformed = preprocessor.fit_transform(X_sample)
    print(f"Transformed data shape: {X_transformed.shape}")
    print(f"Any NaN in transformed data: {np.isnan(X_transformed).any()}")
except Exception as e:
    print(f"Error during preprocessing sample: {e}")

# Create pipeline with the correct preprocessor
pitstop_count_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Simple parameter grid for GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 15],
    'classifier__min_samples_split': [2, 5]
}

# GridSearchCV for hyperparameter tuning
print("Performing grid search for best parameters...")
grid_search = GridSearchCV(pitstop_count_pipe, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train_num)

# Best model
best_pitstop_count_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate on test set
test_accuracy = best_pitstop_count_model.score(X_test, y_test_num)
print(f"Test accuracy for pitstop count model: {test_accuracy:.4f}")

# MODEL 2: Pitstop Lap Predictor
print("\nTraining pitstop lap prediction model...")

# The rest of your code for lap prediction...

# Function to create training data for lap prediction
def create_lap_prediction_data(features_df):
    X_laps = []
    y_laps = []
    
    for _, row in features_df.iterrows():
        # Skip if no pitstops or if PitStopLaps is NaN
        if row['NumPitStops'] == 0 or not isinstance(row['PitStopLaps'], list):
            continue
            
        # Get race features
        race_features = [
            row['Year'], row['TrackType'], row['CircuitLength_first'],
            row['TotalLaps_first'], row['AirTemp_mean'], row['TrackTemp_mean'],
            row['Humidity_mean'], row['Rainfall_max'], row['NumPitStops']
        ]
        
        # For each pitstop, create a prediction instance
        for stop_num, lap in enumerate(row['PitStopLaps'], 1):
            # Add which stop number this is (1st, 2nd, etc.)
            features = race_features + [stop_num]
            X_laps.append(features)
            y_laps.append(lap)
    
    # Convert to DataFrame
    X_laps_df = pd.DataFrame(X_laps, columns=[
        'Year', 'TrackType', 'CircuitLength', 'TotalLaps',
        'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall',
        'NumPitStops', 'StopNumber'
    ])
    
    return X_laps_df, np.array(y_laps)

# Create lap prediction data
print("Creating lap prediction training data...")
X_laps_df, y_laps = create_lap_prediction_data(features_df)
print(f"Lap prediction data shape: {X_laps_df.shape}")

# Split the data
X_train_laps, X_test_laps, y_train_laps, y_test_laps = train_test_split(
    X_laps_df, y_laps, test_size=0.2, random_state=42)

categorical_features_laps = ['TrackType']
numerical_features_laps = ['Year', 'CircuitLength', 'TotalLaps',
                           'AirTemp', 'TrackTemp', 'Humidity',
                           'NumPitStops', 'StopNumber']
binary_features_laps = ['Rainfall']

preprocessor_laps = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_features_laps),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features_laps),
        ('bin', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('pass', 'passthrough')
        ]), binary_features_laps)
    ])

# Create pipeline for lap prediction
lap_predictor_pipe = Pipeline([
    ('preprocessor', preprocessor_laps),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Parameter grid for GridSearchCV
param_grid_laps = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 15],
    'regressor__min_samples_split': [2, 5]
}

# GridSearchCV for hyperparameter tuning
print("Performing grid search for lap predictor...")
grid_search_laps = GridSearchCV(lap_predictor_pipe, param_grid_laps, cv=5, scoring='neg_mean_squared_error')
grid_search_laps.fit(X_train_laps, y_train_laps)

# Best model
best_lap_predictor_model = grid_search_laps.best_estimator_
print(f"Best parameters for lap predictor: {grid_search_laps.best_params_}")

# Evaluate on test set
test_rmse = np.sqrt(-grid_search_laps.score(X_test_laps, y_test_laps))
print(f"RMSE for pitstop lap prediction: {test_rmse:.4f}")

# Calculate and plot feature importance
def plot_feature_importance(model, feature_names, title, filename):
    """Plot feature importance for a model"""
    try:
        if 'classifier' in model.named_steps:
            importances = model.named_steps['classifier'].feature_importances_
        elif 'regressor' in model.named_steps:
            importances = model.named_steps['regressor'].feature_importances_
        else:
            print(f"Could not find feature importances in model")
            return None
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"plots/{filename}.png")
        plt.close()
        
        return importance_df
    except Exception as e:
        print(f"Error plotting feature importance: {e}")
        return None

# Create correlation heatmap of numerical features
print("\nCreating correlation heatmap of features...")
numerical_df = features_df[numerical_features + ['NumPitStops']].copy()
plt.figure(figsize=(10, 8))
correlation = numerical_df.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, fmt=".2f", mask=mask, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

# Create distribution plots for number of pitstops
plt.figure(figsize=(8, 6))
sns.countplot(x=y_num_pitstops)
plt.title("Distribution of Number of Pitstops")
plt.xlabel("Number of Pitstops")
plt.ylabel("Count")
plt.savefig("plots/pitstop_distribution.png")
plt.close()

# Examine the relationship between track type and number of pitstops
plt.figure(figsize=(12, 6))
sns.boxplot(x='TrackType', y='NumPitStops', data=features_df)
plt.title("Number of Pitstops by Track Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/pitstops_by_track.png")
plt.close()

# Examine the relationship between weather and number of pitstops
plt.figure(figsize=(10, 6))
sns.scatterplot(x='AirTemp_mean', y='NumPitStops', hue='Rainfall_max', data=features_df)
plt.title("Number of Pitstops vs Air Temperature (Colored by Rainfall)")
plt.savefig("plots/pitstops_vs_temp.png")
plt.close()

# Save the models
print("\nSaving trained models...")
joblib.dump(best_pitstop_count_model, 'models/pitstop_count_model.pkl')
joblib.dump(best_lap_predictor_model, 'models/pitstop_lap_predictor_model.pkl')

print("\nTraining and model saving complete!")
print("Models saved in 'models/' directory")
print("Visualization plots saved in 'plots/' directory")

# Save a simplified feature importance for manual inspection
try:
    # For pitstop count model
    feature_names = list(X.columns)
    importances = best_pitstop_count_model.named_steps['classifier'].feature_importances_
    all_features = []
    
    # Get feature names after one-hot encoding
    for i, feature in enumerate(feature_names):
        if feature == 'TrackType':
            # This will be one-hot encoded
            for track_type in features_df['TrackType'].unique():
                all_features.append(f"TrackType_{track_type}")
        else:
            all_features.append(feature)
    
    # Save feature importance to CSV
    count_importance = pd.DataFrame({
        'Feature': all_features[:len(importances)],
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    count_importance.to_csv('plots/count_feature_importance.csv', index=False)
    print("Feature importance data saved to CSV")
except Exception as e:
    print(f"Error saving feature importance: {e}")