import os
import fastf1
import pandas as pd
import numpy as np

# Create cache directory if it doesn't exist
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"Created cache directory: {cache_dir}")

# Enable cache
fastf1.Cache.enable_cache(cache_dir)

def collect_race_data(years=range(2021, 2024)):
    """Collect and combine race data from multiple years"""
    all_race_data = []
    
    for year in years:
        # Get schedule for the year
        schedule = fastf1.get_event_schedule(year)
        
        # For each race in the schedule
        for event_idx, event in schedule.iterrows():
            try:
                # Only consider races (not sprint races, qualifying, etc.)
                race = fastf1.get_session(year, event['EventName'], 'R')
                race.load()
                print(f"Loaded {year} {event['EventName']}")
                
                # Get lap data - this contains pitstop information
                laps_data = race.laps
                
                # Get weather data if available
                try:
                    weather_data = race.weather_data
                    # Add weather averages to each lap
                    avg_temp = weather_data['AirTemp'].mean()
                    avg_humidity = weather_data['Humidity'].mean()
                    track_temp = weather_data['TrackTemp'].mean()
                    rainfall = (weather_data['Rainfall'].max() > 0)
                    
                    laps_data['AirTemp'] = avg_temp
                    laps_data['Humidity'] = avg_humidity
                    laps_data['TrackTemp'] = track_temp
                    laps_data['Rainfall'] = rainfall
                except Exception as e:
                    print(f"Weather data issue for {year} {event['EventName']}: {e}")
                    # If weather data not available, fill with NaN
                    laps_data['AirTemp'] = np.nan
                    laps_data['Humidity'] = np.nan
                    laps_data['TrackTemp'] = np.nan
                    laps_data['Rainfall'] = False
                
                # Add race info
                laps_data['Track'] = event['EventName']
                laps_data['Year'] = year
                
                # Get circuit info
                try:
                    circuit_info = race.get_circuit_info()
                    laps_data['CircuitLength'] = circuit_info['CIRCUITLENGTH'] if 'CIRCUITLENGTH' in circuit_info else np.nan
                except Exception as e:
                    print(f"Circuit info issue for {year} {event['EventName']}: {e}")
                    laps_data['CircuitLength'] = np.nan
                
                laps_data['TotalLaps'] = race.total_laps
                
                all_race_data.append(laps_data)
                
            except Exception as e:
                print(f"Error loading {year} {event['EventName']}: {e}")
    
    # Combine all race data
    if all_race_data:
        combined_data = pd.concat(all_race_data, ignore_index=True)
        return combined_data
    else:
        print("No race data collected.")
        return None

def preprocess_data(combined_data):
    """Preprocess the combined race data for modeling"""
    if combined_data is None or combined_data.empty:
        print("No data to preprocess.")
        return None
    
    # Extract pitstop information - a pitstop is indicated by PitInTime or PitOutTime not being NA
    pitstops = combined_data[combined_data['PitInTime'].notna() | combined_data['PitOutTime'].notna()].copy()
    
    if pitstops.empty:
        print("No pitstop data found.")
        return None
    
    # Group by Driver, Year, and Track to get pitstop patterns
    pitstop_summary = pitstops.groupby(['Year', 'Track', 'Driver']).agg({
        'LapNumber': ['count', list],  # Count gives number of pitstops, list gives the lap numbers
        'Compound': lambda x: list(x) if len(x) > 0 else None,  # Tire compounds used
        'TyreLife': 'max',  # Maximum tire life
        'AirTemp': 'mean',
        'TrackTemp': 'mean',
        'Humidity': 'mean',
        'Rainfall': 'max',
        'CircuitLength': 'first',
        'TotalLaps': 'first'
    })
    
    # Flatten the multi-index columns
    pitstop_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in pitstop_summary.columns.values]
    pitstop_summary.rename(columns={'LapNumber_count': 'NumPitStops', 'LapNumber_list': 'PitStopLaps', 'Compound_<lambda>': 'Compounds'}, inplace=True)
    
    # Reset index for easier manipulation
    pitstop_summary.reset_index(inplace=True)
    
    # Create features for modeling
    features_df = pitstop_summary.copy()
    
    # Add track characteristics (you might want to manually create a dictionary with track info)
    track_types = {
        'Bahrain': 'high-degradation',
        'Saudi': 'street',
        'Australia': 'mixed',
        'Emilia': 'traditional',
        'Miami': 'street',
        'Monaco': 'street',
        'Spain': 'high-downforce',
        'Austria': 'power',
        'British': 'mixed',
        'Hungarian': 'high-downforce',
        'Belgian': 'power',
        'Dutch': 'high-downforce',
        'Italian': 'power',
        'Singapore': 'street',
        'Japanese': 'high-downforce',
        'Qatar': 'mixed',
        'United States': 'mixed',
        'Mexico': 'high-altitude',
        'Brazilian': 'mixed',
        'Las Vegas': 'street',
        'Abu Dhabi': 'mixed',
        # Add more tracks as needed
    }
    
    # Apply track type mapping - look for partial matches in track names
    features_df['TrackType'] = features_df['Track'].apply(
        lambda x: next((v for k, v in track_types.items() if k.lower() in x.lower()), 'unknown')
    )
    
    # Handle any missing values
    for col in ['AirTemp_mean', 'TrackTemp_mean', 'Humidity_mean']:
        if col in features_df.columns:
            mean_value = features_df[col].mean()
            features_df[col].fillna(mean_value, inplace=True)
    
    return features_df

# Main execution
print("Starting data collection...")
combined_data = collect_race_data(years=[2021, 2022, 2023])

if combined_data is not None and not combined_data.empty:
    print(f"Collected data for {len(combined_data)} laps.")
    print("Processing data...")
    features_df = preprocess_data(combined_data)
    
    if features_df is not None and not features_df.empty:
        print(f"Processed data contains {len(features_df)} driver-race entries.")
        # Save the processed data
        features_df.to_csv('f1_pitstop_data.csv', index=False)
        print("Data saved to f1_pitstop_data.csv")
    else:
        print("Data processing failed.")
else:
    print("No data collected.")