import numpy as np
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup
from pybaseball import pitching_stats
from pybaseball import statcast, playerid_reverse_lookup
import pandas as pd

def pitcher_pitch_averages(start_date: str, end_date: str, pitcher_id: int) -> pd.DataFrame:
    """
    Get average stats for each pitch type a pitcher throws in a date range.
    
    :param start_date: 'YYYY-MM-DD'
    :param end_date: 'YYYY-MM-DD'
    :param pitcher_id: MLBAM player ID
    :return: DataFrame with averages by pitch type
    """
    # Pull pitch-by-pitch data
    data = statcast_pitcher(start_date, end_date, pitcher_id)
    
    if data.empty:
        print("No data found for given inputs.")
        return pd.DataFrame()
    
    # Group by pitch type and calculate averages
    grouped = data.groupby("pitch_type").agg({
        "release_speed": "mean",
        "release_spin_rate": "mean",
        "release_pos_x": "mean",
        "release_pos_z": "mean",
        "release_extension": "mean",
        "pfx_x": "mean",
        "pfx_z": "mean",
        "plate_x": "mean",
        "plate_z": "mean"
    })
    
    # Rename columns to make it clear they're averages
    grouped = grouped.rename(columns={
        "release_speed": "avg_velocity",
        "release_spin_rate": "avg_spin_rate",
        "release_pos_x": "avg_release_x",
        "release_pos_z": "avg_release_z",
        "release_extension": "avg_extension",
        "pfx_x": "avg_break_x",
        "pfx_z": "avg_break_z"
    })
    
    return grouped.reset_index()


def get_candidates(start_date: str, end_date: str, pitch_type: str) -> list:
    """
    Return a list of MLBAM pitcher IDs who threw a specific pitch type
    in the given date range, using pybaseball Statcast API.

    Pulls all pitch-level data in one call and filters by pitch type.

    :param start_date: 'YYYY-MM-DD'
    :param end_date: 'YYYY-MM-DD'
    :param pitch_type: pitch type string, e.g., 'FF', 'SL', 'CU'
    :return: list of pitcher IDs
    """
    print(f"Fetching Statcast data from {start_date} to {end_date}...")
    
    # Pull all pitch-level data for the date range
    try:
        data = statcast(start_date, end_date)
    except Exception as e:
        print(f"Error fetching Statcast data: {e}")
        return []

    if data.empty:
        print("No pitch data found for this date range.")
        return []

    if 'pitch_type' not in data.columns or 'pitcher' not in data.columns:
        print("Required columns ('pitch_type', 'pitcher') missing in data.")
        return []

    # Filter for the requested pitch type
    filtered = data[data['pitch_type'] == pitch_type]

    if filtered.empty:
        print(f"No pitchers threw pitch type '{pitch_type}' in this range.")
        return []

    # Return unique pitcher IDs
    candidates = filtered['pitcher'].unique().tolist()
    print(f"Found {len(candidates)} pitchers who threw pitch {pitch_type}")

    return candidates


def find_similar(start_date, end_date, pitcher_id, pitch_type, top_n=5):
    """
    Find pitchers with similar pitch characteristics for a given pitch type using Euclidean distance.
    Single API call for all pitchers in the date range, returns names along with IDs.
    
    :param start_date: 'YYYY-MM-DD'
    :param end_date: 'YYYY-MM-DD'
    :param pitcher_id: target pitcher's MLBAM ID
    :param pitch_type: pitch type string, e.g., 'FF'
    :param top_n: number of closest matches to return
    :return: DataFrame with closest pitchers, distances, and names
    """
    print(f"Fetching all Statcast pitch data from {start_date} to {end_date}...")
    data = statcast(start_date, end_date)

    if data.empty or 'pitch_type' not in data.columns or 'pitcher' not in data.columns:
        print("No valid pitch data found for this date range.")
        return pd.DataFrame()

    # Filter for the requested pitch type
    filtered = data[data['pitch_type'] == pitch_type]

    if filtered.empty:
        print(f"No pitchers threw pitch type '{pitch_type}' in this range.")
        return pd.DataFrame()

    # Compute average stats per pitcher
    features = ['release_speed', 'release_spin_rate', 'release_pos_x', 
                'release_pos_z', 'pfx_x', 'pfx_z']
    pitcher_averages = filtered.groupby('pitcher')[features].mean().reset_index()

    # Get target pitcher's feature vector
    target_row = pitcher_averages[pitcher_averages['pitcher'] == pitcher_id]
    if target_row.empty:
        print("Target pitcher did not throw this pitch type in the range.")
        return pd.DataFrame()
    target_vector = target_row[features].values.flatten()

    # Compute Euclidean distance for all other pitchers
    pitcher_averages = pitcher_averages[pitcher_averages['pitcher'] != pitcher_id]
    pitcher_averages['distance'] = pitcher_averages[features].apply(
        lambda row: np.linalg.norm(target_vector - row.values), axis=1
    )

    # Lookup names for all pitcher IDs at once
    pitcher_ids = pitcher_averages['pitcher'].tolist()
    id_name_df = playerid_reverse_lookup(pitcher_ids, key_type='mlbam')
    id_name_map = {row['key_mlbam']: f"{row['name_first']} {row['name_last']}" 
               for _, row in id_name_df.iterrows()}

    # Map names to DataFrame
    pitcher_averages['name'] = pitcher_averages['pitcher'].map(id_name_map)

    # Return top N closest pitchers with names
    return pitcher_averages[['pitcher', 'name', 'distance']].sort_values('distance').head(top_n).reset_index(drop=True)


def main():
    # data = pd.read_csv('all_pitches_2024.csv')  # contains pitcher, pitch_type
    # ff_pitchers = data[data['pitch_type'] == 'FF']['pitcher'].unique()

    # Gerrit Cole's MLB ID is 543037 (you can look up player IDs)
    # Suppose you have a list of candidate pitcher IDs
    similar = find_similar("2024-04-01", "2024-07-07", 543037, "FF", top_n=5)
    print(similar)
    
    #similar = find_similar("2024-04-01", "2024-07-07", 543037, "FF", candidate_ids)
    #print(similar)

    # Print out some basic info
    #print("Number of pitches:", len(data))
    #print("Columns available:", list(data.columns)[:15], "...")
    
    # Show the first few rows
    #print(data['pitch_type'].head())

    # Optionally save to CSV
    #data.to_csv("cole_statcast_sample.csv", index=False)

if __name__ == "__main__":
    main()
