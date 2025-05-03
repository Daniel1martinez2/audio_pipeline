import pandas as pd


fwod_path = "/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/fwod_database.csv"
audio_tensor_path = "/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/improved_audio_features.csv"
output_csv_path = "/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/combined_drum_database.csv"

def join_fwod_audio_databases(fwod_path, audio_tensor_path, output_path):
    """
    Joins FWOD and audio tensor databases based on common metadata fields.
    
    Args:
        fwod_path (str): Path to the FWOD database CSV
        audio_tensor_path (str): Path to the audio tensor database CSV
        output_path (str): Path where the joined CSV will be saved
    """
    # Load both databases
    fwod_database = pd.read_csv(fwod_path)
    audio_tensor = pd.read_csv(audio_tensor_path)
    
    # Clean tensor values (remove 'tensor(' and ')' if needed)
    for col in audio_tensor.columns:
        if col.startswith('low_') or col.startswith('mid_') or col.startswith('high_'):
            audio_tensor[col] = audio_tensor[col].astype(str).str.replace('tensor(', '').str.replace(')', '')
            audio_tensor[col] = pd.to_numeric(audio_tensor[col])
    
    # Define join keys
    join_keys = ['sequence', 'drummer', 'session', 'id', 'midi_filename', 'audio_filename']
    
    # Merge the two dataframes
    joined_df = pd.merge(fwod_database, audio_tensor, on=join_keys, how='inner', 
                        suffixes=('_fwod', '_audio'))
    
    # Handle duplicate columns (style, bpm, split might appear in both)
    # Keep only one copy of each
    for col in ['style', 'bpm', 'split']:
        if f'{col}_fwod' in joined_df.columns and f'{col}_audio' in joined_df.columns:
            joined_df[col] = joined_df[f'{col}_fwod']  # Keep the FWOD version
            joined_df = joined_df.drop([f'{col}_fwod', f'{col}_audio'], axis=1)
    
    # Save to CSV
    joined_df.to_csv(output_path, index=False)
    print(f"Joined database saved to {output_path} with {len(joined_df)} rows")
    
    return joined_df


# if main
if __name__ == "__main__":
    
    joined_df = join_fwod_audio_databases(
        fwod_path=fwod_path,
        audio_tensor_path=audio_tensor_path,
        output_path=output_csv_path,
    )

    print(f"Columns in joined database: {joined_df.columns.tolist()}")
    print(f"Sample row:\n{joined_df.iloc[0]}")