import pandas as pd
import os
import mido
import numpy as np

def midi_to_fwod(file_path):
    """
    Takes a MIDI file path as input and returns a list of 1x16 normalized FWOD vectors.
    
    FWOD (Flattened Weighted Onset Density) is a representation where:
    - Each drum hit is weighted based on its category (low, mid, high)
    - Weights are multiplied by velocity
    - The resulting values are normalized
    
    Args:
        file_path (str): Path to the MIDI file
        
    Returns:
        list: List of 1x16 numpy arrays, each representing a flattened pattern
    """
    # MIDI General MIDI drum mapping dictionary
    GM_dict = {
        # key is midi note number
        # values are:
        # [0] name (as string)
        # [1] name category low mid or high (as string)
        # [2] substiture midi number for simplified MIDI (all instruments)
        22: ["Closed Hi-Hat edge", "high", 42],
        26: ["Open Hi-Hat edge", "high", 46],
        35: ["Acoustic Bass Drum", "low", 36],
        36: ["Bass Drum 1", "low", 36],
        37: ["Side Stick", "mid", 37],
        38: ["Acoustic Snare", "mid", 38],
        39: ["Hand Clap", "mid", 39],
        40: ["Electric Snare", "mid", 38],
        41: ["Low Floor Tom", "low", 45],
        42: ["Closed Hi Hat", "high", 42],
        43: ["High Floor Tom", "mid", 45],
        44: ["Pedal Hi-Hat", "high", 46],
        45: ["Low Tom", "low", 45],
        46: ["Open Hi-Hat", "high", 46],
        47: ["Low-Mid Tom", "low", 47],
        48: ["Hi-Mid Tom", "mid", 47],
        49: ["Crash Cymbal 1", "high", 49],
        50: ["High Tom", "mid", 50],
        51: ["Ride Cymbal 1", "high", 51],
        52: ["Chinese Cymbal", "high", 52],
        53: ["Ride Bell", "high", 53],
        54: ["Tambourine", "high", 54],
        55: ["Splash Cymbal", "high", 55],
        56: ["Cowbell", "high", 56],
        57: ["Crash Cymbal 2", "high", 57],
        58: ["Vibraslap", "mid", 58],
        59: ["Ride Cymbal 2", "high", 59],
        60: ["Hi Bongo", "high", 60],
        61: ["Low Bongo", "mid", 61],
        62: ["Mute Hi Conga", "mid", 62],
        63: ["Open Hi Conga", "high", 63],
        64: ["Low Conga", "low", 64],
        65: ["High Timbale", "mid", 65],
        66: ["Low Timbale", "low", 66],
        67: ["High Agogo", "high", 67],
        68: ["Low Agogo", "mid", 68],
        69: ["Cabasa", "high", 69],
        70: ["Maracas", "high", 69],
        71: ["Short Whistle", "high", 71],
        72: ["Long Whistle", "high", 72],
        73: ["Short Guiro", "high", 73],
        74: ["Long Guiro", "high", 74],
        75: ["Claves", "high", 75],
        76: ["Hi Wood Block", "high", 76],
        77: ["Low Wood Block", "mid", 77],
        78: ["Mute Cuica", "high", 78],
        79: ["Open Cuica", "high", 79],
        80: ["Mute Triangle", "high", 80],
        81: ["Open Triangle", "high", 81],
    }
    
    # Lists of instruments by category
    lows = [35, 36, 41, 45, 47, 64, 66]
    mids = [37, 38, 39, 40, 43, 48, 50, 61, 62, 65, 68, 77]
    highs = [22, 26, 42, 44, 46, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 63, 67, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81]
    
    # Extract patterns from MIDI file
    def midifile2hv_list(file_name):
        '''
        Get a MIDI file and convert it to a list of hv_lists (note numbers and velocity)
        Each hv_list represents a 16-step pattern
        '''
        pattern = []
        mid = mido.MidiFile(file_name)  # create a mido file instance
        sixteenth = mid.ticks_per_beat / 4  # find the length of a sixteenth note
        
        acc = 0  # use this to keep track of time
        
        for i, track in enumerate(mid.tracks):
            for msg in track:  # process all messages
                acc += msg.time  # accumulate time of any message type
                if msg.type == "note_on" and msg.velocity != 0:  # skip velocity 0 format of note off
                    if msg.note in list(GM_dict.keys()):
                        midinote = GM_dict[msg.note][2]  # remap msg.note
                        rounded_step = int((acc / sixteenth) + 0.45)
                        midivelocity = msg.velocity / 127  # normalize upfront
                        pattern.append((int(acc / sixteenth), midinote, midivelocity))  # step, note, velocity
        
        if len(pattern) > 0:  # just proceed if analyzed pattern has at least one onset
            # round the pattern to the next multiple of 16
            rounded_step = max([p[0] for p in pattern]) if pattern else 0
            if (rounded_step / 16) - (rounded_step // 16) != 0:
                pattern_len_in_steps = (rounded_step // 16) * 16 + 16
            else:
                pattern_len_in_steps = (rounded_step // 16) * 16
            
            # create an empty list of lists the size of the pattern
            output_pattern = [[] for _ in range(pattern_len_in_steps)]
            
            # group the instruments and their velocity that played at a specific step
            for step, note, velocity in pattern:
                if step < pattern_len_in_steps:
                    output_pattern[step].append((note, velocity))
                    # make sure no notes are repeated and events are sorted
                    output_pattern[step] = list(set(output_pattern[step]))
                    output_pattern[step].sort()
            
            # split the pattern every 16 steps
            hv_lists_split = []
            for x in range(len(output_pattern) // 16):
                patt_fragment = output_pattern[x * 16 : (x * 16) + 16]
                patt_density = sum([1 for x in patt_fragment if x != []])
                
                # filter out patterns that have less than 4 events with notes
                if patt_density > 4:
                    hv_lists_split.append(patt_fragment)
            
            return hv_lists_split
        return []
    
    # Flatten hv_list to FWOD representation
    def flatten_hv_list(hv_list):
        '''
        Input an hv_list and output a flattened representation as a vector
        '''
        flat = np.zeros([len(hv_list), 1])
        
        # multiply velocities and categories
        for i, step in enumerate(hv_list):
            step_weight = 0
            for onset in step:
                if onset[0] in lows:
                    step_weight += onset[1] * 3  # Low sounds weighted by 3
                elif onset[0] in mids:
                    step_weight += onset[1] * 2  # Mid sounds weighted by 2
                else:
                    step_weight += onset[1] * 1  # High sounds weighted by 1
            flat[i] = step_weight
        
        # Normalize
        max_val = max(flat) if max(flat) > 0 else 1
        flat = flat / max_val
        
        return flat
    
    # Process the MIDI file
    try:
        # Get list of 16-step patterns
        hv_lists = midifile2hv_list(file_path)
        
        # Convert each pattern to FWOD representation
        fwod_list = []
        for pattern in hv_lists:
            fwod = flatten_hv_list(pattern)
            fwod_list.append(fwod)
        
        return fwod_list
    
    except Exception as e:
        print(f"Error processing MIDI file: {e}")
        return []

info_csv_path = '/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/info_clean.csv'
info_clean = pd.read_csv(info_csv_path)
root_dir = '/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/groove-aligned'
output_csv_path = '/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/fwod_database.csv'



def create_fwod_database(info_clean_df, root_dir, output_csv_path, midi_to_fwod_func):
    """
    Creates a database CSV with FWOD vectors and metadata for each bar of each MIDI file.
    
    Args:
        info_clean_df (pd.DataFrame): DataFrame containing metadata for MIDI files
                                      Must include columns: 'drummer', 'session', 'id', 'style', 
                                      'midi_filename', 'audio_filename', 'bpm', 'split'
        root_dir (str): Root directory where MIDI files are stored
        output_csv_path (str): Path where the output CSV will be saved
        midi_to_fwod_func (function): Function that converts MIDI file to FWOD vectors
        
    Returns:
        pd.DataFrame: The created DataFrame that was saved to CSV
    """
    # Create an empty list to store all rows
    all_rows = []
    
    # Process each MIDI file in the info_clean_df
    total_files = len(info_clean_df)
    
    print(f"Processing {total_files} MIDI files...")
    
    for i in range(total_files):
        # Get metadata for the current MIDI file
        row_data = info_clean_df.iloc[i]
        
        # Construct the full path to the MIDI file
        midi_path = os.path.join(root_dir, row_data['midi_filename'])
        
        try:
            # Convert MIDI to FWOD vectors
            fwod_vectors = midi_to_fwod_func(midi_path)
            
            # If we got vectors, create a row for each vector (bar)
            if fwod_vectors and len(fwod_vectors) > 0:
                for bar_number, fwod_vector in enumerate(fwod_vectors):
                    # Create a dictionary for this row
                    row_dict = {
                        'sequence': bar_number,
                        'drummer': row_data['drummer'],
                        'session': row_data['session'],
                        'id': row_data['id'],
                        'style': row_data['style'],
                        'midi_filename': row_data['midi_filename'],
                        'audio_filename': row_data['audio_filename'],
                        'bpm': row_data['bpm'],
                        'split': row_data['split']
                    }
                    
                    # Add the 16 FWOD values (y_0 to y_15)
                    for step in range(16):
                        row_dict[f'y_{step}'] = fwod_vector[step][0]
                    
                    # Add this row to our collection
                    all_rows.append(row_dict)
                
                if (i + 1) % 10 == 0 or (i + 1) == total_files:
                    print(f"Processed {i+1}/{total_files} files, extracted {len(all_rows)} bars so far...")
                
            else:
                print(f"Warning: No FWOD vectors extracted from {midi_path}")
                
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
    
    # Create DataFrame from all rows
    fwod_df = pd.DataFrame(all_rows)
    
    # Save to CSV
    fwod_df.to_csv(output_csv_path, index=False)
    print(f"Database saved to {output_csv_path} with {len(fwod_df)} total bars")
    
    return fwod_df



if __name__ == "__main__":
    fwod_database = create_fwod_database(
        info_clean_df=info_clean, 
        root_dir=root_dir,
        output_csv_path=output_csv_path,
        midi_to_fwod_func=midi_to_fwod
    )