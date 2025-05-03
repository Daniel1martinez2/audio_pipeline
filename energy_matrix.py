import pandas as pd
import numpy as np
import torch
import librosa
import os
import matplotlib.pyplot as plt


root_dir = '/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/groove-aligned'
info_csv_path = '/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/info_clean.csv'
info_clean = pd.read_csv(info_csv_path)
energy_matrix_df = info_clean[['drummer', 'session', 'id', 'style', 'midi_filename', 'audio_filename', 'bpm', 'split']]
output_csv_path = '/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/improved_audio_features.csv'



import numpy as np
import torch
import librosa
import os
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm

# Define improved frequency bands for better instrument separation
FREQUENCY_BANDS = {
    'kick': (40, 120),      # Bombo fundamental
    'low_mid': (120, 300),  # Bombo alto y toms bajos
    'snare': (300, 800),    # Caja fundamental
    'high_mid': (800, 2500), # Caja alta y toms
    'high': (2500, 8000),   # Hi-hats y platillos
    'very_high': (8000, 16000) # Brillo y armónicos superiores
}

# Simplified bands for final matrix (mapping from detailed bands to simplified)
SIMPLIFIED_BANDS = {
    'low': ['kick', 'low_mid'],
    'mid': ['snare', 'high_mid'],
    'high': ['high', 'very_high']
}

def extract_improved_onsets(audio_path, sr=44100, hop_length=512, 
                           onset_threshold=0.4, onset_backtrack=True):
    """
    Extract onset information with improved detection
    
    Parameters:
    audio_path (str): Path to audio file
    sr (int): Sample rate (higher for better high-frequency resolution)
    hop_length (int): Hop length for onset detection
    onset_threshold (float): Threshold for onset detection sensitivity
    onset_backtrack (bool): Whether to backtrack onsets to precise timing
    
    Returns:
    dict: Dictionary containing onset information
    """
    # Load audio with higher sample rate for better frequency resolution
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Calculate standard onset strength - FIX: use standard method instead of multi-band
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length,
        feature=librosa.feature.melspectrogram,
        aggregate=np.mean
    )
    
    # Adaptive threshold based on signal statistics
    # This helps with varying dynamic levels across different recordings
    threshold = onset_threshold * np.std(onset_env) + 0.5 * np.mean(onset_env)
    
    # Find peaks with variable properties
    peaks, properties = find_peaks(
        onset_env, 
        height=threshold,
        distance=hop_length // (sr // 200),  # Min ~5ms between onsets
        prominence=0.1 * np.max(onset_env)   # Ensure peaks are significant
    )
    
    # Get peak heights (intensity)
    peak_heights = properties['peak_heights']
    
    # Convert peaks to frames
    onset_frames = peaks
    
    # Backtrack to precise onset positions if requested
    if onset_backtrack:
        precise_frames = librosa.onset.onset_backtrack(onset_frames, onset_env)
    else:
        precise_frames = onset_frames
    
    # Convert to time
    onset_times = librosa.frames_to_time(precise_frames, sr=sr, hop_length=hop_length)
    
    return {
        'y': y, 
        'sr': sr,
        'onset_env': onset_env,
        'onset_frames': precise_frames,
        'onset_times': onset_times,
        'onset_strengths': peak_heights if len(peak_heights) == len(precise_frames) else None
    }

def get_spectral_features(y, sr, onset_frames, hop_length=512):
    """
    Extract detailed spectral features at each onset
    
    Parameters:
    y (np.ndarray): Audio signal
    sr (int): Sample rate
    onset_frames (np.ndarray): Frame indices of onsets
    hop_length (int): Hop length
    
    Returns:
    dict: Dictionary with spectral features
    """
    # Compute STFT with higher frequency resolution
    n_fft = 4096  # Larger FFT size for better frequency resolution
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    
    # Get frequency mapping
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Create frequency band indices
    band_indices = {}
    for band_name, (low_freq, high_freq) in FREQUENCY_BANDS.items():
        band_indices[band_name] = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
    
    # Initialize feature containers
    band_energies = np.zeros((len(onset_frames), len(FREQUENCY_BANDS)))
    spectral_features = {}
    
    # Calculate features for each onset
    for i, frame in enumerate(onset_frames):
        if frame >= D.shape[1]:
            # Skip if frame is out of bounds
            continue
            
        # Get spectrum at this frame and neighboring frames
        frame_window = np.arange(max(0, frame-1), min(D.shape[1], frame+3))
        spec_window = D[:, frame_window]
        
        # Use max energy in the window to capture attack transients
        spec = np.max(spec_window, axis=1)
        
        # Calculate energy in each frequency band
        for j, (band_name, indices) in enumerate(band_indices.items()):
            if len(indices) > 0:
                band_energies[i, j] = np.sum(spec[indices])
    
    # Calculate additional features
    spectral_features['band_energies'] = band_energies
    
    # Calculate spectral centroid (brightness) at each onset
    centroids = []
    for frame in onset_frames:
        if frame < D.shape[1]:
            centroid = librosa.feature.spectral_centroid(
                S=D[:, frame:frame+1], sr=sr, n_fft=n_fft
            )
            centroids.append(centroid.item())
        else:
            centroids.append(0)
    spectral_features['spectral_centroid'] = np.array(centroids)
    
    return spectral_features

def improved_quantize_onsets(onset_times, tempo, bars=1, grid_strength=0.7):
    """
    Quantize onsets to 16th note grid with adjustable strength
    
    Parameters:
    onset_times (np.ndarray): Times of onsets in seconds
    tempo (float): Tempo in BPM
    bars (int): Number of bars to quantize
    grid_strength (float): 0.0 = no quantization, 1.0 = full quantization
    
    Returns:
    dict: Dictionary with quantized information
    """
    # Make sure tempo is a scalar
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo)
    
    # Duration of a 16th note in seconds
    sixteenth_duration = 60.0 / (tempo * 4)
    
    # Duration of a bar in seconds
    bar_duration = 16 * sixteenth_duration
    
    # Initialize arrays
    quantized_steps = []
    quantized_times = []
    quantization_offsets = []  # How far from exact grid
    bar_numbers = []
    
    for onset_time in onset_times:
        # Determine which bar this onset belongs to
        bar_number = int(onset_time / bar_duration)
        
        # If we've exceeded our desired number of bars, stop
        if bars > 0 and bar_number >= bars:
            break
            
        # Time since the beginning of this bar
        time_in_bar = onset_time - (bar_number * bar_duration)
        
        # Calculate nearest 16th note (0-15) within the bar
        # and the exact time of that 16th note
        exact_step = time_in_bar / sixteenth_duration
        nearest_step = int(np.round(exact_step))
        nearest_step_time = nearest_step * sixteenth_duration + (bar_number * bar_duration)
        
        # Handle rounding to exactly 16 (which should be 0 of next bar)
        if nearest_step == 16:
            nearest_step = 0
            bar_number += 1
        
        # Only add steps that fit within our bars
        if nearest_step < 16:
            # Calculate the true step with partial quantization
            # Interpolate between exact and quantized position
            quantized_time = (1 - grid_strength) * onset_time + grid_strength * nearest_step_time
            
            # Determine the effective step and offset after partial quantization
            effective_step = nearest_step
            offset = onset_time - nearest_step_time
            
            # Store information
            quantized_steps.append(effective_step % 16 + (bar_number * 16))
            quantized_times.append(quantized_time)
            quantization_offsets.append(offset)
            bar_numbers.append(bar_number)
    
    return {
        'steps': np.array(quantized_steps) % 16,  # Steps within a bar (0-15)
        'absolute_steps': np.array(quantized_steps),  # Steps across all bars
        'times': np.array(quantized_times),  # Partially quantized times
        'offsets': np.array(quantization_offsets),  # Timing offsets
        'bar_numbers': np.array(bar_numbers)  # Bar numbers
    }

def extract_rhythmic_matrix(audio_path, tempo=None, bars_to_extract=1, grid_strength=0.7):
    """
    Extract improved rhythmic matrix from audio
    
    Parameters:
    audio_path (str): Path to audio file
    tempo (float, optional): Tempo in BPM. If None, will be detected
    bars_to_extract (int): Number of bars to extract (0 = all detected)
    grid_strength (float): 0.0 = no quantization, 1.0 = full quantization
    
    Returns:
    list: List of dictionaries with matrix and metadata for each bar
    """
    try:
        # Extract onset information with improved detection
        onset_data = extract_improved_onsets(audio_path)
        y, sr = onset_data['y'], onset_data['sr']
        onset_frames, onset_times = onset_data['onset_frames'], onset_data['onset_times']
        
        if len(onset_frames) == 0:
            print(f"No onsets detected in {audio_path}")
            return []
        
        # Extract spectral features
        spectral_data = get_spectral_features(y, sr, onset_frames)
        band_energies = spectral_data['band_energies']
        
        # Detect tempo if not provided
        if tempo is None:
            # More accurate tempo detection using beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
            print(f"Detected tempo: {tempo:.1f} BPM")
        else:
            tempo = float(tempo)
        
        # Quantize onsets to 16th note grid with adjustable strength
        quantized_data = improved_quantize_onsets(
            onset_times, tempo, bars=bars_to_extract, grid_strength=grid_strength
        )
        steps = quantized_data['steps']
        bar_numbers = quantized_data['bar_numbers']
        
        # Calculate bar information
        if bars_to_extract <= 0:
            # If extracting all bars, determine number of bars from audio
            audio_duration = librosa.get_duration(y=y, sr=sr)
            bar_duration = 60.0 / tempo * 4  # Duration of a bar in seconds
            num_bars = int(np.ceil(audio_duration / bar_duration))
        else:
            num_bars = bars_to_extract
            
        # Prepare to collect matrices for each bar
        bar_matrices = []
        
        # Process each bar
        for bar in range(num_bars):
            # Create matrix for current bar (6 detailed bands x 16 steps)
            detailed_matrix = np.zeros((len(FREQUENCY_BANDS), 16))
            
            # Get indices of onsets in current bar
            bar_indices = np.where(bar_numbers == bar)[0]
            
            if len(bar_indices) == 0:
                # Skip empty bars if extracting all
                if bars_to_extract <= 0:
                    continue
            
            # Fill matrix with band energies
            for i in bar_indices:
                if i < len(steps) and i < len(band_energies):
                    step = steps[i]
                    # Only fill if step is within range
                    if 0 <= step < 16:
                        for j, band_name in enumerate(FREQUENCY_BANDS.keys()):
                            detailed_matrix[j, step] += band_energies[i, j]
            
            # Normalize the detailed matrix
            # Use a more robust normalization that preserves relative intensities
            matrix_max = np.max(detailed_matrix)
            if matrix_max > 0:
                # Global normalization with compression for extreme values
                detailed_matrix = np.tanh(detailed_matrix / (matrix_max * 0.7))
            
            # Create the simplified 3-band matrix
            simplified_matrix = np.zeros((3, 16))
            for i, (simplified_band, detailed_bands) in enumerate(SIMPLIFIED_BANDS.items()):
                for band in detailed_bands:
                    band_idx = list(FREQUENCY_BANDS.keys()).index(band)
                    simplified_matrix[i] += detailed_matrix[band_idx]
                
                # Normalize each simplified band to have max=1 if it has any non-zero values
                band_max = np.max(simplified_matrix[i])
                if band_max > 0:
                    simplified_matrix[i] /= band_max
            
            # Convert to tensor
            tensor_matrix = torch.tensor(simplified_matrix, dtype=torch.float32)
            
            # Store matrix with metadata
            bar_matrices.append({
                'matrix': tensor_matrix,
                'bar': bar,
                'tempo': tempo
            })
        
        return bar_matrices
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def create_audio_tensor_dataframe(info_df, file_index, root_dir, y_target_included=False):
    """
    Creates a DataFrame from audio file with metadata from info_df.
    
    Args:
        info_df (pd.DataFrame): DataFrame with file metadata
        file_index (int): Index in the info_df to get metadata
        root_dir (str): Root directory for audio files
        y_target_included (bool): Whether y_targets are included in the info dataframe
    
    Returns:
        pd.DataFrame: DataFrame with metadata and audio features
    """
    # Get metadata for the current file
    file_metadata = info_df.iloc[file_index]
    
    # Form the full path to the audio file
    audio_filename = file_metadata['audio_filename']
    audio_path = os.path.join(root_dir, audio_filename)
    
    # Get tempo
    tempo = file_metadata['bpm']
    
    # Extract matrices from audio
    try:
        # Process all bars in the audio
        matrices = extract_rhythmic_matrix(audio_path, tempo=tempo, bars_to_extract=0, grid_strength=0.8)
        
        # Create list to hold all rows
        rows = []
        
        # Process each bar (matrix)
        for bar_data in matrices:
            matrix = bar_data['matrix']
            bar_num = bar_data['bar']
            
            # Create a row dictionary with metadata
            row = {
                'sequence': bar_num,
                'drummer': file_metadata['drummer'],
                'session': file_metadata['session'],
                'id': file_metadata['id'],
                'style': file_metadata['style'],
                'midi_filename': file_metadata['midi_filename'],
                'audio_filename': file_metadata['audio_filename'],
                'bpm': file_metadata['bpm'],
                'split': file_metadata['split']
            }
            
            # Add matrix values (low, mid, high for each step)
            for step in range(16):
                row[f'low_{step}'] = float(matrix[0, step])
                row[f'mid_{step}'] = float(matrix[1, step])
                row[f'high_{step}'] = float(matrix[2, step])
            
            # Add y-targets if they're part of the info dataframe
            if y_target_included:
                for step in range(16):
                    target_col = f'y_{step}'
                    if target_col in file_metadata:
                        row[target_col] = file_metadata[target_col]
            
            # Add to rows list
            rows.append(row)
        
        # Create DataFrame from rows
        return pd.DataFrame(rows)
    
    except Exception as e:
        print(f"Error processing {audio_filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()  # Return empty dataframe on error

def process_all_files(info_csv_path, root_dir, output_csv_path, y_target_included=False):
    """
    Process all audio files and create a combined dataframe
    
    Args:
        info_csv_path (str): Path to info CSV file
        root_dir (str): Root directory containing audio files
        output_csv_path (str): Path to save output CSV
        y_target_included (bool): Whether target y values are in the info CSV
    """
    # Load info dataframe
    info_df = pd.read_csv(info_csv_path)
    
    # Process all files and create a combined dataframe
    all_audio_df = pd.DataFrame()
    
    # Use tqdm for progress tracking
    for i in tqdm(range(len(info_df)), desc="Processing audio files"):
        # Create dataframe for this file
        file_df = create_audio_tensor_dataframe(info_df, i, root_dir, y_target_included)
        
        # Append to the combined dataframe
        all_audio_df = pd.concat([all_audio_df, file_df], ignore_index=True)
    
    # Save the result
    all_audio_df.to_csv(output_csv_path, index=False)
    print(f"Created database with {len(all_audio_df)} rows, saved to {output_csv_path}")

# Example usage
if __name__ == "__main__":
    process_all_files(info_csv_path, root_dir, output_csv_path)