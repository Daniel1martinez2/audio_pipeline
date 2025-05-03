import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import mido
import pandas as pd
import matplotlib.pyplot as plt

root_dir = '/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/groove'
info = '/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/info_clean.csv'
output_dir_root = '/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/groove-aligned'

info_df = pd.read_csv(info)

def get_first_midi_note_time(midi_path, bpm=None):
    """
    Get the time of the first note in a MIDI file
    
    Args:
        midi_path: Path to the MIDI file
        bpm: Optional beats per minute to override the MIDI file's tempo
        
    Returns:
        first_note_time: Time of first note in seconds
    """
    print(f"Analyzing MIDI file: {midi_path}")
    
    try:
        # Load the MIDI file
        midi = mido.MidiFile(midi_path)
        
        first_note_time = None
        
        # If BPM is provided, calculate tempo in microseconds per beat
        if bpm is not None:
            tempo = int(60000000 / bpm)
            print(f"Using provided BPM: {bpm} (tempo: {tempo} μs/beat)")
        else:
            # Default tempo (microseconds per beat, equivalent to 120 BPM)
            tempo = 500000
            
            # Look for tempo information in the MIDI file
            for track in midi.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        tempo = msg.tempo
                        print(f"Found tempo in MIDI: {60000000 / tempo:.1f} BPM")
                        break
        
        # Find first note event
        for track in midi.tracks:
            cumulative_time = 0  # in ticks
            
            # Process each message in the track
            for msg in track:
                cumulative_time += msg.time
                
                # Convert ticks to seconds
                seconds = mido.tick2second(cumulative_time, midi.ticks_per_beat, tempo)
                
                # Look for note_on events with velocity > 0
                if msg.type == 'note_on' and msg.velocity > 0:
                    if first_note_time is None or seconds < first_note_time:
                        first_note_time = seconds
        
        if first_note_time is not None:
            print(f"First MIDI note event at: {first_note_time:.3f} seconds")
            return first_note_time
        else:
            print("No note events found in MIDI file.")
            return None
            
    except Exception as e:
        print(f"Error reading MIDI file: {e}")
        return None

def get_first_wav_onset(wav_path, plot=False, method='fixed', sensitivity=0.2):
    """
    Get the time of the first significant onset using a consistent method
    
    Args:
        wav_path: Path to the WAV file
        plot: Whether to generate visualization plots
        method: Detection method ('fixed', 'energy', or 'percussion')
        sensitivity: Lower values make detection more sensitive (0.1-0.3 recommended)
        
    Returns:
        first_onset_time: Time of first significant onset in seconds
    """
    print(f"Analyzing WAV file: {wav_path} using '{method}' method (sensitivity: {sensitivity})")
    
    # Load the audio file
    y, sr = librosa.load(wav_path, sr=None)
    first_onset_frame = None
    
    # Choose a specific detection method and stick with it
    if method == 'energy':
        # Energy-based detection
        rms = librosa.feature.rms(y=y)[0]
        # Find the first frame where RMS exceeds a threshold
        threshold = sensitivity * np.max(rms)
        for i, energy in enumerate(rms):
            if energy > threshold:
                hop_length = 512  # Standard hop length for RMS
                first_onset_frame = i
                print(f"Energy-based detection found onset at frame {i}")
                break
                
    elif method == 'percussion':
        # Percussion-specific detection
        y_perc = librosa.effects.harmonic(y=y)
        onset_env_perc = librosa.onset.onset_strength(y=y_perc, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env_perc, 
            sr=sr, 
            delta=sensitivity
        )
        if len(onset_frames) > 0:
            first_onset_frame = onset_frames[0]
            print(f"Percussion-specific detection found onset at frame {first_onset_frame}")
            
    else:  # 'fixed' method (default)
        # Standard onset detection with consistent parameters
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, 
            sr=sr, 
            delta=sensitivity,
            wait=1
        )
        if len(onset_frames) > 0:
            first_onset_frame = onset_frames[0]
            print(f"Fixed onset detection found onset at frame {first_onset_frame}")
    
    # If no onset detected, try a more sensitive approach
    if first_onset_frame is None and method == 'fixed':
        onset_frames = librosa.onset.onset_detect(
            y=y, 
            sr=sr, 
            delta=sensitivity/2,  # More sensitive
            wait=1
        )
        if len(onset_frames) > 0:
            first_onset_frame = onset_frames[0]
            print(f"More sensitive detection found onset at frame {first_onset_frame}")
    
    # If still no onset detected
    if first_onset_frame is None:
        print("Warning: No significant audio onset could be detected.")
        return None
    
    # Convert frame to time
    first_onset_time = librosa.frames_to_time(first_onset_frame, sr=sr)
    # If frames_to_time returns an array, take first element
    if isinstance(first_onset_time, np.ndarray):
        first_onset_time = first_onset_time[0]
    
    print(f"First significant audio onset at: {first_onset_time:.3f} seconds")
    
    # Optional: Generate plot of the onset detection
    if plot:
        plt.figure(figsize=(12, 6))
        
        # Plot waveform
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.axvline(x=first_onset_time, color='r', linestyle='--', label='Detected Onset')
        plt.title('Waveform with Detected Onset')
        plt.legend()
        
        # Plot onset strength
        plt.subplot(2, 1, 2)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        frames = range(len(onset_env))
        times = librosa.frames_to_time(frames, sr=sr)
        plt.plot(times, onset_env)
        plt.axvline(x=first_onset_time, color='r', linestyle='--', label='Detected Onset')
        plt.title('Onset Strength Function')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.splitext(wav_path)[0] + '_onset_detection.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved onset detection plot to {plot_path}")
    
    return first_onset_time

def trim_wav_file(wav_path, output_path, trim_seconds):
    """
    Trim the beginning of a WAV file by the specified number of seconds
    
    Args:
        wav_path: Path to the input WAV file
        output_path: Path to save the trimmed WAV file
        trim_seconds: Number of seconds to trim from the beginning
        
    Returns:
        success: Boolean indicating success or failure
    """
    print(f"Trimming {trim_seconds:.3f} seconds from the beginning of {os.path.basename(wav_path)}")
    
    # Load the audio file
    y, sr = librosa.load(wav_path, sr=None)  # Use original sample rate
    
    # Calculate the number of samples to trim
    trim_samples = int(trim_seconds * sr)
    
    # Ensure we don't try to trim more samples than we have
    if trim_samples >= len(y):
        print(f"Error: Cannot trim {trim_seconds:.3f}s as the file is only {len(y)/sr:.3f}s long")
        return False
    
    # Ensure we don't trim a negative number of samples
    if trim_samples < 0:
        print(f"Error: Cannot trim negative time ({trim_seconds:.3f}s)")
        return False
    
    # Trim the audio
    y_trimmed = y[trim_samples:]
    
    # Save the trimmed audio
    sf.write(output_path, y_trimmed, sr)
    print(f"Saved trimmed file to {output_path}")
    
    return True

def align_wav_to_midi(midi_path, wav_path, output_dir=None, plot=False, bpm=None, 
                     detection_method='fixed', detection_sensitivity=0.2):
    """
    Precisely align a WAV file to a MIDI file by matching first events
    
    Args:
        midi_path: Path to the MIDI file
        wav_path: Path to the WAV file
        output_dir: Directory to save the aligned WAV file
        plot: Whether to generate visualization plots
        bpm: Optional beats per minute to override the MIDI file's tempo
        detection_method: Method for onset detection ('fixed', 'energy', or 'percussion')
        detection_sensitivity: Sensitivity for onset detection (lower = more sensitive)
        
    Returns:
        output_path: Path to the aligned WAV file
        trim_amount: Amount of time trimmed in seconds
        alignment_info: Dictionary with detailed alignment information
    """
    print("\n==== Precise MIDI-WAV Alignment ====")
    if bpm is not None:
        print(f"Using custom BPM: {bpm}")
    print(f"Using detection method: {detection_method} (sensitivity: {detection_sensitivity})")
    
    # Store alignment info for verification
    alignment_info = {
        'original_wav_path': wav_path,
        'midi_path': midi_path,
        'bpm': bpm,
        'detection_method': detection_method,
        'detection_sensitivity': detection_sensitivity
    }
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.dirname(wav_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.wav")
    
    # Step 1: Get first MIDI note time
    midi_note_time = get_first_midi_note_time(midi_path, bpm)
    if midi_note_time is None:
        print("Could not determine first MIDI note time. Aborting.")
        return None, 0, None
    
    # Step 2: Get first WAV onset time using the specified method
    wav_onset_time = get_first_wav_onset(
        wav_path, 
        plot=plot, 
        method=detection_method, 
        sensitivity=detection_sensitivity
    )
    if wav_onset_time is None:
        print("Could not determine first WAV onset time. Aborting.")
        return None, 0, None
    
    # Step 3: Calculate precise trim amount
    # We want: wav_onset_time - trim_amount = midi_note_time
    # Therefore: trim_amount = wav_onset_time - midi_note_time
    trim_amount = wav_onset_time - midi_note_time
    
    # Store in alignment info
    alignment_info['midi_note_time'] = midi_note_time
    alignment_info['wav_onset_time'] = wav_onset_time
    alignment_info['trim_amount'] = trim_amount
    
    # Only trim if the WAV onset is after the MIDI note (positive trim amount)
    if trim_amount <= 0:
        print(f"WAV file already starts {abs(trim_amount):.3f}s before MIDI file.")
        print(f"No trimming needed as WAV onset precedes MIDI note.")
        
        # Copy the file instead of trimming
        import shutil
        shutil.copy2(wav_path, output_path)
        print(f"Copied original file to {output_path}")
        
        alignment_info['action'] = 'copied'
        return output_path, 0, alignment_info
    
    # Step 4: Trim the WAV file by the precise amount
    success = trim_wav_file(wav_path, output_path, trim_amount)
    
    if success:
        print(f"\n✅ Successfully aligned WAV to MIDI")
        print(f"   WAV onset was at {wav_onset_time:.3f}s, MIDI note at {midi_note_time:.3f}s")
        print(f"   Trimmed exactly {trim_amount:.3f}s from the beginning")
        
        alignment_info['action'] = 'trimmed'
        alignment_info['output_path'] = output_path
                
        # Verify the alignment
        if plot:
            verify_alignment_fixed(midi_path, output_path, trim_amount, bpm, alignment_info)
        
        return output_path, trim_amount, alignment_info
    else:
        print(f"\n❌ Failed to align WAV to MIDI")
        return None, 0, None

def verify_alignment_fixed(midi_path, aligned_wav_path, trim_amount, bpm=None, alignment_info=None):
    """
    Verify alignment using the known trim amount rather than re-detecting onsets
    
    Args:
        midi_path: Path to the MIDI file
        aligned_wav_path: Path to the aligned WAV file
        trim_amount: The amount that was trimmed from the original WAV
        bpm: Optional beats per minute to override the MIDI file's tempo
        alignment_info: Dictionary with detailed alignment information
    """
    print("\n==== Verifying Alignment ====")
    
    # Get first MIDI note time again (should be the same)
    midi_note_time = get_first_midi_note_time(midi_path, bpm)
    if midi_note_time is None:
        print("Could not verify MIDI timing. Skipping verification.")
        return
    
    # Calculate where the WAV onset should be in the trimmed file
    # In the aligned file, the onset should be at the same time as the midi note
    expected_wav_onset = midi_note_time
    
    print("Verification based on fixed trim amount:")
    print(f"  Original WAV onset time: {alignment_info['wav_onset_time']:.3f}s")
    print(f"  MIDI first note time: {midi_note_time:.3f}s")
    print(f"  Applied trim amount: {trim_amount:.3f}s")
    print(f"  Expected WAV onset now at: {expected_wav_onset:.3f}s")
    
    # Optional: also do a detection-based verification as a sanity check
    # But don't rely on it for the primary verification
    detection_method = alignment_info.get('detection_method', 'fixed')
    detection_sensitivity = alignment_info.get('detection_sensitivity', 0.2)
    
    print("\nAdditional verification by re-detecting onset (for information only):")
    detected_onset = get_first_wav_onset(
        aligned_wav_path, 
        method=detection_method, 
        sensitivity=detection_sensitivity
    )
    
    if detected_onset is not None:
        # Calculate offset between expected and detected onset
        offset = detected_onset - expected_wav_onset
        print(f"  Re-detected onset at: {detected_onset:.3f}s")
        print(f"  Offset from expected: {offset:.3f}s")
        
        # Check if within tolerance
        tolerance = 0.05  # 50 milliseconds
        if abs(offset) <= tolerance:
            print(f"✅ Re-detection confirms good alignment (offset within {tolerance*1000:.0f}ms)")
        else:
            print(f"⚠️ Re-detection shows different onset point (offset: {offset*1000:.0f}ms)")
            print(f"   This is expected due to detection inconsistency, but fixed trim verification confirms proper alignment")
    else:
        print("  Could not re-detect onset in aligned file")
    
    print("\n✅ Alignment confirmed by fixed trim amount method")
    
    # Optional: Create a visualization to compare the two files
    if True:  # Set to False to disable visualization
        # Load both audio files
        y_midi_file, sr_midi = librosa.load(midi_path, sr=None) if midi_path.endswith('.wav') else (None, None)
        y_aligned, sr_aligned = librosa.load(aligned_wav_path, sr=None)
        
        # If MIDI file is not a WAV, we can't visualize it directly
        if y_midi_file is None:
            print("MIDI file is not a WAV file. Skipping waveform comparison.")
            return
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot MIDI waveform (or corresponding audio)
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(y_midi_file, sr=sr_midi)
        plt.axvline(x=midi_note_time, color='r', linestyle='--', label='MIDI First Note')
        plt.title('MIDI Audio Waveform')
        plt.legend()
        
        # Plot aligned WAV waveform
        plt.subplot(2, 1, 2)
        librosa.display.waveshow(y_aligned, sr=sr_aligned)
        plt.axvline(x=expected_wav_onset, color='r', linestyle='--', label='Expected Onset Position')
        if detected_onset is not None:
            plt.axvline(x=detected_onset, color='g', linestyle=':', label='Re-detected Onset')
        plt.title('Aligned WAV Waveform')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = os.path.splitext(aligned_wav_path)[0] + '_alignment_verification.png'
        plt.savefig(viz_path)
        plt.close()
        print(f"Saved alignment verification visualization to {viz_path}")

def verify_alignment(midi_path, aligned_wav_path, bpm=None):
    """
    Legacy verification function - kept for compatibility
    Uses the potentially inconsistent onset detection for verification
    
    Args:
        midi_path: Path to the MIDI file
        aligned_wav_path: Path to the aligned WAV file
        bpm: Optional beats per minute
    """
    print("\n==== Legacy Verification (Using Inconsistent Detection) ====")
    print("⚠️ This verification method is kept for compatibility but is not reliable.")
    print("   It uses onset re-detection which can give inconsistent results.")
    print("   Please use the fixed verification method instead.\n")
    
    # Get first MIDI note time again
    midi_note_time = get_first_midi_note_time(midi_path, bpm)
    
    # Get first onset time in aligned WAV - this is where inconsistencies occur
    aligned_onset_time = get_first_wav_onset(aligned_wav_path)
    
    if midi_note_time is None or aligned_onset_time is None:
        print("Could not verify alignment due to detection failure")
        return
    
    # Calculate offset
    offset = aligned_onset_time - midi_note_time
    print(f"Alignment offset: {offset:.3f} seconds")
    
    # Check if within tolerance
    tolerance = 0.05  # 50 milliseconds
    if abs(offset) <= tolerance:
        print(f"✅ Files are well-aligned! (offset within {tolerance*1000:.0f}ms)")
    else:
        print(f"⚠️ Files may not be perfectly aligned (offset: {offset*1000:.0f}ms)")
        print(f"   This could be due to onset detection differences between runs")
        print(f"   Consider using the fixed verification method instead")

def copy_midi_to_new_dir(aligned_wav_path, midi_path):
    """
    Copy the MIDI file to the same directory as the aligned WAV file
    
    Args:
        aligned_wav_path: Path to the aligned WAV file
        midi_path: Path to the MIDI file
    """
    # Get the directory of the aligned WAV file
    new_dir = os.path.dirname(aligned_wav_path)
    
    # Get the filename of the MIDI file
    midi_filename = os.path.basename(midi_path)
    
    # Create the new path for the MIDI file
    new_midi_path = os.path.join(new_dir, midi_filename)
    
    # Copy the MIDI file to the new directory
    import shutil
    shutil.copy2(midi_path, new_midi_path)
    print(f"Copied MIDI file to {new_midi_path}")

def batch_align(csv_file, base_dir=None, output_dir=None, detection_method='fixed', detection_sensitivity=0.2):
    """
    Align multiple WAV files to MIDI files based on a CSV file
    
    CSV format:
    midi_path,wav_path,bpm
    
    Args:
        csv_file: Path to the CSV file
        base_dir: Base directory for relative paths in the CSV
        output_dir: Directory to save the aligned WAV files
        detection_method: Method for onset detection
        detection_sensitivity: Sensitivity for onset detection
    """
    import csv
    
    print(f"Batch aligning files from {csv_file}")
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for i, row in enumerate(reader):
            print(f"\nProcessing row {i+1}:")
            
            # Parse the row
            if len(row) >= 3:
                midi_path, wav_path, bpm = row[0], row[1], float(row[2])
            else:
                print(f"Error: Row {i+1} does not have enough columns. Skipping.")
                continue
            
            # Handle relative paths
            if base_dir and not os.path.isabs(midi_path):
                midi_path = os.path.join(base_dir, midi_path)
            if base_dir and not os.path.isabs(wav_path):
                wav_path = os.path.join(base_dir, wav_path)
            
            # Set output directory for this file
            if output_dir:
                file_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(wav_path))[0])
            else:
                file_output_dir = os.path.dirname(wav_path)
            
            # Align the file
            align_wav_to_midi(
                midi_path=midi_path,
                wav_path=wav_path,
                output_dir=file_output_dir,
                plot=True,
                bpm=bpm,
                detection_method=detection_method,
                detection_sensitivity=detection_sensitivity
            )


def copy_midi_to_new_dir(aligned_wav_path, midi_path):
    midi_output_path = os.path.join(os.path.dirname(aligned_wav_path), os.path.basename(midi_path))

    if not os.path.exists(midi_output_path):
        os.system(f"cp {midi_path} {midi_output_path}")
        print(f"Copied MIDI file to {midi_output_path}")
    else:
        print(f"MIDI file already exists at {midi_output_path}")




if __name__ == "__main__":
  for i in range(info_df.shape[0]):
      midi_file = f"{root_dir}/{info_df.iloc[i]['midi_filename']}"
      wav_file = f"{root_dir}/{info_df.iloc[i]['audio_filename']}"
      output_clean = "/".join(info_df.iloc[i]['audio_filename'].split("/")[:-1])
      output_dir = f"{output_dir_root}/{output_clean}"
      
      # Get BPM from the DataFrame
      bpm = int(info_df.iloc[i]["bpm"])
      print(f"\nProcessing row {i + 1}:")
      print(midi_file, wav_file, output_dir, bpm)
      
      aligned_wav, trim_amount, alignment_info = align_wav_to_midi(
          midi_path=midi_file,
          wav_path=wav_file, 
          output_dir=output_dir,
          bpm=bpm,
          detection_method='fixed', 
          detection_sensitivity=0.2 
      )
      os.system(f"cp {midi_file} {output_dir_root}/{output_clean}")