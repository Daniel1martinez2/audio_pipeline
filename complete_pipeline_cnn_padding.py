#!/usr/bin/env python
"""
Complete Audio Pipeline with Variable-Length Approach
- No BPM normalization (preserves natural tempo)
- Variable-length mel spectrograms stored without padding
- Padding done during CNN batching for efficiency
- FWOD vector computation from MIDI

This pipeline stores original variable-length spectrograms,
optimizing storage and preserving musical characteristics.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import os
import sys
import time
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from datetime import datetime
from functools import lru_cache
import mido

# Set environment variable to suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore:The behavior of resample:FutureWarning'

# Import librosa for mel spectrogram calculation
import librosa

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_SR = 44100
N_FFT = 1024
N_MELS = 128
F_MIN, F_MAX = 20, 20000

# Frame configuration (no pre-padding)
MIN_FRAMES_PER_BAR = 8   # Minimum frames per bar (for very fast songs)

# GM map for FWOD computation
_GM_MAP = {  # Abridged version
    35: ("low"), 36: ("low"), 41: ("low"), 45: ("low"), 47: ("low"), 64: ("low"), 66: ("low"),
    37: ("mid"), 38: ("mid"), 39: ("mid"), 40: ("mid"), 43: ("mid"), 48: ("mid"), 50: ("mid"),
    61: ("mid"), 62: ("mid"), 65: ("mid"), 68: ("mid"), 77: ("mid"),
    22: ("high"), 26: ("high"), 42: ("high"), 44: ("high"), 46: ("high"), 49: ("high"),
    51: ("high"), 52: ("high"), 53: ("high"), 54: ("high"), 55: ("high"), 56: ("high"),
    57: ("high"), 58: ("high"), 59: ("high"), 60: ("high"), 63: ("high"), 67: ("high"),
    69: ("high"), 70: ("high"), 71: ("high"), 72: ("high"), 73: ("high"), 74: ("high"),
    75: ("high"), 76: ("high"), 78: ("high"), 79: ("high"), 80: ("high"), 81: ("high"),
}

_WEIGHTS = {"low": 3.0, "mid": 2.0, "high": 1.0}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Example:
    """A single training example containing mel spectrogram and FWOD vector."""
    mel: np.ndarray      # shape (N_MELS, variable_frames) - original length
    fwod: np.ndarray     # shape (16,) - always 16 steps per bar
    meta: Dict[str, Any]

class Stats:
    """Tracking statistics for the pipeline."""
    def __init__(self):
        self.start_time = time.time()
        self.files_processed = 0
        self.files_successful = 0
        self.total_files = 0
        self.total_examples = 0
        
    def update(self, success, examples_added=0):
        self.files_processed += 1
        if success:
            self.files_successful += 1
            self.total_examples += examples_added
            
    def print_progress(self):
        if self.total_files == 0:
            return
            
        elapsed = time.time() - self.start_time
        percent = (self.files_processed / self.total_files) * 100
        
        if self.files_processed > 0:
            remaining = (elapsed / self.files_processed) * (self.total_files - self.files_processed)
        else:
            remaining = 0
            
        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
              f"Progress: {percent:.1f}% ({self.files_processed}/{self.total_files}, "
              f"{self.files_successful} successful, {self.total_examples} examples) - "
              f"ETA: {remaining/60:.1f}min", end="")
        sys.stdout.flush()

# Initialize global stats
stats = Stats()

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

def log(msg, level=0, update_progress=False):
    """Log message with timestamp and indentation."""
    if update_progress:
        stats.print_progress()
        return
        
    prefix = "  " * level
    if level == 0:
        prefix = "► "
    elif level == 1:
        prefix = "  ├─ "
        
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {prefix}{msg}")
    sys.stdout.flush()

# ============================================================================
# AUDIO PROCESSING FUNCTIONS
# ============================================================================

def stereo_to_mono(y):
    """Convert stereo audio to mono by averaging channels."""
    if y.ndim > 1 and y.shape[1] > 1:
        log(f"Converting stereo to mono", 1)
        return np.mean(y, axis=1)
    return y

def fast_resample(y, orig_sr, target_sr):
    """Fast resampling using scipy.signal."""
    if orig_sr == target_sr:
        return y
        
    ratio = target_sr / orig_sr
    output_samples = int(len(y) * ratio)
    
    log(f"Resampling from {orig_sr}Hz to {target_sr}Hz (ratio: {ratio:.3f})", 1)
    
    start_time = time.time()
    resampled = signal.resample(y, output_samples)
    elapsed = time.time() - start_time
    
    log(f"Resampled {len(y)} → {len(resampled)} samples ({elapsed:.2f}s)", 1)
    return resampled

def calculate_hop_length(bpm):
    """Calculate hop length based on BPM to maintain 16 steps per bar."""
    # 16 steps per bar = 4 beats * 4 subdivisions per beat
    # hop_length determines time resolution
    steps_per_second = (bpm / 60) * 4  # 4 subdivisions per beat
    samples_per_step = TARGET_SR / steps_per_second
    hop_length = int(samples_per_step)
    
    # Ensure hop_length is reasonable
    hop_length = max(128, min(1024, hop_length))
    
    log(f"BPM {bpm} → hop_length {hop_length}", 1)
    return hop_length

def compute_mel_spectrogram(y, bpm):
    """Compute mel spectrogram with BPM-adjusted hop length."""
    hop_length = calculate_hop_length(bpm)
    log(f"Computing mel spectrogram (n_fft={N_FFT}, hop={hop_length})", 1)
    
    # Ensure audio is long enough
    if len(y) < N_FFT:
        log(f"Audio too short ({len(y)} samples), padding to {N_FFT}", 1)
        y = np.pad(y, (0, N_FFT - len(y)), 'constant')
    
    try:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=TARGET_SR,
            n_fft=N_FFT,
            hop_length=hop_length,
            n_mels=N_MELS,
            fmin=F_MIN,
            fmax=F_MAX,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log(f"Mel spectrogram shape: {log_mel.shape}", 1)
        return log_mel, hop_length
    except Exception as e:
        log(f"Error computing mel spectrogram: {e}", 1)
        return np.zeros((N_MELS, 1)), hop_length

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def calculate_frames_per_bar(bpm, sr, hop_length):
    """Calculate how many frames represent one bar for given BPM."""
    seconds_per_bar = 240 / bpm  # 4 beats per bar, 60 seconds per minute
    samples_per_bar = seconds_per_bar * sr
    frames_per_bar = int(samples_per_bar / hop_length)
    
    # Ensure minimum frames
    frames_per_bar = max(MIN_FRAMES_PER_BAR, frames_per_bar)
    
    log(f"BPM {bpm} → {frames_per_bar} frames per bar", 1)
    return frames_per_bar

def validate_mel_slice(mel_slice, min_frames=MIN_FRAMES_PER_BAR):
    """Validate mel slice meets minimum requirements."""
    current_frames = mel_slice.shape[1]
    
    if current_frames < min_frames:
        log(f"Mel slice too short: {current_frames} < {min_frames} frames", 1)
        return False
    
    return True

# ============================================================================
# MIDI AND FWOD PROCESSING
# ============================================================================

def _flatten_hv_list(hv_list: List[List[tuple]]) -> np.ndarray:
    """Convert a 16-step pattern to normalized FWOD vector (1×16)."""
    fwod = np.zeros(16, dtype=np.float32)

    for step_idx, onsets in enumerate(hv_list):
        for note, vel in onsets:
            cat = _GM_MAP.get(note, "high")  # default to 'high'
            fwod[step_idx] += vel * _WEIGHTS[cat]

    max_val = fwod.max() or 1.0
    return fwod / max_val

def _midi_to_fwod_vectors(path: str) -> List[np.ndarray]:
    """Read a MIDI file and return a list of FWOD vectors (one per 16-step bar)."""
    try:
        mid = mido.MidiFile(path)
        sixteenth = mid.ticks_per_beat / 4
        events = []

        time_acc = 0
        for track in mid.tracks:
            for msg in track:
                time_acc += msg.time
                if msg.type == "note_on" and msg.velocity:
                    if msg.note in _GM_MAP:
                        step = int(time_acc / sixteenth + 0.45)
                        events.append((step, msg.note, msg.velocity / 127.0))

        if not events:
            log(f"No events found in MIDI file", 1)
            return []

        # Get total steps (rounded to multiple of 16)
        last_step = max(e[0] for e in events)
        total_steps = ((last_step // 16) + 1) * 16

        # Create empty grid
        grid = [[] for _ in range(total_steps)]
        for step, note, vel in events:
            if step < total_steps:
                grid[step].append((note, vel))

        # Split into bars and filter sparse bars
        fwod_vectors = []
        for bar_idx in range(total_steps // 16):
            fragment = grid[bar_idx * 16 : (bar_idx + 1) * 16]
            if sum(bool(p) for p in fragment) > 4:  # Minimum density check
                fwod_vectors.append(_flatten_hv_list(fragment))
                
        log(f"Generated {len(fwod_vectors)} FWOD vectors", 1)
        return fwod_vectors
    except Exception as e:
        log(f"Error processing MIDI file: {e}", 1)
        return []

# Cache MIDI processing to avoid repeated parsing
_fwod_cache = lru_cache(maxsize=1024)(_midi_to_fwod_vectors)

def compute_fwod_vector(midi_path: Path, bar_index: int) -> np.ndarray:
    """Get the FWOD vector (1×16) for a specific bar in a MIDI file."""
    try:
        vectors = _fwod_cache(str(midi_path))
        if 0 <= bar_index < len(vectors):
            return vectors[bar_index]
        log(f"Bar index {bar_index} out of range (max: {len(vectors)-1 if vectors else -1})", 1)
    except Exception as e:
        log(f"Error computing FWOD vector: {e}", 1)
    
    return np.zeros(16, dtype=np.float32)

# ============================================================================
# SLICING AND DATASET CREATION
# ============================================================================

def bar_frame_bounds(num_frames: int, frames_per_bar: int) -> List[Tuple[int, int]]:
    """Calculate the frame boundaries for each bar in a spectrogram."""
    log(f"Calculating bar boundaries for {num_frames} frames ({frames_per_bar} per bar)", 1)
    
    bars = []
    start = 0
    
    # Handle too short spectrograms
    if num_frames < frames_per_bar:
        log(f"Warning: Only {num_frames} frames, not enough for a complete bar", 1)
        return []
    
    # Generate complete bars
    while start + frames_per_bar <= num_frames:
        bars.append((start, start + frames_per_bar))
        start += frames_per_bar
    
    log(f"Found {len(bars)} bars", 1)
    return bars

def slice_mel_and_fwod(
    mel: np.ndarray,
    midi_path: Path,
    bpm: float,
    hop_length: int,
    meta: Dict,
) -> List[Example]:
    """Generate Example objects for each bar, storing original variable lengths."""
    log(f"Slicing spectrogram and matching with FWOD", 1)
    
    if not midi_path.exists():
        log(f"MIDI file not found: {midi_path}", 1)
        return []
    
    try:
        # Calculate frames per bar for this BPM
        frames_per_bar = calculate_frames_per_bar(bpm, TARGET_SR, hop_length)
        
        # Get bar boundaries
        bar_bounds = bar_frame_bounds(mel.shape[1], frames_per_bar)
        examples = []
        
        if not bar_bounds:
            log(f"No valid bars found", 1)
            return []
        
        # Process each bar
        for bar_idx, (f0, f1) in enumerate(bar_bounds):
            try:
                # Get FWOD vector for this bar
                fwod_vector = compute_fwod_vector(midi_path, bar_idx)
                
                # Skip if using all zeros (not found in MIDI)
                if np.sum(fwod_vector) == 0:
                    continue
                
                # Extract the mel slice for this bar
                mel_slice = mel[:, f0:f1]
                
                # Validate slice
                if not validate_mel_slice(mel_slice):
                    continue
                
                # Create example (no padding)
                examples.append(Example(
                    mel=mel_slice,
                    fwod=fwod_vector,
                    meta={
                        **meta, 
                        "bar_index": bar_idx,
                        "frames": mel_slice.shape[1],
                        "frames_per_bar": frames_per_bar,
                    },
                ))
            except Exception as e:
                log(f"Error processing bar {bar_idx}: {e}", 1)
                continue
        
        log(f"Created {len(examples)} examples", 1)
        return examples
    except Exception as e:
        log(f"Error in slice_mel_and_fwod: {e}", 1)
        return []

def save_examples_to_npz(examples: List[Example], out_path: Path) -> None:
    """Save examples to a compressed NPZ file."""
    if not examples:
        log(f"No examples to save")
        return
    
    try:
        # Convert to Path object if it's a string
        if not isinstance(out_path, Path):
            out_path = Path(out_path)
        
        # Handle empty path
        if str(out_path) == '':
            script_dir = Path(__file__).parent
            out_path = script_dir / "output" / "mel_fwod_dataset_variable.npz"
            log(f"Empty output path detected, using default: {out_path}")
            
        # Handle relative paths
        if not out_path.is_absolute():
            script_dir = Path(__file__).parent
            out_path = script_dir / "output" / out_path.name
            
        log(f"Saving {len(examples)} examples to {out_path}")
        
        # Make sure directory exists
        os.makedirs(out_path.parent, exist_ok=True)
        
        # Store as object arrays (variable lengths)
        mels = np.empty(len(examples), dtype=object)
        for i, ex in enumerate(examples):
            mels[i] = ex.mel
        
        fwods = np.stack([ex.fwod for ex in examples])
        
        metas = np.empty(len(examples), dtype=object)
        for i, ex in enumerate(examples):
            metas[i] = ex.meta
        
        # Save compressed
        np.savez_compressed(
            out_path,
            mel=mels,
            fwod=fwods,
            meta=metas,
        )
        
        log(f"✅ Saved {len(examples)} examples to {out_path}")
        frame_counts = [ex.mel.shape[1] for ex in examples]
        log(f"Dataset shapes: mel={mels.shape} (frames: {min(frame_counts)}-{max(frame_counts)}), fwod={fwods.shape}")
    except Exception as e:
        log(f"Error saving examples: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================================

def process_audio_file(audio_path: Path, midi_path: Path, bpm: float, meta: Dict) -> List[Example]:
    """Process a single audio file with its corresponding MIDI."""
    log(f"Processing file: {audio_path.name} (BPM: {bpm})")
    
    try:
        # 1. Load audio
        start_time = time.time()
        y, sr = sf.read(audio_path, dtype="float32")
        log(f"Audio loaded: {len(y)} samples, shape={y.shape}, sr={sr}Hz", 1)
        
        # 2. Convert to mono
        y_mono = stereo_to_mono(y)
        
        # 3. Resample to target SR if needed
        if sr != TARGET_SR:
            y_mono = fast_resample(y_mono, sr, TARGET_SR)
        
        # 4. Compute mel spectrogram (with BPM-adjusted hop length)
        mel_db, hop_length = compute_mel_spectrogram(y_mono, bpm)
        
        # 5. Slice and match with FWOD
        examples = slice_mel_and_fwod(mel_db, midi_path, bpm, hop_length, meta)
        
        elapsed = time.time() - start_time
        log(f"Processed in {elapsed:.2f}s: generated {len(examples)} examples", 1)
        
        return examples
    except Exception as e:
        log(f"Error processing file: {e}", 1)
        return []

def audio_to_train_dataset(
    csv_rows: List[Dict],
    root_dir: Path,
    out_npz: Path,
    batch_size: int = 20,
) -> None:
    """Main pipeline function that processes audio files to create a dataset."""
    # Update stats
    stats.total_files = len(csv_rows)
    stats.start_time = time.time()
    
    log(f"Processing {stats.total_files} files in batches of {batch_size}")
    log(f"Using variable-length approach (padding during batching)")
    log(f"Output will be saved to: {out_npz}")
    
    all_examples = []
    
    # Process in batches
    for batch_idx, i in enumerate(range(0, len(csv_rows), batch_size)):
        log(f"===== Batch {batch_idx+1}/{(len(csv_rows) + batch_size - 1) // batch_size} =====")
        stats.print_progress()
        print()  # New line after progress
        
        batch = csv_rows[i:i+batch_size]
        batch_examples = []
        
        # Process each file in batch
        for row in batch:
            try:
                audio_path = Path(os.path.join(root_dir, row["audio_filename"]))
                midi_path = Path(os.path.join(root_dir, row["midi_filename"]))
                bpm = float(row["bpm"])
                
                # Create metadata
                meta_common = {
                    "drummer": row["drummer"],
                    "style": row["style"],
                    "session": row["session"],
                    "bpm_orig": bpm,
                    "filename": row["audio_filename"],
                }
                
                # Process this file
                examples = process_audio_file(audio_path, midi_path, bpm, meta_common)
                
                # Update stats
                stats.update(len(examples) > 0, len(examples))
                
                # Add to batch examples
                batch_examples.extend(examples)
                
            except Exception as e:
                log(f"Error processing row: {e}", 1)
                stats.update(False)
        
        # Add batch examples to total
        all_examples.extend(batch_examples)
        log(f"Batch {batch_idx+1} complete: +{len(batch_examples)} examples, {len(all_examples)} total")
        
        # Save intermediate results
        if batch_examples:
            intermediate_path = out_npz.with_stem(f"{out_npz.stem}_partial{batch_idx+1}")
            save_examples_to_npz(batch_examples, intermediate_path)
    
    # Save final dataset
    if all_examples:
        save_examples_to_npz(all_examples, out_npz)
        log(f"✅ Complete dataset saved with {len(all_examples)} examples")
    else:
        log(f"❌ No examples were generated")
    
    # Print final stats
    elapsed = time.time() - stats.start_time
    log(f"\nProcessing complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log(f"Files processed: {stats.files_processed}/{stats.total_files}")
    log(f"Files successful: {stats.files_successful}/{stats.total_files} ({stats.files_successful/stats.total_files*100:.1f}%)")
    log(f"Total examples: {stats.total_examples}")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Audio processing pipeline with variable-length approach",
    )
    
    parser.add_argument(
        "--mode", 
        choices=["test", "small", "all"], 
        default="test",
        help="Processing mode: test (1 file), small (5 files), or all files"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=5,
        help="Number of files to process in each batch"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="mel_fwod_dataset_variable.npz",
        help="Output file path for the dataset"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/groove",
        help="Root directory containing audio files"
    )
    
    parser.add_argument(
        "--csv", 
        type=str, 
        default="/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/info_clean.csv",
        help="Path to CSV file with metadata"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    # Parse arguments
    args = parse_args()
    
    # Load metadata
    log(f"Loading metadata from {args.csv}")
    df = pd.read_csv(args.csv)
    rows = df.to_dict(orient="records")
    
    # Select subset based on mode
    if args.mode == "test":
        log(f"TEST MODE: Processing only first file")
        rows = [rows[0]]
    elif args.mode == "small":
        log(f"SMALL MODE: Processing first 5 files")
        rows = rows[:5]
    else:
        log(f"Processing all {len(rows)} files")
    
    # Create output directory
    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Run pipeline
    audio_to_train_dataset(
        rows,
        Path(args.data_dir),
        output_path,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎵 AUDIO PIPELINE WITH VARIABLE-LENGTH APPROACH")
    print("="*70 + "\n")
    main()
