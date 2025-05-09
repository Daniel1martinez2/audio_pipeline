import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import librosa
import os
import matplotlib.pyplot as plt
import mido
import math
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display
import librosa
import librosa.display

import os
import pandas as pd
import torchaudio
import librosa
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = '/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/groove-aligned'
info_csv_path = '/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/info_clean.csv'
info_clean = pd.read_csv(info_csv_path)
energy_matrix_df = info_clean[['drummer', 'session', 'id',
                               'style', 'midi_filename', 'audio_filename', 'bpm', 'split']]
output_csv_path = '/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/improved_audio_features.csv'


weight_vector = torch.load("/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/jose_code/weight_vector.pt").to(device)
weight_vector = torch.nn.Parameter(weight_vector)


def generate_prediction_dataset(csv_path, data_root, weight_vector, device='cpu'):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["audio_filename", "midi_filename"])  # Eliminate rows with null paths

    results = []

    for filename, group in df.groupby("midi_filename"):
        row = group.iloc[0]  # Only need metadata once per file
        audio_path = os.path.join(data_root, row["audio_filename"])
        bpm = row["bpm"]

        try:
            # Load audio
            x, fs = torchaudio.load(audio_path)
            x = x[0].numpy()  # Convert to mono and numpy

            # Duration of one 16-step bar in seconds and samples
            segment_duration_sec = 16 * (60 / bpm / 4)
            segment_samples = int(segment_duration_sec * fs)

            total_samples = len(x)
            num_bars = total_samples // segment_samples

            for seq in range(num_bars):
                start = seq * segment_samples
                end = start + segment_samples
                x_bar = x[start:end]

                # Compute mel spectrogram
                mel = 10 * np.log10(librosa.feature.melspectrogram(y=x_bar, sr=fs, fmax=8000) + 1e-6)
                n_bins, n_frames = mel.shape
                frame_size = n_frames // 16
                mel_avg = mel[:, :frame_size * 16].reshape(n_bins, 16, frame_size).mean(axis=2)

                # Prediction
                mel_tensor = torch.tensor(mel_avg, dtype=torch.float32).to(device)
                y_hat = torch.matmul(weight_vector.to(device), mel_tensor)
                y_hat = (y_hat / y_hat.max()).detach().cpu().numpy().flatten()

                # Store result row
                output_row = {
                    "sequence": seq,
                    "drummer": row["drummer"],
                    "session": row["session"],
                    "id": row["id"],
                    "style": row["style"],
                    "midi_filename": row["midi_filename"],
                    "audio_filename": row["audio_filename"],
                    "bpm": row["bpm"],
                    "split": row["split"]
                }

                for i in range(16):
                    output_row[f"y_{i}"] = y_hat[i]

                results.append(output_row)

        except Exception as e:
            print(f"❌ Error processing {audio_path}: {e}")
            continue

    return pd.DataFrame(results)

dataset_df = generate_prediction_dataset(
    csv_path=info_csv_path,
    data_root=root_dir,
    weight_vector=weight_vector,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

dataset_df.to_csv("predicted_vectors.csv", index=False)
