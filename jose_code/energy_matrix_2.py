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


root_dir = '/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/groove-aligned'
info_csv_path = '/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/info_clean.csv'
info_clean = pd.read_csv(info_csv_path)
energy_matrix_df = info_clean[['drummer', 'session', 'id',
                               'style', 'midi_filename', 'audio_filename', 'bpm', 'split']]
output_csv_path = '/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/data/improved_audio_features.csv'


GM_dict = {
    # key is midi note number
    # values are:
    # [0] name (as string)
    # [1] name category low mid or high (as string)
    # [2] substiture midi number for simplified MIDI (all instruments)
    # [3] name of instrument for 8 note conversion (as string)
    # [4] number of instrument for 8 note conversion
    # [5] substiture midi number for conversion to 8 note
    # [6] substiture midi number for conversion to 16 note
    # [7] substiture midi number for conversion to 3 note
    # if we are going to remap just use GM_dict[msg.note][X]
    22: ["Closed Hi-Hat edge", "high", 42, "CH", 3, 42, 42, 42],
    26: ["Open Hi-Hat edge", "high", 46, "OH", 4, 46, 46, 42],
    35: ["Acoustic Bass Drum", "low", 36, "K", 1, 36, 36, 36],
    36: ["Bass Drum 1", "low", 36, "K", 1, 36, 36, 36],
    37: ["Side Stick", "mid", 37, "RS", 6, 37, 37, 38],
    38: ["Acoustic Snare", "mid", 38, "SN", 2, 38, 38, 38],
    39: ["Hand Clap", "mid", 39, "CP", 5, 39, 39, 38],
    40: ["Electric Snare", "mid", 38, "SN", 2, 38, 38, 38],
    41: ["Low Floor Tom", "low", 45, "LT", 7, 45, 45, 36],
    42: ["Closed Hi Hat", "high", 42, "CH", 3, 42, 42, 42],
    43: ["High Floor Tom", "mid", 45, "HT", 8, 45, 45, 38],
    44: ["Pedal Hi-Hat", "high", 46, "OH", 4, 46, 46, 42],
    45: ["Low Tom", "low", 45, "LT", 7, 45, 45, 36],
    46: ["Open Hi-Hat", "high", 46, "OH", 4, 46, 46, 42],
    47: ["Low-Mid Tom", "low", 47, "MT", 7, 45, 47, 36],
    48: ["Hi-Mid Tom", "mid", 47, "MT", 7, 50, 50, 38],
    49: ["Crash Cymbal 1", "high", 49, "CC", 4, 46, 42, 42],
    50: ["High Tom", "mid", 50, "HT", 8, 50, 50, 38],
    51: ["Ride Cymbal 1", "high", 51, "RC", -1, 42, 51, 42],
    52: ["Chinese Cymbal", "high", 52, "", -1, 46, 51, 42],
    53: ["Ride Bell", "high", 53, "", -1, 42, 51, 42],
    54: ["Tambourine", "high", 54, "", -1, 42, 69, 42],
    55: ["Splash Cymbal", "high", 55, "OH", 4, 46, 42, 42],
    56: ["Cowbell", "high", 56, "CB", -1, 37, 56, 42],
    57: ["Crash Cymbal 2", "high", 57, "CC", 4, 46, 42, 42],
    58: ["Vibraslap", "mid", 58, "VS", 6, 37, 37, 42],
    59: ["Ride Cymbal 2", "high", 59, "RC", 3, 42, 51, 42],
    60: ["Hi Bongo", "high", 60, "LB", 8, 45, 63, 42],
    61: ["Low Bongo", "mid", 61, "HB", 7, 45, 64, 38],
    62: ["Mute Hi Conga", "mid", 62, "MC", 8, 50, 62, 38],
    63: ["Open Hi Conga", "high", 63, "HC", 8, 50, 63, 42],
    64: ["Low Conga", "low", 64, "LC", 7, 45, 64, 36],
    65: ["High Timbale", "mid", 65, "", 8, 45, 63, 38],
    66: ["Low Timbale", "low", 66, "", 7, 45, 64, 36],
    67: ["High Agogo", "high", 67, "", -1, 37, 56, 42],
    68: ["Low Agogo", "mid", 68, "", -1, 37, 56, 38],
    69: ["Cabasa", "high", 69, "MA", -1, 42, 69, 42],
    70: ["Maracas", "high", 69, "MA", -1, 42, 69, 42],
    71: ["Short Whistle", "high", 71, "", -1, 37, 56, 42],
    72: ["Long Whistle", "high", 72, "", -1, 37, 56, 42],
    73: ["Short Guiro", "high", 73, "", -1, 42, 42, 42],
    74: ["Long Guiro", "high", 74, "", -1, 46, 46, 42],
    75: ["Claves", "high", 75, "", -1, 37, 75, 42],
    76: ["Hi Wood Block", "high", 76, "", 8, 50, 63, 42],
    77: ["Low Wood Block", "mid", 77, "", 7, 45, 64, 38],
    78: ["Mute Cuica", "high", 78, "", -1, 50, 62, 42],
    79: ["Open Cuica", "high", 79, "", -1, 45, 63, 42],
    80: ["Mute Triangle", "high", 80, "", -1, 37, 75, 42],
    81: ["Open Triangle", "high", 81, "", -1, 37, 75, 42],
}


def midifile2hv_list(file_name, mapping):
    '''
    pattern name must include .mid
    get a MIDI file and convert it to an hv_list (a list of note numbers and velocity)
    use the "mapping" variable to define the type of instrument mapping
    that will be used in the hv_list "all", "16", "8", "3"
    '''
    pattern = []
    mid = mido.MidiFile(file_name)  # create a mido file instance
    sixteenth = mid.ticks_per_beat/4  # find the length of a sixteenth note
    # print ("sixteenth", sixteenth)

    # time: inside a track, it is delta time in ticks (integrer).
    # A delta time is how long to wait before the next message.
    acc = 0  # use this to keep track of time

    # depending on the instruments variable select a notemapping
    if mapping == "all":
        column = 2
    elif mapping == "16":
        column = 6
    elif mapping == "8":
        column = 5
    elif mapping == "3":
        column = 7
    else:
        column = 2  # if no mapping is selected use "allinstrument" mapping

    for i, track in enumerate(mid.tracks):
        for msg in track:  # process all messages
            acc += msg.time  # accumulate time of any message type
            if msg.type == "note_on":
                # remap msg.note by demand
                midinote = GM_dict[msg.note][column]
                rounded_step = int((acc/sixteenth)+0.45)
                midivelocity = float(msg.velocity)/127  # normalize upfront
                # step, note, velocity
                pattern.append((int(acc/sixteenth), midinote, midivelocity))
        if len(pattern) > 0:  # just proceed if analyzed pattern has at least one onset
            # round the pattern to the next multiple of 16
            pattern_len_in_steps = 16 * \
                ((rounded_step//16)+((rounded_step % 16)+16)//16)
            # create an empty list of lists the size of the pattern
            output_pattern = [[]]*pattern_len_in_steps
            # group the instruments and their velocity that played at a specific step
            i = 0
            for step in range(pattern_len_in_steps):
                step_content = [(x[1], x[2]) for x in pattern if x[0] == step]
                # make sure no notes are repeated and events are sorted
                # present_notes = [x[0] for x in step_content]
                # unique_notes_in_step = list(set(present_notes))
                # remove repeated notes at the same step
                result = {}
                for tup in step_content:
                    note, vel = tup
                    if note not in result or vel > result[note][1]:
                        result[note] = tup

                # dictionary to tuple list
                step_content = list(result.values())
                step_content.sort()  # sort by note ascending
                output_pattern[step] = step_content

    ##################################
    # split the pattern every 16 steps
    ##################################
    hv_lists_split = []
    for x in range(len(output_pattern)//16):
        patt_fragment = output_pattern[x*16:(x*16)+16]
        patt_density = sum([1 for x in patt_fragment if x != []])
        #############################################################
        # filter out patterns that have less than 4 events with notes
        #############################################################
        # NOTE: more conditions could be added (i.e. kick on step 0, etc)
        #############################################################
        if patt_density > 4:
            hv_lists_split.append(patt_fragment)
  # output is a 16-step pattern
    return hv_lists_split


def flatten_hv_list(hv_list):
    # input an hv list and output a flattened representation as a v_list

  # list of MIDI instruments and categories
    lows = [35, 36, 41, 45, 47, 64, 66]
    mids = [37, 38, 39, 40, 43, 48, 50, 61, 62, 65, 68, 77]
    his = [22, 26, 42, 44, 46, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59,
           60, 63, 67, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81]

    flat = np.zeros([len(hv_list), 1])

    # multiply velocities and categories
    for i, step in enumerate(hv_list):
        step_weight = 0
        for onset in step:
            if onset[0] in lows:
                step_weight += onset[1]*3
            elif onset[0] in mids:
                step_weight += onset[1]*2
            else:
                step_weight += onset[1]*1
            flat[i] = step_weight

    flat = flat/max(flat)

    return flat


"""## Dataloader

We read the dataframe, filter the examples that are not beats and return the audio vector, sample rate, midi representation, bpm, and the style
"""


class GrooveDataset(Dataset):
    "Audio dataset for midi groove"

    def __init__(self, root_dir):

        df = pd.read_csv(info_csv_path)

        df = df[df["beat_type"] == "beat"]
        initial_len = len(df)
        df = df.dropna(subset=["audio_filename", "midi_filename"])
        filtered_len = len(df)

        if initial_len > filtered_len:
            print(
                f"Filtered out {initial_len - filtered_len} examples with NaN filenames")
        self.root_dir = root_dir
        self.data = df.to_dict(orient="records")

    def __getitem__(self, index):
        data_dict = self.data[index]

        x, fs = torchaudio.load(os.path.join(
            self.root_dir, data_dict["audio_filename"]))
        bpm = data_dict["bpm"]
        hv_list = midifile2hv_list(os.path.join(
            self.root_dir, data_dict["midi_filename"]), GM_dict)
        midi_reps = []
        for e in hv_list:
            midi_reps.append(flatten_hv_list(e))

        return x, fs, midi_reps, bpm, data_dict["style"]

    def __len__(self):
        return len(self.data)


groove = GrooveDataset(root_dir)
dataloader = DataLoader(groove, batch_size=1, shuffle=True)


batch = next(iter(dataloader))
x, fs, midi_rep, bpm, style = batch

# select only the first segment
segment_duration_seconds = 16 * (60 / bpm / 4)
print(f"Segment duration {segment_duration_seconds} seconds")
segment_duration_samples = int(segment_duration_seconds * fs.item())
sample = x[0, 0, :segment_duration_samples].numpy()


mel = 10*np.log10(librosa.feature.melspectrogram(y=sample,
                  fmax=8000) + 0.000001)
n_bins, n_frames = mel.shape
print(f"mel shape {n_bins, n_frames}")

n_sixteenth = 16
frame_size = n_frames // n_sixteenth  # Determine grouping size

# Reshape and average over time
mel_time_average = mel[:, :frame_size *
                       n_sixteenth].reshape(n_bins, n_sixteenth, frame_size).mean(axis=2)

print(f"mel time average shape {mel_time_average.shape}")  # (128, 16)


# Define weight vector
weight_vector = torch.zeros(1, n_bins)
weight_vector[0, 0:20] = 0.4
weight_vector[0, 20:60] = 0.3
weight_vector[0, 60:] = 0.2

mel_time_average = torch.tensor(mel_time_average, dtype=torch.float32)
y_hat = weight_vector @ mel_time_average
print(f"y_hat shape {y_hat.shape}")  # (1, 16)


n_sixteenth = 16
n_bins = 128

# Define weight vector with proper initialization
weight_vector = nn.Parameter(torch.zeros(1, n_bins))
# Initialize values
weight_vector.data[0, 0:20] = 0.4
weight_vector.data[0, 20:60] = 0.3
weight_vector.data[0, 60:] = 0.2

# Define loss and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam([weight_vector], lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_vector.to(device)
for epoch in range(10):
    # Training loop
    for i, batch in enumerate(dataloader):
        x, fs, midi_rep, bpm, style = batch
        x.to(device)
        fs.to(device)
        bpm.to(device)
        # Select only the first segment
        segment_duration_seconds = n_sixteenth * (60 / bpm.item() / 4)
        segment_duration_samples = int(segment_duration_seconds * fs.item())
        sample = x[0, 0, :segment_duration_samples].cpu().numpy()

        # Extract mel
        mel = 10 * np.log10(librosa.feature.melspectrogram(y=sample,
                            sr=fs.item(), fmax=8000) + 0.000001)
        n_bins, n_frames = mel.shape
        frame_size = n_frames // n_sixteenth

        # Reshape and average over time
        mel_time_average = mel[:, :frame_size * n_sixteenth].reshape(
            n_bins, n_sixteenth, frame_size).mean(axis=2)

        # Convert to tensor and move to device (same as input x)

        mel_time_average = torch.tensor(
            mel_time_average, dtype=torch.float32).to(device)
        weight_vector = weight_vector.to(device)

        # Forward pass
        y_hat = torch.matmul(weight_vector, mel_time_average)
        y_hat = y_hat / y_hat.max()
        # Prepare target
        if len(midi_rep) < 1:
            continue

        y = midi_rep[0].view(1, -1).to(device).to(torch.float32)

        # Check if dimensions match before computing loss
        if y.shape != y_hat.shape:
            print(f"Shape mismatch: y {y.shape}, y_hat {y_hat.shape}")
            continue

        # Backward pass and optimization
        optimizer.zero_grad()
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Batch {i}, Loss: {loss.item():.6f}")

print(weight_vector)
torch.save(weight_vector.detach().cpu(), "/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/jose_code/weight_vector.pt")

