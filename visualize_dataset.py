import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory for plots
plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "plots")
os.makedirs(plot_dir, exist_ok=True)

# Load the NPZ file
data_path = "/Users/danielmartinezvillegas/Developer/master-ds/✨TDG/audio_pipeline/output/mel_fwod_dataset.npz"
data = np.load(data_path, allow_pickle=True)

# Print the keys in the file
print("Keys in the file:", data.files)

# Get the shapes of the arrays
print("\nArray shapes:")
for key in data.files:
    print(f"{key}: {data[key].shape}")

# Print number of examples
num_examples = len(data['mel'])
print(f"\nNumber of examples: {num_examples}")

# Visualization function
def visualize_example(index, save=True):
    """Visualize one example from the dataset"""
    # Get data for this example
    mel = data['mel'][index]
    fwod = data['fwod'][index]
    meta = data['meta'][index]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot mel spectrogram
    plt.subplot(2, 1, 1)
    plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel Spectrogram - {meta.get('style', 'Unknown Style')}")
    plt.ylabel('Mel Band')
    
    # Plot FWOD vector
    plt.subplot(2, 1, 2)
    plt.bar(range(len(fwod)), fwod)
    plt.title('FWOD Vector')
    plt.xlabel('Step')
    plt.ylabel('Weighted Density')
    plt.tight_layout()
    
    # Save or show
    if save:
        style = meta.get('style', 'unknown').replace('/', '-')
        filename = f"example_{index}_{style}.png"
        plt.savefig(os.path.join(plot_dir, filename))
        print(f"Saved visualization to {os.path.join(plot_dir, filename)}")
        plt.close()
    else:
        plt.show()

# Visualize first few examples
print("\nGenerating visualizations...")
for i in range(min(5, num_examples)):
    print(f"Example {i}:")
    print(f"  - Mel shape: {data['mel'][i].shape}")
    print(f"  - FWOD shape: {data['fwod'][i].shape}")
    print(f"  - Style: {data['meta'][i].get('style', 'Unknown')}")
    print(f"  - File: {data['meta'][i].get('filename', 'Unknown')}")
    
    # Create visualization
    visualize_example(i)

print(f"\nAll visualizations saved to {plot_dir}")
print("\nDataset Summary:")
print(f"- Total examples: {num_examples}")

# Count examples by style
styles = {}
for i in range(num_examples):
    style = data['meta'][i].get('style', 'Unknown')
    if style not in styles:
        styles[style] = 0
    styles[style] += 1

print("- Examples per style:")
for style, count in styles.items():
    print(f"  • {style}: {count}")
