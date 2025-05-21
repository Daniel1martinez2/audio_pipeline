#!/usr/bin/env python
"""
Simple script to manually fix the output directory issue.
Run this instead of complete_pipeline.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Create output directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)

# Get arguments from command line
args = sys.argv[1:]

# Add output path if not specified
has_output = False
for i, arg in enumerate(args):
    if arg == "--output" and i+1 < len(args):
        # Fix the output path if provided
        output_path = args[i+1]
        if not os.path.isabs(output_path):
            args[i+1] = os.path.join(output_dir, os.path.basename(output_path))
        has_output = True
        break

# Add default output if none provided
if not has_output:
    output_path = os.path.join(output_dir, "mel_fwod_dataset.npz")
    args.extend(["--output", output_path])

print(f"Output will be saved to: {args[args.index('--output')+1]}")

# Run the complete pipeline with fixed output path
cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "complete_pipeline.py")] + args
subprocess.call(cmd)
