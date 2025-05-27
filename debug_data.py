#!/usr/bin/env python3
"""Debug script to verify data loading."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video_penibility.datasets import DatasetFactory
from video_penibility.config.schema import DataConfig
import torch

# Create config
data_config = DataConfig(
    annotation_path="/mnt/user_disk/skhazem/storage_1_10T/ai_vision_share/data/labels/annotation_skeewai_v2.csv",
    features_root="/mnt/user_disk/skhazem/storage_1_10T/ai_vision_share/features",
    features_type="swin3d_t",
    data_type="full_body"
)

# Create dataset
dataset = DatasetFactory.create_dataset(data_config)

print(f"Dataset size: {len(dataset)}")
print(f"Feature dimension: {dataset.get_feature_dim()}")

# Check first 10 samples
print("\nFirst 10 samples:")
for i in range(min(10, len(dataset))):
    features, target = dataset[i]
    item = dataset.data_items[i]
    
    print(f"Sample {i+1}:")
    print(f"  Clip ID: {item['clip_id']}")
    print(f"  Features shape: {features.shape}")
    print(f"  Target: {target.item():.1f} (raw: {item['raw_target']})")
    print(f"  Video: {item['video_path']}")
    print()

# Check if there are any obvious patterns
print("Checking for potential issues:")

# Check if all features have same sequence length
seq_lengths = []
for i in range(min(50, len(dataset))):
    features, _ = dataset[i]
    seq_lengths.append(features.shape[0])

print(f"Sequence lengths - Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, Mean: {sum(seq_lengths)/len(seq_lengths):.1f}")

# Check target variance per subject
subjects_targets = {}
for item in dataset.data_items[:50]:
    subject = item['subject_id']
    target = item['raw_target']
    if subject not in subjects_targets:
        subjects_targets[subject] = []
    subjects_targets[subject].append(target)

print("\nTarget variance per subject (first 50 samples):")
for subject, targets in subjects_targets.items():
    print(f"  {subject}: {targets} (unique: {len(set(targets))})") 