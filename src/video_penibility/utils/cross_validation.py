"""Cross-validation utilities for video penibility assessment."""

import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import KFold
import logging

logger = logging.getLogger(__name__)


def create_subject_splits(
    dataset,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """Create subject-wise cross-validation splits.
    
    Args:
        dataset: Dataset object with data_items attribute.
        n_splits: Number of cross-validation splits.
        random_state: Random state for reproducibility.
        
    Returns:
        List of (train_indices, val_indices) tuples.
    """
    # Extract subject information
    subjects = []
    subject_to_indices = {}
    
    for idx, item in enumerate(dataset.data_items):
        subject_id = item['subject_id']
        subjects.append(subject_id)
        
        if subject_id not in subject_to_indices:
            subject_to_indices[subject_id] = []
        subject_to_indices[subject_id].append(idx)
    
    # Get unique subjects
    unique_subjects = list(subject_to_indices.keys())
    unique_subjects.sort()  # For reproducibility
    
    logger.info(f"Found {len(unique_subjects)} unique subjects: {unique_subjects}")
    
    # Create subject-wise splits
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    
    for fold_idx, (train_subject_indices, val_subject_indices) in enumerate(kfold.split(unique_subjects)):
        # Get subject IDs for this fold
        train_subjects = [unique_subjects[i] for i in train_subject_indices]
        val_subjects = [unique_subjects[i] for i in val_subject_indices]
        
        # Get all data indices for these subjects
        train_indices = []
        val_indices = []
        
        for subject in train_subjects:
            train_indices.extend(subject_to_indices[subject])
        
        for subject in val_subjects:
            val_indices.extend(subject_to_indices[subject])
        
        # Sort indices for consistency
        train_indices.sort()
        val_indices.sort()
        
        splits.append((train_indices, val_indices))
        
        logger.info(
            f"Fold {fold_idx + 1}: "
            f"Train subjects: {train_subjects} ({len(train_indices)} samples), "
            f"Val subjects: {val_subjects} ({len(val_indices)} samples)"
        )
    
    return splits


def create_random_splits(
    dataset,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """Create random cross-validation splits.
    
    Args:
        dataset: Dataset object.
        n_splits: Number of cross-validation splits.
        random_state: Random state for reproducibility.
        
    Returns:
        List of (train_indices, val_indices) tuples.
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(indices)):
        train_indices = train_indices.tolist()
        val_indices = val_indices.tolist()
        
        splits.append((train_indices, val_indices))
        
        logger.info(
            f"Fold {fold_idx + 1}: "
            f"Train samples: {len(train_indices)}, "
            f"Val samples: {len(val_indices)}"
        )
    
    return splits 