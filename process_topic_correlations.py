"""
Process BERTopic outputs to generate time series of topic correlation matrices.

This script takes BERTopic outputs (topic distributions over time) and computes
rolling correlation matrices between topics for each participant.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import argparse

# Default paths
DEFAULT_INPUT_DIR = Path("outputs/BERTopic_outputs")
DEFAULT_OUTPUT_DIR = Path("outputs/topic_correlation_matrices")
DEFAULT_WINDOW_SIZE = 10  # Number of time steps for rolling correlation
DEFAULT_MIN_WINDOW = 5    # Minimum number of observations needed to compute correlation

def load_bertopic_output(participant_file: Path) -> pd.DataFrame:
    """Load BERTopic output for a single participant."""
    try:
        df = pd.read_csv(participant_file)
        # Ensure we have the required columns
        required_cols = ['timestamp', 'text']
        metatopic_cols = [col for col in df.columns if col.startswith('MetaTopic') and not col.endswith('_prob')]
        
        if not all(col in df.columns for col in required_cols) or not metatopic_cols:
            print(f"Warning: Missing required columns in {participant_file}")
            return None
            
        # Convert timestamp to datetime and sort
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        return df
    except Exception as e:
        print(f"Error loading {participant_file}: {str(e)}")
        return None

def compute_rolling_correlations(df: pd.DataFrame, 
                               window_size: int = 10, 
                               min_window: int = 5) -> Dict[str, np.ndarray]:
    """
    Compute rolling correlation matrices for a participant's topic distributions.
    
    Args:
        df: DataFrame with columns 'timestamp' and metatopic columns
        window_size: Number of time steps to include in each window
        min_window: Minimum number of observations needed to compute correlation
        
    Returns:
        Dictionary with 'timestamps' and 'correlation_matrices' arrays
    """
    # Get metatopic columns (those starting with 'MetaTopic' and not ending with '_prob')
    metatopic_cols = [col for col in df.columns 
                     if col.startswith('MetaTopic') and not col.endswith('_prob')]
    
    if not metatopic_cols:
        return {'timestamps': np.array([]), 'correlation_matrices': np.array([])}
    
    # Get topic distributions
    topic_distributions = df[metatopic_cols].values
    timestamps = df['timestamp'].values
    
    # Initialize arrays to store results
    n_windows = max(1, len(df) - window_size + 1)
    n_topics = len(metatopic_cols)
    
    # If we don't have enough data for even one full window, return empty results
    if len(df) < min_window:
        return {'timestamps': np.array([]), 'correlation_matrices': np.array([])}
    
    # For each window, compute correlation matrix
    correlation_matrices = []
    window_timestamps = []
    
    for i in range(len(df) - window_size + 1):
        window_data = topic_distributions[i:i + window_size]
        
        # Only compute correlation if we have enough non-NaN values
        valid_rows = ~np.isnan(window_data).any(axis=1)
        if np.sum(valid_rows) < min_window:
            continue
            
        # Compute correlation matrix
        corr_matrix = np.corrcoef(window_data[valid_rows].T)
        
        # Handle potential NaN values in correlation matrix
        np.fill_diagonal(corr_matrix, 1.0)  # Diagonal should be 1.0
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        correlation_matrices.append(corr_matrix)
        window_timestamps.append(timestamps[i + window_size - 1])  # Use end of window as timestamp
    
    if not correlation_matrices:
        return {'timestamps': np.array([]), 'correlation_matrices': np.array([])}
    
    return {
        'timestamps': np.array(window_timestamps),
        'correlation_matrices': np.stack(correlation_matrices),
        'topic_names': metatopic_cols
    }

def save_correlation_matrices(participant_id: str, 
                            correlation_data: Dict[str, np.ndarray],
                            output_dir: Path):
    """Save correlation matrices and metadata for a participant."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create participant directory
    participant_dir = output_dir / f"participant_{participant_id}"
    participant_dir.mkdir(exist_ok=True)
    
    # Save timestamps
    np.save(participant_dir / 'timestamps.npy', correlation_data['timestamps'])
    
    # Save correlation matrices
    np.save(participant_dir / 'correlation_matrices.npy', 
            correlation_data['correlation_matrices'])
    
    # Save metadata
    metadata = {
        'participant_id': participant_id,
        'num_matrices': len(correlation_data['correlation_matrices']),
        'num_topics': len(correlation_data.get('topic_names', [])),
        'topic_names': correlation_data.get('topic_names', []),
        'timestamp_format': 'ISO8601',
        'matrix_format': 'numpy_array',
        'matrix_shape': list(correlation_data['correlation_matrices'].shape[1:]) if correlation_data['correlation_matrices'].size > 0 else []
    }
    
    with open(participant_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def process_participant(participant_file: Path, 
                       output_dir: Path, 
                       window_size: int, 
                       min_window: int) -> bool:
    """Process a single participant's data."""
    try:
        participant_id = participant_file.stem
        print(f"Processing {participant_id}...")
        
        # Load BERTopic output
        df = load_bertopic_output(participant_file)
        if df is None or df.empty:
            print(f"  No valid data for {participant_id}")
            return False
        
        # Compute rolling correlations
        correlation_data = compute_rolling_correlations(
            df, window_size=window_size, min_window=min_window
        )
        
        # Save results
        if correlation_data['correlation_matrices'].size > 0:
            save_correlation_matrices(participant_id, correlation_data, output_dir)
            print(f"  Saved {len(correlation_data['correlation_matrices'])} correlation matrices")
            return True
        else:
            print(f"  Not enough data to compute correlations for {participant_id}")
            return False
            
    except Exception as e:
        print(f"Error processing {participant_file}: {str(e)}")
        return False

def process_all_participants(input_dir: Path, 
                           output_dir: Path, 
                           window_size: int = DEFAULT_WINDOW_SIZE,
                           min_window: int = DEFAULT_MIN_WINDOW):
    """Process all participant files in the input directory."""
    # Find all CSV files in the input directory
    participant_files = list(input_dir.glob('*.csv'))
    
    if not participant_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(participant_files)} participant files")
    
    # Process each participant
    success_count = 0
    for participant_file in tqdm(participant_files, desc="Processing participants"):
        if process_participant(participant_file, output_dir, window_size, min_window):
            success_count += 1
    
    print(f"\nProcessing complete. Successfully processed {success_count}/{len(participant_files)} participants.")
    print(f"Results saved to: {output_dir.absolute()}")

def main():
    parser = argparse.ArgumentParser(description='Process BERTopic outputs to generate topic correlation matrices.')
    parser.add_argument('--input-dir', type=str, default=str(DEFAULT_INPUT_DIR),
                        help=f'Directory containing BERTopic output CSVs (default: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f'Directory to save correlation matrices (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--window-size', type=int, default=DEFAULT_WINDOW_SIZE,
                        help=f'Number of time steps in each rolling window (default: {DEFAULT_WINDOW_SIZE})')
    parser.add_argument('--min-window', type=int, default=DEFAULT_MIN_WINDOW,
                        help=f'Minimum observations needed to compute correlation (default: {DEFAULT_MIN_WINDOW})')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing BERTopic outputs from: {input_dir}")
    print(f"Saving correlation matrices to: {output_dir}")
    print(f"Window size: {args.window_size}, Minimum window: {args.min_window}")
    
    # Process all participants
    process_all_participants(
        input_dir=input_dir,
        output_dir=output_dir,
        window_size=args.window_size,
        min_window=args.min_window
    )

if __name__ == "__main__":
    main()
