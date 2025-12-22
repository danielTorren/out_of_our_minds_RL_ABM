"""
Master Pipeline for Synthetic Sentiment Analysis

This script orchestrates the entire pipeline:
1. Run BERTopic to process input data and generate topic distributions
2. Process topic distributions into time series of correlation matrices
3. Run the ABM simulation using the correlation matrices
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import yaml
import numpy as np
import pandas as pd

# Default paths
DEFAULT_CONFIG_FILE = "config/pipeline_config.yaml"
DEFAULT_DATA_DIR = "data/raw"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_MODELS_DIR = "models"

# Default BERTopic parameters
DEFAULT_BERTOPIC_PARAMS = {
    "embedding_model": "all-MiniLM-L6-v2",
    "nr_topics": "auto",
    "min_topic_size": 10,
    "verbose": True,
    "calculate_probabilities": True,
    "diversity": 0.5
}

# Default correlation matrix parameters
DEFAULT_CORR_PARAMS = {
    "window_size": 10,
    "min_window": 5
}

# Default ABM parameters
DEFAULT_ABM_PARAMS = {
    "num_agents": 100,
    "num_steps": 1000,
    "influence_strength": 0.1,
    "random_seed": 42
}

class PipelineRunner:
    """Orchestrates the entire pipeline from data processing to ABM simulation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the pipeline with configuration."""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up paths
        self.root_dir = Path(__file__).parent.parent
        self.data_dir = Path(self.config.get('data_dir', DEFAULT_DATA_DIR)).resolve()
        self.output_dir = Path(self.config.get('output_dir', DEFAULT_OUTPUT_DIR)).resolve()
        self.models_dir = Path(self.config.get('models_dir', DEFAULT_MODELS_DIR)).resolve()
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Get parameters
        self.bertopic_params = {**DEFAULT_BERTOPIC_PARAMS, **self.config.get('bertopic_params', {})}
        self.corr_params = {**DEFAULT_CORR_PARAMS, **self.config.get('correlation_params', {})}
        self.abm_params = {**DEFAULT_ABM_PARAMS, **self.config.get('abm_params', {})}
        
        # Set up paths for intermediate and final outputs
        self.bertopic_output_dir = self.output_dir / 'bertopic_outputs'
        self.correlation_output_dir = self.output_dir / 'topic_correlation_matrices'
        self.abm_output_dir = self.output_dir / 'abm_results'
        
        # Create output directories
        self.bertopic_output_dir.mkdir(exist_ok=True)
        self.correlation_output_dir.mkdir(exist_ok=True)
        self.abm_output_dir.mkdir(exist_ok=True)
    
    @staticmethod
    def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG_FILE)
        
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"Warning: Config file not found at {config_path}, using defaults")
            return {}
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ('.yaml', '.yml'):
                return yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def run_bertopic(self) -> bool:
        """Run BERTopic to process input data and generate topic distributions."""
        print("\n=== Running BERTopic Processing ===")
        
        # Check if BERTopic outputs already exist
        if self.bertopic_output_dir.exists() and any(self.bertopic_output_dir.glob('*.csv')):
            print(f"Found existing BERTopic outputs in {self.bertopic_output_dir}")
            print("Skipping BERTopic processing. Use --force to reprocess.")
            return True
        
        try:
            # Import BERTopic here to avoid hard dependency
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from bertopic.representation import KeyBERTInspired
            
            print(f"Loading data from {self.data_dir}")
            
            # Find all participant files
            participant_files = list(self.data_dir.glob('*.csv'))
            if not participant_files:
                print(f"Error: No CSV files found in {self.data_dir}")
                return False
            
            print(f"Found {len(participant_files)} participant files")
            
            # Initialize BERTopic model
            print("Initializing BERTopic model...")
            embedding_model = SentenceTransformer(self.bertopic_params['embedding_model'])
            representation_model = KeyBERTInspired()
            
            topic_model = BERTopic(
                embedding_model=embedding_model,
                representation_model=representation_model,
                nr_topics=self.bertopic_params['nr_topics'],
                min_topic_size=self.bertopic_params['min_topic_size'],
                verbose=self.bertopic_params['verbose'],
                calculate_probabilities=self.bertopic_params['calculate_probabilities'],
                diversity=self.bertopic_params['diversity']
            )
            
            # Process each participant
            for p_file in participant_files:
                print(f"\nProcessing {p_file.name}...")
                
                # Load and preprocess data
                df = pd.read_csv(p_file)
                texts = df['text'].fillna('').astype(str).tolist()
                
                # Fit and transform with BERTopic
                topics, probs = topic_model.fit_transform(texts)
                
                # Get topic distributions
                topic_distributions = topic_model.approximate_distribution(
                    texts, 
                    min_similarity=0.1
                )
                
                # Create output DataFrame
                output_df = df[['timestamp', 'text']].copy()
                
                # Add topic distributions
                for i in range(topic_distributions.shape[1]):
                    output_df[f'MetaTopic{i}'] = topic_distributions[:, i]
                
                # Save to CSV
                output_path = self.bertopic_output_dir / f"{p_file.stem}_topics.csv"
                output_df.to_csv(output_path, index=False)
                print(f"  Saved topic distributions to {output_path}")
            
            # Save the model
            model_path = self.models_dir / 'bertopic_model'
            topic_model.save(str(model_path), serialization='safetensors')
            print(f"\nSaved BERTopic model to {model_path}")
            
            return True
            
        except Exception as e:
            print(f"Error running BERTopic: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_correlations(self) -> bool:
        """Process BERTopic outputs to generate correlation matrices."""
        print("\n=== Processing Topic Correlations ===")
        
        # Check if correlation matrices already exist
        if self.correlation_output_dir.exists() and any(self.correlation_output_dir.glob('participant_*')):
            print(f"Found existing correlation matrices in {self.correlation_output_dir}")
            print("Skipping correlation processing. Use --force to reprocess.")
            return True
        
        try:
            # Import the correlation processor
            from process_topic_correlations import process_all_participants
            
            # Run the correlation processing
            process_all_participants(
                input_dir=self.bertopic_output_dir,
                output_dir=self.correlation_output_dir,
                window_size=self.corr_params['window_size'],
                min_window=self.corr_params['min_window']
            )
            
            return True
            
        except Exception as e:
            print(f"Error processing correlations: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_abm(self) -> bool:
        """Run the ABM simulation using the correlation matrices."""
        print("\n=== Running ABM Simulation ===")
        
        try:
            # Check if ABM outputs already exist
            if self.abm_output_dir.exists() and any(self.abm_output_dir.glob('*.csv')):
                print(f"Found existing ABM outputs in {self.abm_output_dir}")
                print("Skipping ABM simulation. Use --force to rerun.")
                return True
            
            # Import ABM components
            from abm_simulation import ABMSimulation
            
            # Find all participant correlation data
            participant_dirs = list(self.correlation_output_dir.glob('participant_*'))
            if not participant_dirs:
                print(f"Error: No participant correlation data found in {self.correlation_output_dir}")
                return False
            
            print(f"Found correlation data for {len(participant_dirs)} participants")
            
            # Initialize ABM simulation
            abm = ABMSimulation(
                num_agents=self.abm_params['num_agents'],
                influence_strength=self.abm_params['influence_strength'],
                random_seed=self.abm_params['random_seed']
            )
            
            # Load correlation data for each participant
            for p_dir in participant_dirs:
                participant_id = p_dir.name.replace('participant_', '')
                print(f"\nProcessing participant {participant_id}...")
                
                # Load correlation matrices and timestamps
                corr_matrices = np.load(p_dir / 'correlation_matrices.npy')
                timestamps = np.load(p_dir / 'timestamps.npy')
                
                # Add participant to ABM
                abm.add_agent(
                    agent_id=participant_id,
                    correlation_matrices=corr_matrices,
                    timestamps=timestamps
                )
            
            # Run the simulation
            print("\nRunning ABM simulation...")
            results = abm.run_simulation(num_steps=self.abm_params['num_steps'])
            
            # Save results
            results_df = pd.DataFrame(results)
            results_path = self.abm_output_dir / 'abm_results.csv'
            results_df.to_csv(results_path, index=False)
            
            print(f"\nABM simulation complete. Results saved to {results_path}")
            
            return True
            
        except Exception as e:
            print(f"Error running ABM simulation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_pipeline(self, force: bool = False) -> bool:
        """Run the entire pipeline."""
        print("=== Starting Synthetic Sentiment Analysis Pipeline ===")
        print(f"Root directory: {self.root_dir}")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Run BERTopic processing
        if not self.run_bertopic():
            print("\nPipeline failed at BERTopic processing step.")
            return False
        
        # Process correlations
        if not self.process_correlations():
            print("\nPipeline failed at correlation processing step.")
            return False
        
        # Run ABM simulation
        if not self.run_abm():
            print("\nPipeline failed at ABM simulation step.")
            return False
        
        print("\n=== Pipeline completed successfully! ===")
        print(f"Results available in: {self.output_dir}")
        return True


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description='Run the Synthetic Sentiment Analysis Pipeline')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_FILE,
                        help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing input CSV files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save all outputs')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if outputs exist')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = PipelineRunner(config_path=args.config)
    
    # Override paths if specified
    if args.data_dir:
        pipeline.data_dir = Path(args.data_dir).resolve()
    if args.output_dir:
        pipeline.output_dir = Path(args.output_dir).resolve()
    
    # Run the pipeline
    success = pipeline.run_pipeline(force=args.force)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
