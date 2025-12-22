import os
import numpy as np
import yaml
import argparse
from datetime import datetime
from abm_model import run_abm_simulation, generate_report_sequence

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_simulation_results(history, parameters, agent_info, output_dir: str):
    """Save simulation results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"abm_results_{timestamp}"
    
    # Save main results
    results_path = os.path.join(output_dir, f"{base_filename}.npz")
    np.savez_compressed(
        results_path,
        history=history,
        parameters=parameters,
        agent_info=agent_info
    )
    
    # Save configuration
    config_path = os.path.join(output_dir, f"{base_filename}_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(parameters, f)
    
    return results_path

def main():
    parser = argparse.ArgumentParser(description='Run ABM simulation')
    parser.add_argument('--config', type=str, default='config/abm_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs/abm_results',
                       help='Directory to save results')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Generate reports if needed
    if config.get('GENERATE_REPORTS', True):
        num_focal = int(config['N_AGENTS'] * config['FOCAL_PERCENTAGE'])
        pre_generated_reports = generate_report_sequence(
            steps=config['SIMULATION_STEPS'],
            num_focal=num_focal,
            M=config['M_TOPICS'],
            internal_noise_std=config.get('INTERNAL_NOISE', 0.05),
            seed=config.get('SEED', 42)
        )
        config['pre_generated_reports'] = pre_generated_reports
    
    # Run simulation
    print("Starting ABM simulation...")
    history, parameters, agent_info = run_abm_simulation(config)
    
    # Save results
    results_path = save_simulation_results(
        history, parameters, agent_info, args.output_dir
    )
    print(f"Simulation complete. Results saved to: {results_path}")

if __name__ == "__main__":
    main()
