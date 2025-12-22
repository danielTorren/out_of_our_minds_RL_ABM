import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Any

def load_simulation_results(filepath: str) -> tuple:
    """Load simulation results from file."""
    data = np.load(filepath, allow_pickle=True)
    history = data['history'].item() if 'history' in data else []
    parameters = data['parameters'].item() if 'parameters' in data else {}
    agent_info = data['agent_info'].item() if 'agent_info' in data else {}
    return history, parameters, agent_info

def plot_dissonance_evolution(history: List[Dict], parameters: Dict, output_dir: str = None):
    """Plot the evolution of dissonance metrics over time."""
    timesteps = [h['time'] for h in history]
    
    plt.figure(figsize=(12, 6))
    
    # Plot external dissonance
    d_ext = [h['d_ext_avg'] for h in history]
    plt.plot(timesteps, d_ext, label='External Dissonance', color='blue')
    
    # Plot internal dissonance if available
    if 'd_int_avg' in history[0]:
        d_int = [h['d_int_avg'] for h in history]
        plt.plot(timesteps, d_int, label='Internal Dissonance', color='red')
    
    plt.title('Average Dissonance Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Dissonance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'dissonance_evolution.png'))
    else:
        plt.show()
    plt.close()

def plot_agent_type_comparison(history: List[Dict], agent_info: Dict, output_dir: str = None):
    """Compare metrics across different agent types."""
    if not agent_info:
        return
        
    timesteps = [h['time'] for h in history]
    
    # Plot external dissonance by agent type
    plt.figure(figsize=(12, 6))
    
    if 'focal_agents' in agent_info and 'focal_d_ext_avg' in history[0]:
        d_ext_focal = [h['focal_d_ext_avg'] for h in history]
        plt.plot(timesteps, d_ext_focal, label='Focal Agents', color='green')
    
    if 'focal_adjacent_agents' in agent_info and 'focal_adjacent_d_ext_avg' in history[0]:
        d_ext_adj = [h['focal_adjacent_d_ext_avg'] for h in history]
        plt.plot(timesteps, d_ext_adj, label='Focal-Adjacent', color='orange')
    
    if 'non_focal_agents' in agent_info and 'non_focal_d_ext_avg' in history[0]:
        d_ext_non = [h['non_focal_d_ext_avg'] for h in history]
        plt.plot(timesteps, d_ext_non, label='Non-Focal', color='purple')
    
    plt.title('External Dissonance by Agent Type')
    plt.xlabel('Time Step')
    plt.ylabel('Average External Dissonance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'agent_type_comparison.png'))
    else:
        plt.show()
    plt.close()

def plot_correlation_matrices(history: List[Dict], output_dir: str = None, num_matrices: int = 3):
    """Plot example correlation matrices at different time points."""
    if not history or 'example_matrices' not in history[0]:
        return
        
    # Select time points to plot
    total_steps = len(history)
    step_indices = np.linspace(0, total_steps-1, num=min(num_matrices, total_steps), dtype=int)
    
    for i, step in enumerate(step_indices):
        matrices = history[step].get('example_matrices', [])
        if not matrices:
            continue
            
        # Plot the first agent's matrix as an example
        plt.figure(figsize=(6, 5))
        plt.imshow(matrices[0], cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.title(f'Correlation Matrix (Step {step})')
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'correlation_matrix_step{step}.png'))
        else:
            plt.show()
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot ABM simulation results')
    parser.add_argument('input_file', type=str, help='Path to simulation results file')
    parser.add_argument('--output-dir', type=str, default='outputs/plots',
                       help='Directory to save plots')
    args = parser.parse_args()
    
    # Load data
    history, parameters, agent_info = load_simulation_results(args.input_file)
    
    # Create plots
    plot_dissonance_evolution(history, parameters, args.output_dir)
    plot_agent_type_comparison(history, agent_info, args.output_dir)
    plot_correlation_matrices(history, args.output_dir)
    
    print(f"Plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
