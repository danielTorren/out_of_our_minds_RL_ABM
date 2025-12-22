import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from typing import Tuple
from datetime import datetime

class PolicyNetworkWithInternal(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        input_size = 3 * M * M
        output_size = M * M
        hidden_size = int(input_size * 0.5)

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.scaling_factor = 0.005
        self.register_buffer("diagonal_mask", torch.ones(M, M))
        for i in range(M):
            self.diagonal_mask[i, i] = 0

    def forward(self, x):
        delta_C_flat = self.net(x)
        delta_C = delta_C_flat.view(-1, self.M, self.M) * self.scaling_factor
        delta_C = delta_C * self.diagonal_mask
        return delta_C

class PolicyNetworkWithoutInternal(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        input_size = 2 * M * M
        output_size = M * M
        hidden_size = int(input_size * 0.5)

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.scaling_factor = 0.005
        self.register_buffer("diagonal_mask", torch.ones(M, M))
        for i in range(M):
            self.diagonal_mask[i, i] = 0

    def forward(self, x):
        delta_C_flat = self.net(x)
        delta_C = delta_C_flat.view(-1, self.M, self.M) * self.scaling_factor
        delta_C = delta_C * self.diagonal_mask
        return delta_C

class FocalAgent:
    def __init__(self, agent_id, M, learning_rate=0.001):
        self.id = agent_id
        self.M = M
        self.C = torch.eye(M)  # Internal correlation matrix
        self.policy = PolicyNetworkWithInternal(M)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.report = None
        self.last_d_int = 0

    def generate_report(self, time_step, internal_noise_std=0.05):
        noise = torch.randn_like(self.C) * internal_noise_std
        self.report = self.C + noise
        self.report = torch.clamp(self.report, -1, 1)  # Keep correlations in [-1, 1]
        return self.report

    def update(self, neighbor_reports, lambda_weight=0.5):
        if not neighbor_reports:
            return 0, 0

        # Calculate external dissonance
        d_ext = 0
        for C_j in neighbor_reports:
            d_ext += torch.norm(self.C - C_j) ** 2
        d_ext = d_ext / len(neighbor_reports)

        # Calculate internal dissonance
        d_int = self.last_d_int  # Use stored internal dissonance

        # Total loss
        loss = (1 - lambda_weight) * d_ext + lambda_weight * d_int

        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update correlation matrix
        with torch.no_grad():
            neighbor_avg = torch.stack(neighbor_reports).mean(dim=0)
            self.C = self.C - self.policy.scaling_factor * (self.C - neighbor_avg)
            # Ensure C remains a valid correlation matrix
            self.C = torch.clamp(self.C, -1, 1)
            for i in range(self.M):
                self.C[i, i] = 1.0

        return d_ext.item(), d_int

class FocalAdjacentAgent(FocalAgent):
    def __init__(self, agent_id, M, learning_rate=0.001, report_interval=3):
        super().__init__(agent_id, M, learning_rate)
        self.report_interval = report_interval
        self.time_since_last_report = 0

    def should_report(self, time_step):
        self.time_since_last_report += 1
        if self.time_since_last_report >= self.report_interval:
            self.time_since_last_report = 0
            return True
        return False

class NonFocalAgent:
    def __init__(self, agent_id, M, learning_rate=0.001):
        self.id = agent_id
        self.M = M
        self.C = torch.eye(M)
        self.policy = PolicyNetworkWithoutInternal(M)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def update(self, neighbor_reports):
        if not neighbor_reports:
            return 0

        # Calculate external dissonance
        d_ext = 0
        for C_j in neighbor_reports:
            d_ext += torch.norm(self.C - C_j) ** 2
        d_ext = d_ext / len(neighbor_reports)

        # Update policy
        self.optimizer.zero_grad()
        d_ext.backward()
        self.optimizer.step()

        # Update correlation matrix
        with torch.no_grad():
            neighbor_avg = torch.stack(neighbor_reports).mean(dim=0)
            self.C = self.C - self.policy.scaling_factor * (self.C - neighbor_avg)
            # Ensure C remains a valid correlation matrix
            self.C = torch.clamp(self.C, -1, 1)
            for i in range(self.M):
                self.C[i, i] = 1.0

        return d_ext.item()

class ABMEnvironment:
    def __init__(self, N=100, M=5, k_neighbors=5, social_noise_std=0.1,
                 internal_noise_std=0.05, lambda_weight=0.5, learning_rate=0.001,
                 focal_percentage=0.1, report_interval=5, pre_generated_reports=None):
        self.N = N
        self.M = M
        self.k_neighbors = k_neighbors
        self.social_noise_std = social_noise_std
        self.internal_noise_std = internal_noise_std
        self.lambda_weight = lambda_weight
        self.learning_rate = learning_rate
        self.focal_percentage = focal_percentage
        self.report_interval = report_interval
        self.pre_generated_reports = pre_generated_reports
        self.time_step = 0
        self.agents = []
        self._initialize_environment()

    def _initialize_environment(self):
        # Create social network
        self.network = nx.watts_strogatz_graph(self.N, self.k_neighbors, 0.3)
        
        # Create agents
        num_focal = int(self.N * self.focal_percentage)
        num_focal_adjacent = int(self.N * 0.3)  # 30% focal-adjacent
        num_non_focal = self.N - num_focal - num_focal_adjacent
        
        self.focal_agents = []
        self.focal_adjacent_agents = []
        self.non_focal_agents = []
        
        # Create focal agents
        for i in range(num_focal):
            agent = FocalAgent(
                i, self.M, 
                learning_rate=self.learning_rate
            )
            self.agents.append(agent)
            self.focal_agents.append(i)
        
        # Create focal-adjacent agents
        for i in range(num_focal, num_focal + num_focal_adjacent):
            agent = FocalAdjacentAgent(
                i, self.M,
                learning_rate=self.learning_rate,
                report_interval=self.report_interval
            )
            self.agents.append(agent)
            self.focal_adjacent_agents.append(i)
        
        # Create non-focal agents
        for i in range(num_focal + num_focal_adjacent, self.N):
            agent = NonFocalAgent(
                i, self.M,
                learning_rate=self.learning_rate
            )
            self.agents.append(agent)
            self.non_focal_agents.append(i)
        
        # Initialize pre-generated reports if provided
        if self.pre_generated_reports is not None:
            self.current_reports = self.pre_generated_reports[0]
        else:
            self.current_reports = [None] * self.N

    def step_simulation(self):
        # Update time step
        self.time_step += 1
        
        # Generate reports
        self._generate_reports()
        
        # Update agents
        metrics = self._update_agents()
        
        # Store metrics
        metrics['time'] = self.time_step
        return metrics
    
    def _generate_reports(self):
        if self.pre_generated_reports is not None:
            if self.time_step < len(self.pre_generated_reports):
                self.current_reports = self.pre_generated_reports[self.time_step]
            return
            
        for i, agent in enumerate(self.agents):
            if i in self.focal_agents or (i in self.focal_adjacent_agents and 
                                         agent.should_report(self.time_step)):
                self.current_reports[i] = agent.generate_report(
                    self.time_step, self.internal_noise_std
                )
            else:
                self.current_reports[i] = None

    def _update_agents(self):
        metrics = {
            'd_ext_avg': 0,
            'd_int_avg': 0,
            'focal_d_ext_avg': 0,
            'focal_adjacent_d_ext_avg': 0,
            'non_focal_d_ext_avg': 0,
            'focal_d_int_avg': 0
        }
        
        focal_d_ext = []
        focal_adjacent_d_ext = []
        non_focal_d_ext = []
        focal_d_int = []
        
        for i, agent in enumerate(self.agents):
            # Get neighbors' reports
            neighbor_indices = list(self.network.neighbors(i))
            neighbor_reports = [self.current_reports[j] for j in neighbor_indices 
                              if self.current_reports[j] is not None]
            
            # Add social noise to neighbor reports
            if neighbor_reports and self.social_noise_std > 0:
                noise = torch.randn_like(torch.stack(neighbor_reports)) * self.social_noise_std
                neighbor_reports = [r + n for r, n in zip(neighbor_reports, noise)]
                neighbor_reports = [torch.clamp(r, -1, 1) for r in neighbor_reports]
            
            # Update agent and get metrics
            if isinstance(agent, FocalAgent):
                d_ext, d_int = agent.update(neighbor_reports, self.lambda_weight)
                focal_d_ext.append(d_ext)
                focal_d_int.append(d_int)
                if i in self.focal_agents:
                    metrics['focal_d_ext_avg'] += d_ext
                else:  # focal-adjacent
                    metrics['focal_adjacent_d_ext_avg'] += d_ext
            else:  # NonFocalAgent
                d_ext = agent.update(neighbor_reports)
                non_focal_d_ext.append(d_ext)
                metrics['non_focal_d_ext_avg'] += d_ext
        
        # Calculate averages
        num_focal = len(self.focal_agents)
        num_focal_adjacent = len(self.focal_adjacent_agents)
        num_non_focal = len(self.non_focal_agents)
        
        if focal_d_ext:
            metrics['focal_d_ext_avg'] /= num_focal if num_focal > 0 else 1
            metrics['focal_adjacent_d_ext_avg'] /= num_focal_adjacent if num_focal_adjacent > 0 else 1
            metrics['focal_d_int_avg'] = sum(focal_d_int) / len(focal_d_int) if focal_d_int else 0
        
        if non_focal_d_ext:
            metrics['non_focal_d_ext_avg'] /= num_non_focal if num_non_focal > 0 else 1
        
        # Calculate overall averages
        total_d_ext = sum(focal_d_ext) + sum(non_focal_d_ext)
        total_agents = len(focal_d_ext) + len(non_focal_d_ext)
        metrics['d_ext_avg'] = total_d_ext / total_agents if total_agents > 0 else 0
        
        if focal_d_int:
            metrics['d_int_avg'] = sum(focal_d_int) / len(focal_d_int)
        
        return metrics

def generate_report_sequence(steps, num_focal, M, internal_noise_std=0.05, seed=42):
    """Generate a sequence of reports for focal agents."""
    if seed is not None:
        torch.manual_seed(seed)
    
    reports = []
    for _ in range(steps):
        # Each focal agent has their own evolving report
        step_reports = [None] * num_focal
        for i in range(num_focal):
            # Create a random positive definite matrix for each agent
            A = torch.randn(M, M)
            C = torch.mm(A, A.t()) / M  # Make it symmetric positive definite
            # Normalize to correlation matrix
            d = torch.diag(1.0 / torch.sqrt(torch.diag(C)))
            C = d @ C @ d
            # Add noise
            noise = torch.randn_like(C) * internal_noise_std
            C = C + noise
            C = torch.clamp(C, -1, 1)
            for i in range(M):
                C[i, i] = 1.0
            step_reports[i] = C
        reports.append(step_reports)
    return reports

def run_abm_simulation(config: dict) -> Tuple[list, dict, dict]:
    """
    Run the ABM simulation with the given configuration.
    
    Args:
        config: Dictionary containing simulation parameters
        
    Returns:
        Tuple of (history, parameters, agent_info)
    """
    # Initialize environment
    env = ABMEnvironment(
        N=config.get('N_AGENTS', 100),
        M=config.get('M_TOPICS', 5),
        k_neighbors=config.get('K_NEIGHBORS', 6),
        social_noise_std=config.get('SOCIAL_NOISE', 0.1),
        internal_noise_std=config.get('INTERNAL_NOISE', 0.05),
        lambda_weight=config.get('LAMBDA_WEIGHT', 0.5),
        learning_rate=config.get('LEARNING_RATE', 0.001),
        focal_percentage=config.get('FOCAL_PERCENTAGE', 0.2),
        report_interval=config.get('REPORT_INTERVAL', 3),
        pre_generated_reports=config.get('pre_generated_reports')
    )
    
    # Run simulation
    history = []
    total_steps = config.get('SIMULATION_STEPS', 40)
    
    for t in range(total_steps):
        metrics = env.step_simulation()
        history.append(metrics)
        if t % 10 == 0:
            print(f"Step {t} complete. Avg D_ext: {metrics['d_ext_avg']:.4f}")
    
    # Prepare output
    parameters = {
        'N_AGENTS': config.get('N_AGENTS', 100),
        'M_TOPICS': config.get('M_TOPICS', 5),
        'K_NEIGHBORS': config.get('K_NEIGHBORS', 6),
        'LAMBDA_WEIGHT': config.get('LAMBDA_WEIGHT', 0.5),
        'LEARNING_RATE': config.get('LEARNING_RATE', 0.001),
        'FOCAL_PERCENTAGE': config.get('FOCAL_PERCENTAGE', 0.2),
        'REPORT_INTERVAL': config.get('REPORT_INTERVAL', 3),
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    agent_info = {
        'focal_agents': np.array(env.focal_agents),
        'focal_adjacent_agents': np.array(env.focal_adjacent_agents),
        'non_focal_agents': np.array(env.non_focal_agents)
    }
    
    return history, parameters, agent_info