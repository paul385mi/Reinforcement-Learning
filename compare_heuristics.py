import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from collections import deque
from gym_environment import JSPGymEnvironment
from torch_ppo_agent import TorchPPOAgent

# Load JSP data
def load_jsp_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Heuristic implementations
def fifo_heuristic(state, jsp_data):
    """First In First Out - selects the job with the lowest index that can be processed"""
    valid_actions_mask = state['valid_actions_mask']
    for job_idx, is_valid in enumerate(valid_actions_mask):
        if is_valid == 1:
            return job_idx
    return 0  # Fallback

def lifo_heuristic(state, jsp_data):
    """Last In First Out - selects the job with the highest index that can be processed"""
    valid_actions_mask = state['valid_actions_mask']
    for job_idx in range(len(valid_actions_mask) - 1, -1, -1):
        if valid_actions_mask[job_idx] == 1:
            return job_idx
    return 0  # Fallback

def spt_heuristic(state, jsp_data):
    """Shortest Processing Time - selects the job with the shortest next operation"""
    valid_actions_mask = state['valid_actions_mask']
    job_progress = state['job_progress']
    
    min_time = float('inf')
    selected_job = 0
    
    for job_idx, is_valid in enumerate(valid_actions_mask):
        if is_valid == 1:
            op_idx = job_progress[job_idx]
            if op_idx < len(jsp_data["jobs"][job_idx]["operations"]):
                proc_time = jsp_data["jobs"][job_idx]["operations"][op_idx]["processingTime"]
                if proc_time < min_time:
                    min_time = proc_time
                    selected_job = job_idx
    
    return selected_job

def random_heuristic(state, jsp_data):
    """Random selection from valid jobs"""
    valid_actions_mask = state['valid_actions_mask']
    valid_jobs = [job_idx for job_idx, is_valid in enumerate(valid_actions_mask) if is_valid == 1]
    
    if valid_jobs:
        return random.choice(valid_jobs)
    return 0  # Fallback

def run_heuristic(env, heuristic_func, jsp_data, num_episodes=10):
    """Run a heuristic for multiple episodes and return average metrics"""
    makespans = []
    utilizations = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = heuristic_func(state, jsp_data)
            state, reward, done, info = env.step(action)
        
        # Calculate metrics
        makespan = max(env.machine_times)
        
        # Calculate machine utilization
        total_processing_time = sum(env.machine_times)
        total_possible_time = makespan * len(env.machine_times)
        utilization = total_processing_time / total_possible_time if total_possible_time > 0 else 0
        
        makespans.append(makespan)
        utilizations.append(utilization)
    
    return np.mean(makespans), np.mean(utilizations)

def run_ppo_agent(env, model_path, jsp_data, num_episodes=10):
    """Run the PPO agent with a loaded model for multiple episodes"""
    # Load the model state to check dimensions
    model_state = torch.load(model_path)
    
    # Detect the number of jobs the model was trained on
    model_num_jobs = None
    if 'output_layer.weight' in model_state:
        model_num_jobs = model_state['output_layer.weight'].shape[0]
    elif 'output_layer.bias' in model_state:
        model_num_jobs = model_state['output_layer.bias'].shape[0]
    elif 'output_layer' in model_state and 'weight' in model_state['output_layer']:
        model_num_jobs = model_state['output_layer']['weight'].shape[0]
    
    if model_num_jobs is None:
        print("Could not determine the number of jobs in the model. Using current dataset's job count.")
        model_num_jobs = len(jsp_data["jobs"])
    else:
        print(f"Model was trained on {model_num_jobs} jobs, current dataset has {len(jsp_data['jobs'])} jobs")
    
    # Create a copy of the JSP data with only the model's number of jobs for the agent
    # This ensures the agent is initialized correctly with the model's job count
    model_jsp_data = {"jobs": jsp_data["jobs"][:model_num_jobs], "machines": jsp_data["machines"]}
    
    # Initialize agent with the model's job count and JSP data to ensure compatibility
    agent = TorchPPOAgent(model_num_jobs, model_jsp_data)
    
    # Check if we're dealing with a transformer-based model
    is_transformer_model = any(key.startswith('transformer_encoder') for key in model_state.keys())
    
    if is_transformer_model:
        print("Loading transformer-based model...")
        # Extract the embedding dimension from the saved model
        if any(key.startswith('transformer_encoder') for key in model_state.keys()):
            # Find a key that contains layer weights to determine embedding dimension
            for key in model_state.keys():
                if 'norm1.weight' in key:
                    emb_dim = model_state[key].size(0)
                    print(f"Detected embedding dimension: {emb_dim}")
                    agent.embedding_dim = emb_dim
                    break
            
            # Set the number of attention heads (nhead)
            # Try to find the in_proj_weight to determine nhead
            for key in model_state.keys():
                if 'in_proj_weight' in key:
                    in_proj_weight_shape = model_state[key].shape
                    nhead = in_proj_weight_shape[0] // (3 * emb_dim)
                    if nhead == 0:  # Fallback if calculation doesn't work
                        nhead = 4  # Default value
                    agent.nhead = nhead
                    print(f"Using {nhead} attention heads")
                    break
            
            # Recreate the transformer with matching dimensions
            import torch.nn as nn
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=emb_dim, 
                nhead=nhead,
                dim_feedforward=2048,  # Standard size
                dropout=0.1,
                batch_first=False  # Match the saved model
            )
            agent.transformer_layers = 2  # Standard for small models
            agent.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=agent.transformer_layers
            )
            
            # Recreate the output layer with the model's job count
            agent.output_layer = nn.Linear(emb_dim, model_num_jobs)
    else:
        print("Loading graph-based model...")
        # Handle graph-based model
        if any(key.startswith('graph_layer1') for key in model_state.keys()):
            # Find the hidden dimension from graph layers
            for key in model_state.keys():
                if 'graph_layer1' in key and 'weight' in key:
                    hidden_dim = model_state[key].shape[0]
                    agent.hidden_dim = hidden_dim
                    break
            
            # Recreate all layers with correct dimensions
            node_features = 7  # From the original code
            agent.node_embedding = torch.nn.Linear(node_features, agent.embedding_dim)
            agent.graph_layer1 = torch.nn.Linear(agent.embedding_dim, agent.hidden_dim)
            agent.graph_layer2 = torch.nn.Linear(agent.hidden_dim, agent.hidden_dim)
            agent.output_layer = torch.nn.Linear(agent.hidden_dim, model_num_jobs)
    
    # Now load the model with adjusted architecture
    agent.load_model(model_path)
    
    makespans = []
    utilizations = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Custom action selection logic to avoid multinomial sampling issues
            try:
                # Get the state embedding using the agent's state_to_tensor method
                state_embedding = agent.state_to_tensor(state)
                
                # Get action logits directly from the output layer
                logits = agent.output_layer(state_embedding)
                
                # Convert logits to probabilities
                probs = torch.nn.functional.softmax(logits, dim=0)
                
                # Get valid actions from the environment
                valid_actions_mask = state['valid_actions_mask']
                env_num_jobs = len(valid_actions_mask)
                
                # Handle job count mismatch between model and environment
                if model_num_jobs != env_num_jobs:
                    if model_num_jobs < env_num_jobs:
                        # Model has fewer jobs than environment - extend probabilities with zeros
                        extended_probs = torch.zeros(env_num_jobs)
                        extended_probs[:model_num_jobs] = probs
                        probs = extended_probs
                    else:
                        # Model has more jobs than environment - truncate probabilities
                        probs = probs[:env_num_jobs]
                
                # Apply valid actions mask
                valid_jobs = [job_idx for job_idx, is_valid in enumerate(valid_actions_mask) if is_valid == 1]
                
                if not valid_jobs:
                    # No valid actions, use fallback
                    action = 0  # Default fallback
                else:
                    # Create masked probabilities
                    masked_probs = torch.zeros_like(probs)
                    for job_idx in valid_jobs:
                        if job_idx < len(probs):
                            masked_probs[job_idx] = probs[job_idx]
                    
                    # Check if we have any non-zero probabilities
                    if torch.sum(masked_probs) > 0:
                        # Normalize probabilities
                        masked_probs = masked_probs / torch.sum(masked_probs)
                        
                        # Select highest probability action
                        action = torch.argmax(masked_probs).item()
                    else:
                        # All probabilities are zero, use uniform distribution over valid actions
                        action = valid_jobs[0]  # Take first valid action as fallback
            
            except Exception as e:
                print(f"Error in action selection: {e}")
                # Fallback to first valid action
                valid_actions_mask = state['valid_actions_mask']
                for job_idx, is_valid in enumerate(valid_actions_mask):
                    if is_valid == 1:
                        action = job_idx
                        break
                else:
                    action = 0  # Ultimate fallback
            
            # Take the selected action
            state, reward, done, info = env.step(action)
        
        # Calculate metrics
        makespan = max(env.machine_times)
        
        # Calculate machine utilization
        total_processing_time = sum(env.machine_times)
        total_possible_time = makespan * len(env.machine_times)
        utilization = total_processing_time / total_possible_time if total_possible_time > 0 else 0
        
        makespans.append(makespan)
        utilizations.append(utilization)
        
        print(f"Episode {episode+1}/{num_episodes}: Makespan = {makespan:.2f}, Utilization = {utilization:.2f}")
    
    return np.mean(makespans), np.mean(utilizations)

def compare_heuristics(jsp_data_path, model_path, num_episodes=10):
    """Compare different heuristics and the PPO agent"""
    jsp_data = load_jsp_data(jsp_data_path)
    env = JSPGymEnvironment(jsp_data)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run each heuristic
    fifo_makespan, fifo_util = run_heuristic(env, fifo_heuristic, jsp_data, num_episodes)
    lifo_makespan, lifo_util = run_heuristic(env, lifo_heuristic, jsp_data, num_episodes)
    spt_makespan, spt_util = run_heuristic(env, spt_heuristic, jsp_data, num_episodes)
    random_makespan, random_util = run_heuristic(env, random_heuristic, jsp_data, num_episodes)
    
    # Run PPO agent
    ppo_makespan, ppo_util = run_ppo_agent(env, model_path, jsp_data, num_episodes)
    
    # Prepare data for plotting
    heuristics = ['FIFO', 'LIFO', 'SPT', 'RANDOM', 'PPO']
    makespans = [fifo_makespan, lifo_makespan, spt_makespan, random_makespan, ppo_makespan]
    utilizations = [fifo_util, lifo_util, spt_util, random_util, ppo_util]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot makespan (lower is better)
    ax1.bar(heuristics, makespans, color=['blue', 'green', 'orange', 'red', 'purple'])
    ax1.set_title('Makespan Comparison (lower is better)')
    ax1.set_ylabel('Makespan')
    
    # Add values on top of bars
    for i, v in enumerate(makespans):
        ax1.text(i, v + 5, f'{v:.1f}', ha='center')
    
    # Plot utilization (higher is better)
    ax2.bar(heuristics, utilizations, color=['blue', 'green', 'orange', 'red', 'purple'])
    ax2.set_title('Machine Utilization Comparison (higher is better)')
    ax2.set_ylabel('Utilization')
    
    # Add values on top of bars
    for i, v in enumerate(utilizations):
        ax2.text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('heuristic_comparison.png')
    plt.show()
    
    # Print results
    print("\nResults Summary:")
    print("-" * 50)
    print(f"{'Heuristic':<10} {'Makespan':<15} {'Utilization':<15}")
    print("-" * 50)
    for h, m, u in zip(heuristics, makespans, utilizations):
        print(f"{h:<10} {m:<15.2f} {u:<15.2f}")

if __name__ == "__main__":
    # Use relative paths for current directory
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    jsp_data_path = os.path.join(current_dir, "data.json")
    
    # Check if results/models directory exists and has model files
    models_dir = os.path.join(current_dir, "results", "models")
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        if model_files:
            # Use the most recent model file
            model_files.sort(reverse=True)  # Sort by name (which includes timestamp)
            model_path = os.path.join(models_dir, model_files[0])
            print(f"Using model: {model_files[0]}")
        else:
            print("No model files found in results/models directory.")
            print("Please train a model first or specify a model path.")
            exit(1)
    else:
        print("results/models directory not found.")
        print("Please train a model first or create the directory structure.")
        exit(1)
    
    # Check if data.json exists
    if not os.path.exists(jsp_data_path):
        print(f"Data file not found: {jsp_data_path}")
        print("Please generate data using data_generator.py first.")
        exit(1)
    
    # Run comparison with 10 episodes per heuristic
    compare_heuristics(jsp_data_path, model_path, num_episodes=10)