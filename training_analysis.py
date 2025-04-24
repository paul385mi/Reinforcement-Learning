import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict

class TrainingAnalyzer:
    def __init__(self, training_data_dir):
        self.training_data_dir = training_data_dir
        self.episode_summaries = []
        self.episode_steps = {}
        self.episode_operations = {}
        self.episode_utilization = {}
        self.load_data()
    
    def load_data(self):
        """Load all training data from the specified directory."""
        print(f"Loading training data from {self.training_data_dir}")
        
        # Load episode summaries
        for filename in os.listdir(self.training_data_dir):
            if filename.endswith("_summary.json"):
                episode_id = int(filename.split("_")[1])
                with open(os.path.join(self.training_data_dir, filename), 'r') as f:
                    summary = json.load(f)
                    self.episode_summaries.append(summary)
        
        # Sort summaries by episode ID
        self.episode_summaries.sort(key=lambda x: x['episode_id'])
        
        # Load step data for each episode
        for summary in self.episode_summaries:
            episode_id = summary['episode_id']
            steps_filename = f"episode_{episode_id:04d}_steps.json"
            steps_filepath = os.path.join(self.training_data_dir, steps_filename)
            if os.path.exists(steps_filepath):
                with open(steps_filepath, 'r') as f:
                    self.episode_steps[episode_id] = json.load(f)
            
            operations_filename = f"episode_{episode_id:04d}_operations.json"
            operations_filepath = os.path.join(self.training_data_dir, operations_filename)
            if os.path.exists(operations_filepath):
                with open(operations_filepath, 'r') as f:
                    self.episode_operations[episode_id] = json.load(f)
            
            utilization_filename = f"episode_{episode_id:04d}_utilization.json"
            utilization_filepath = os.path.join(self.training_data_dir, utilization_filename)
            if os.path.exists(utilization_filepath):
                with open(utilization_filepath, 'r') as f:
                    self.episode_utilization[episode_id] = json.load(f)
        
        print(f"Loaded data for {len(self.episode_summaries)} episodes")
    
    def plot_training_progress(self):
        """Plot overall training progress metrics."""
        if not self.episode_summaries:
            print("No episode data available")
            return
        
        # Extract metrics
        episodes = [summary['episode_id'] for summary in self.episode_summaries]
        makespans = [summary['makespan'] for summary in self.episode_summaries]
        completed_jobs = [summary['completed_jobs'] for summary in self.episode_summaries]
        met_deadlines = [summary.get('met_deadlines', 0) for summary in self.episode_summaries]
        total_rewards = [summary.get('total_reward', 0) for summary in self.episode_summaries]
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Plot makespan
        axs[0, 0].plot(episodes, makespans, 'b-')
        axs[0, 0].set_title('Makespan')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Time')
        axs[0, 0].grid(True)
        
        # Plot completed jobs
        axs[0, 1].plot(episodes, completed_jobs, 'g-')
        axs[0, 1].set_title('Completed Jobs')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Count')
        axs[0, 1].grid(True)
        
        # Plot met deadlines
        axs[1, 0].plot(episodes, met_deadlines, 'r-')
        axs[1, 0].set_title('Met Deadlines')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Count')
        axs[1, 0].grid(True)
        
        # Plot total reward
        axs[1, 1].plot(episodes, total_rewards, 'purple')
        axs[1, 1].set_title('Total Reward')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Reward')
        axs[1, 1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.training_data_dir, 'training_progress.png'))
        plt.close()
        
        print(f"Training progress plot saved to {os.path.join(self.training_data_dir, 'training_progress.png')}")
    
    def plot_reward_components(self):
        """Plot the evolution of reward components over episodes."""
        if not self.episode_summaries:
            print("No episode data available")
            return
        
        # Extract reward components
        episodes = [summary['episode_id'] for summary in self.episode_summaries]
        reward_components = defaultdict(list)
        
        for summary in self.episode_summaries:
            if 'cumulative_reward' in summary:
                for component, value in summary['cumulative_reward'].items():
                    reward_components[component].append(value)
            else:
                # Handle missing data
                for component in reward_components:
                    if len(reward_components[component]) < len(episodes):
                        reward_components[component].append(0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        fig.suptitle('Reward Components Evolution', fontsize=16)
        
        # Plot each component
        for component, values in reward_components.items():
            if len(values) == len(episodes):  # Ensure data alignment
                ax.plot(episodes, values, label=component)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.training_data_dir, 'reward_components.png'))
        plt.close()
        
        print(f"Reward components plot saved to {os.path.join(self.training_data_dir, 'reward_components.png')}")
    
    def analyze_episode(self, episode_id):
        """Analyze a specific episode in detail."""
        if episode_id not in [summary['episode_id'] for summary in self.episode_summaries]:
            print(f"Episode {episode_id} not found")
            return
        
        # Get episode data
        summary = next(s for s in self.episode_summaries if s['episode_id'] == episode_id)
        steps = self.episode_steps.get(episode_id, [])
        operations = self.episode_operations.get(episode_id, [])
        
        print(f"\nAnalysis of Episode {episode_id}:")
        print(f"  Total steps: {summary['total_steps']}")
        print(f"  Makespan: {summary['makespan']:.2f}")
        print(f"  Completed jobs: {summary['completed_jobs']}/{summary['total_jobs']}")
        print(f"  Met deadlines: {summary.get('met_deadlines', 0)}")
        print(f"  Total reward: {summary.get('total_reward', 0):.2f}")
        
        # Analyze reward components
        if 'cumulative_reward' in summary:
            print("\nReward Components:")
            for component, value in summary['cumulative_reward'].items():
                print(f"  {component}: {value:.2f}")
        
        # Analyze job completion
        if 'job_completion_times' in summary:
            print("\nJob Completion:")
            for job_id, data in summary['job_completion_times'].items():
                deadline_status = "Met" if data.get('deadline_met', False) else "Missed"
                print(f"  Job {job_id} (Priority {data.get('priority', 'N/A')}): Completed at {data.get('completion_time', 0):.2f}, Deadline: {data.get('deadline', 0):.2f} ({deadline_status})")
        
        # Plot reward per step
        if steps:
            step_numbers = [step.get('step', i) for i, step in enumerate(steps)]
            rewards = [step.get('reward', 0) for step in steps]
            
            plt.figure(figsize=(12, 6))
            plt.plot(step_numbers, rewards, 'b-')
            plt.title(f'Rewards per Step - Episode {episode_id}')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.savefig(os.path.join(self.training_data_dir, f'episode_{episode_id}_rewards.png'))
            plt.close()
            
            print(f"\nReward per step plot saved to {os.path.join(self.training_data_dir, f'episode_{episode_id}_rewards.png')}")
        
        # Create Gantt chart for operations
        if operations:
            self._create_gantt_chart(operations, episode_id)
    
    def _create_gantt_chart(self, operations, episode_id):
        """Create a Gantt chart for the operations in an episode."""
        # Sort operations by machine and start time
        operations.sort(key=lambda x: (x.get('machine_idx', 0), x.get('start_time', 0)))
        
        # Prepare data for Gantt chart
        df = pd.DataFrame(operations)
        df['duration'] = df['end_time'] - df['start_time']
        df['job_operation'] = df.apply(lambda row: f"J{row['job_idx']}-O{row['operation_idx']}", axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        fig.suptitle(f'Operation Schedule - Episode {episode_id}', fontsize=16)
        
        # Plot each operation as a horizontal bar
        machines = sorted(df['machine_idx'].unique())
        colors = plt.cm.tab10.colors
        
        for i, machine in enumerate(machines):
            machine_ops = df[df['machine_idx'] == machine]
            for _, op in machine_ops.iterrows():
                color_idx = op['job_idx'] % len(colors)
                ax.barh(y=machine, width=op['duration'], left=op['start_time'], 
                        color=colors[color_idx], alpha=0.8, 
                        label=f"Job {op['job_idx']}" if i == 0 else "")
                
                # Add text label
                ax.text(op['start_time'] + op['duration']/2, machine, op['job_operation'], 
                        ha='center', va='center', color='black', fontsize=8)
        
        # Set labels and ticks
        ax.set_yticks(machines)
        ax.set_yticklabels([f"Machine {m}" for m in machines])
        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')
        ax.grid(True, axis='x')
        
        # Add legend for jobs
        handles, labels = [], []
        for job_idx in sorted(df['job_idx'].unique()):
            color_idx = job_idx % len(colors)
            handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[color_idx], alpha=0.8))
            labels.append(f"Job {job_idx}")
        
        ax.legend(handles, labels, loc='upper right')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.training_data_dir, f'episode_{episode_id}_gantt.png'))
        plt.close()
        
        print(f"Gantt chart saved to {os.path.join(self.training_data_dir, f'episode_{episode_id}_gantt.png')}")
    
    def analyze_action_selection(self, episode_id):
        """Analyze the action selection process for a specific episode."""
        if episode_id not in self.episode_steps:
            print(f"Step data for episode {episode_id} not found")
            return
        
        steps = self.episode_steps[episode_id]
        
        # Filter steps with action probabilities
        steps_with_probs = [step for step in steps if step.get('action_probs') is not None]
        
        if not steps_with_probs:
            print(f"No action probability data available for episode {episode_id}")
            return
        
        # Create figure
        fig, axs = plt.subplots(len(steps_with_probs), 1, figsize=(15, 3*len(steps_with_probs)))
        fig.suptitle(f'Action Selection Analysis - Episode {episode_id}', fontsize=16)
        
        # Plot action probabilities for each step
        for i, step in enumerate(steps_with_probs):
            ax = axs[i] if len(steps_with_probs) > 1 else axs
            
            # Get action probabilities and valid actions
            probs = step['action_probs']
            valid_actions = step.get('valid_actions', [])
            selected_action = step['action']
            
            # Create bar chart
            bars = ax.bar(range(len(probs)), probs, alpha=0.7)
            
            # Highlight valid actions and selected action
            for j, bar in enumerate(bars):
                if j in valid_actions:
                    bar.set_color('blue')
                if j == selected_action:
                    bar.set_color('green')
            
            ax.set_title(f"Step {step['step']}")
            ax.set_xlabel('Action (Job Index)')
            ax.set_ylabel('Probability')
            ax.set_xticks(range(len(probs)))
            ax.grid(True, axis='y')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.training_data_dir, f'episode_{episode_id}_action_selection.png'))
        plt.close()
        
        print(f"Action selection analysis saved to {os.path.join(self.training_data_dir, f'episode_{episode_id}_action_selection.png')}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze training data')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
    parser.add_argument('--episode', type=int, help='Specific episode to analyze')
    
    args = parser.parse_args()
    
    analyzer = TrainingAnalyzer(args.data_dir)
    analyzer.plot_training_progress()
    analyzer.plot_reward_components()
    
    if args.episode is not None:
        analyzer.analyze_episode(args.episode)
        analyzer.analyze_action_selection(args.episode)
    else:
        # Analyze the last episode by default
        if analyzer.episode_summaries:
            last_episode = analyzer.episode_summaries[-1]['episode_id']
            analyzer.analyze_episode(last_episode)
            analyzer.analyze_action_selection(last_episode)