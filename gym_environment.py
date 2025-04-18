import gym
from gym import spaces
import numpy as np
import logging
from datetime import datetime


class JSPGymEnvironment(gym.Env):
    """
    A Gym-compatible environment for Job-Shop Scheduling Problems.
    
    This environment follows the OpenAIz Gym interface, making it compatible with
    standard RL algorithms and frameworks.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, jsp_data, enable_logging=False, log_level=logging.INFO):
        """
        Initialize the JSP Gym environment.
        
        Args:
            jsp_data: Dictionary containing JSP problem data
            enable_logging: Whether to enable detailed logging
            log_level: Logging level (default: INFO)
        """
        super(JSPGymEnvironment, self).__init__()
        
        self.jobs = jsp_data["jobs"]
        self.machines = jsp_data["machines"]
        self.setupTimes = jsp_data["setupTimes"]
        self.num_jobs = len(self.jobs)
        self.num_machines = len(self.machines)
        
        # Setup logging
        self.enable_logging = enable_logging
        if enable_logging:
            self._setup_logging(log_level)
        
        # Create mappings between IDs and indices
        self.job_id_to_idx = {job["id"]: idx for idx, job in enumerate(self.jobs)}
        self.idx_to_job_id = {idx: job["id"] for idx, job in enumerate(self.jobs)}
        self.machine_id_to_idx = {machine["id"]: idx for idx, machine in enumerate(self.machines)}
        self.idx_to_machine_id = {idx: machine["id"] for idx, machine in enumerate(self.machines)}
        
        # Define action and observation spaces
        # Action space: Choose one of the jobs
        self.action_space = spaces.Discrete(self.num_jobs)
        
        # Observation space: Dictionary with job progress, machine times, etc.
        self.observation_space = spaces.Dict({
            'job_progress': spaces.Box(
                low=0, 
                high=max([len(job["operations"]) for job in self.jobs]),
                shape=(self.num_jobs,), 
                dtype=np.int32
            ),
            'machine_times': spaces.Box(
                low=0, 
                high=float('inf'),  # No upper bound on machine times
                shape=(self.num_machines,), 
                dtype=np.float32
            ),
            'current_time': spaces.Box(
                low=0, 
                high=float('inf'),
                shape=(1,), 
                dtype=np.float32
            ),
            'job_priorities': spaces.Box(
                low=0, 
                high=10,  # Assuming priorities are between 0-10
                shape=(self.num_jobs,), 
                dtype=np.float32
            ),
            'job_deadlines': spaces.Box(
                low=0, 
                high=float('inf'),
                shape=(self.num_jobs,), 
                dtype=np.float32
            ),
            'machine_materials': spaces.Box(
                low=0, 
                high=len(set([op["material"] for job in self.jobs for op in job["operations"]])),
                shape=(self.num_machines,), 
                dtype=np.int32
            ),
            'valid_actions_mask': spaces.Box(
                low=0, 
                high=1,
                shape=(self.num_jobs,), 
                dtype=np.int32
            )
        })
        
        # Initialize state
        self.reset()
        
        # Create a mapping for materials to indices for observation space
        self.materials = list(set([op["material"] for job in self.jobs for op in job["operations"]]))
        self.material_to_idx = {material: idx for idx, material in enumerate(self.materials)}
        self.idx_to_material = {idx: material for idx, material in enumerate(self.materials)}
        
        # Initialize operation tracking for detailed logging
        self.operation_history = []
        self.machine_utilization = [[] for _ in range(self.num_machines)]
        self.material_changes = []
        self.job_completion_times = {}
    
    def _setup_logging(self, log_level):
        """
        Setup logging for the environment.
        
        Args:
            log_level: Logging level
        """
        self.logger = logging.getLogger('JSPGymEnvironment')
        self.logger.setLevel(log_level)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fh = logging.FileHandler(f'jsp_env_{timestamp}.log')
        fh.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    def reset(self):
        """
        Reset the environment to its initial state.
        
        Returns:
            observation: The initial observation
        """
        
        if hasattr(self, 'machine_times') and len(self.machine_times) > 0:
            self.previous_episode_makespan = max(self.machine_times)
        else:
            self.previous_episode_makespan = 0
    
        # Bestehender Reset-Code hier...
        self.job_progress = [0] * self.num_jobs
        self.machine_times = [0] * self.num_machines
        self.current_time = 0
        self.completed_jobs = 0
        self.current_machine_material = [""] * self.num_machines
        self.machine_material_idx = [0] * self.num_machines
        self.episode_steps = 0
        self.episode_reward = 0
        self.episode_makespan = 0
        self.episode_completed_jobs = 0
        self.episode_met_deadlines = 0
        self.operation_history = []
        self.machine_utilization = [[] for _ in range(self.num_machines)]
        self.material_changes = []
        self.job_completion_times = {}
        self.action_history = []
        self.pending_penalties = []
        # Initialize cumulative reward components
        self.cumulative_reward_components = {
            'makespan_reward': 0.0,
            'setup_reward': 0.0,
            'idle_penalty': 0.0,
            'deadline_reward': 0.0,
            'priority_reward': 0.0,
            'critical_job_reward': 0.0,
            'global_progress_reward': 0.0,
            'placement_reward': 0.0,
            'lookahead_reward': 0.0
        }
        
        # Reset tracking variables
        self.operation_history = []
        self.machine_utilization = [[] for _ in range(self.num_machines)]
        self.material_changes = []
        self.job_completion_times = {}
        
        # Retain previous insights if available
        if not hasattr(self, 'placement_insights'):
            self.placement_insights = {}
        
        if self.enable_logging:
            self.logger.info("Environment reset")
        
        return self._get_observation()
    
    def _get_observation(self):
        """
        Get the current observation of the environment.
        
        Returns:
            observation: Dictionary containing the current state
        """
        job_priorities = np.array([job["priority"] for job in self.jobs], dtype=np.float32)
        job_deadlines = np.array([job["deadline"] for job in self.jobs], dtype=np.float32)
        valid_actions_mask = np.zeros(self.num_jobs, dtype=np.int32)
        for job_idx in range(self.num_jobs):
            if (self.job_progress[job_idx] < len(self.jobs[job_idx]["operations"]) and
                self._check_predecessors(job_idx, self.job_progress[job_idx])):
                valid_actions_mask[job_idx] = 1
        
        return {
            'job_progress': self.job_progress.copy(),
            'machine_times': self.machine_times.copy(),
            'current_time': np.array([self.current_time], dtype=np.float32),
            'job_priorities': job_priorities,
            'job_deadlines': job_deadlines,
            'machine_materials': self.machine_material_idx.copy(),
            'valid_actions_mask': valid_actions_mask
        }
    
    def _check_predecessors(self, job_idx, op_idx):
        """
        Check if all predecessor operations are completed.
        
        Args:
            job_idx: Index of the job
            op_idx: Index of the operation
            
        Returns:
            bool: True if all predecessors are completed, False otherwise
        """
        if op_idx >= len(self.jobs[job_idx]["operations"]):
            return False
        operation = self.jobs[job_idx]["operations"][op_idx]
        if not operation.get("predecessors", []):
            return True
        for pred in operation["predecessors"]:
            pred_job_id, pred_op_id = pred.split(":")
            pred_job_idx = self.job_id_to_idx[pred_job_id]
            pred_op_idx = None
            for i, op in enumerate(self.jobs[pred_job_idx]["operations"]):
                if op["id"] == pred_op_id:
                    pred_op_idx = i
                    break
            if pred_op_idx is None or self.job_progress[pred_job_idx] <= pred_op_idx:
                return False
        return True
    
    def _calculate_setup_time(self, machine_id, new_material):
        """
        Calculate setup time based on current and new material.
        
        Args:
            machine_id: ID of the machine
            new_material: New material to be processed
            
        Returns:
            float: Setup time
        """
        machine_idx = self.machine_id_to_idx[machine_id]
        current_material = self.current_machine_material[machine_idx]
        if current_material == "":
            return 0
        if current_material == new_material:
            return self.setupTimes[machine_id]["standard"]
        else:
            return self.setupTimes[machine_id]["materialChange"]
    
    def step(self, action, model=None):
        """
        Execute one step in the environment.
        
        Args:
            action: Index of the job to process next
            model: Optional vortrainiertes Modell für Lookahead-Bewertung
            
        Returns:
            observation: New observation after taking the action
            reward: Reward received
            done: Whether the episode is finished
            info: Additional information
        """
        # Remove the debug statement
        self.episode_steps += 1
        
        # Rest of the method remains unchanged
        if action >= self.num_jobs:
            return self._get_observation(), -10.0, False, {"error": "Invalid job index"}
        job_idx = action
        if self.job_progress[job_idx] >= len(self.jobs[job_idx]["operations"]):
            return self._get_observation(), -10.0, False, {"error": "Job already completed"}
        
        # Get the next operation for the job
        op_idx = self.job_progress[job_idx]
        op = self.jobs[job_idx]["operations"][op_idx]
        machine_id = op["machineId"]
        machine_idx = self.machine_id_to_idx[machine_id]
        proc_time = op["processingTime"]
        material = op["material"]
        
        if not self._check_predecessors(job_idx, op_idx):
            return self._get_observation(), -10.0, False, {"error": "Predecessors not completed"}
        
        setup_time = self._calculate_setup_time(machine_id, material)
        start_time = max(self.machine_times[machine_idx], self.current_time) + setup_time
        end_time = start_time + proc_time
        self.machine_times[machine_idx] = end_time
        
        old_material = self.current_machine_material[machine_idx]
        self.current_machine_material[machine_idx] = material
        self.machine_material_idx[machine_idx] = self.material_to_idx.get(material, 0)
        
        # Log material change if it occurred
        if old_material != material:
            material_change = {
                'step': self.episode_steps,
                'machine_id': machine_id,
                'machine_idx': machine_idx,
                'old_material': old_material,
                'new_material': material,
                'setup_time': setup_time,
                'time': self.current_time
            }
            self.material_changes.append(material_change)
            if self.enable_logging:
                self.logger.info(f"Material change on machine {machine_id}: {old_material} -> {material}, setup time: {setup_time}")
        
        self.job_progress[job_idx] += 1
        
        operation_record = {
            'step': self.episode_steps,
            'job_id': self.idx_to_job_id[job_idx],
            'job_idx': job_idx,
            'operation_idx': op_idx,
            'machine_id': machine_id,
            'machine_idx': machine_idx,
            'start_time': start_time,
            'end_time': end_time,
            'processing_time': proc_time,
            'setup_time': setup_time,
            'material': material
        }
        self.operation_history.append(operation_record)
        
        machine_util = {
            'step': self.episode_steps,
            'machine_id': machine_id,
            'start_time': start_time,
            'end_time': end_time,
            'busy_time': proc_time,
            'setup_time': setup_time,
            'idle_time': start_time - max(self.current_time, self.machine_times[machine_idx]) if start_time > max(self.current_time, self.machine_times[machine_idx]) else 0
        }
        self.machine_utilization[machine_idx].append(machine_util)
        
        if self.enable_logging:
            self.logger.info(f"Executed operation: Job {self.idx_to_job_id[job_idx]}, Op {op_idx}, Machine {machine_id}, Start: {start_time}, End: {end_time}, Setup: {setup_time}")
        
        prev_time = self.current_time
        self.current_time = max(self.machine_times)
        
        job_completed = False
        if self.job_progress[job_idx] >= len(self.jobs[job_idx]["operations"]):
            job_completed = True
            self.completed_jobs += 1
            job_id = self.idx_to_job_id[job_idx]
            deadline = self.jobs[job_idx]["deadline"]
            deadline_met = self.current_time <= deadline
            self.job_completion_times[job_id] = {
                'completion_time': self.current_time,
                'deadline': deadline,
                'deadline_met': deadline_met,
                'priority': self.jobs[job_idx]["priority"]
            }
            if deadline_met:
                self.episode_met_deadlines += 1
            if self.enable_logging:
                self.logger.info(f"Job {job_id} completed at time {self.current_time}, deadline: {deadline}, met: {deadline_met}, priority: {self.jobs[job_idx]['priority']}")
        
        reward = self._calculate_reward(job_idx, job_completed, setup_time, prev_time, self.current_time, model)
        observation = self._get_observation()
        done = self.completed_jobs >= self.num_jobs
        
        info = {
            "makespan": max(self.machine_times),
            "completed_jobs": self.completed_jobs,
            "met_deadlines": self.episode_met_deadlines
        }
        
        if done:
            critical_path = self.analyze_critical_path()
            insights = self.identify_suboptimal_placements(critical_path)
            self.update_reward_function(insights)
            info["critical_path_length"] = len(critical_path)
            info["suboptimal_placements"] = len(insights)
            # Füge Reward-Statistiken hinzu
            info["reward_stats"] = self.get_reward_stats()
            if self.enable_logging:
                self.logger.info(f"Episode completed. Critical path: {len(critical_path)} operations, Suboptimal placements: {len(insights)}")
                self.logger.info(f"Reward components: {self.reward_components}")
        
        return observation, reward, done, info

    def analyze_critical_path(self):
        """
        Analyzes the critical path and identifies suboptimal placements in the schedule.
        """
        if not self.operation_history:
            return []
        
        makespan = max(op['end_time'] for op in self.operation_history)
        critical_ops = [op for op in self.operation_history if abs(op['end_time'] - makespan) < 0.001]
        if len(critical_ops) > 1:
            critical_ops.sort(key=lambda op: op['processing_time'] + op['setup_time'], reverse=True)
        critical_path = [critical_ops[0]]
        current_op = critical_ops[0]
        
        while current_op['start_time'] > 0.001:
            prev_machine_ops = [op for op in self.operation_history 
                                if op['machine_idx'] == current_op['machine_idx'] and 
                                abs(op['end_time'] - current_op['start_time']) < 0.001]
            prev_job_ops = []
            if current_op['operation_idx'] > 0:
                prev_job_ops = [op for op in self.operation_history 
                                if op['job_idx'] == current_op['job_idx'] and 
                                op['operation_idx'] == current_op['operation_idx'] - 1]
            prev_ops = prev_machine_ops + prev_job_ops
            if not prev_ops:
                break
            prev_ops.sort(key=lambda op: op['end_time'], reverse=True)
            current_op = prev_ops[0]
            critical_path.append(current_op)
        critical_path.sort(key=lambda op: op['start_time'])
        return critical_path
    
    def identify_suboptimal_placements(self, critical_path):
        """
        Identifies suboptimal placements in the schedule based on the critical path.
        """
        insights = []
        for i, op in enumerate(critical_path):
            if op['setup_time'] > self.setupTimes[op['machine_id']]['standard']:
                insights.append({
                    'type': 'material_change_on_critical_path',
                    'job_idx': op['job_idx'],
                    'operation_idx': op['operation_idx'],
                    'machine_idx': op['machine_idx'],
                    'time': op['start_time'],
                    'severity': op['setup_time'] / self.setupTimes[op['machine_id']]['materialChange'],
                    'message': f"Material change on critical path at Job {op['job_idx']} Operation {op['operation_idx']} on Machine {op['machine_id']}"
                })
            if i > 0:
                prev_op = critical_path[i-1]
                if op['machine_idx'] == prev_op['machine_idx'] and op['start_time'] > prev_op['end_time'] + 0.001:
                    idle_time = op['start_time'] - prev_op['end_time']
                    insights.append({
                        'type': 'idle_time_on_critical_path',
                        'job_idx': op['job_idx'],
                        'operation_idx': op['operation_idx'],
                        'machine_idx': op['machine_idx'],
                        'time': prev_op['end_time'],
                        'idle_time': idle_time,
                        'severity': idle_time / op['processing_time'],
                        'message': f"Idle time on critical path before Job {op['job_idx']} Operation {op['operation_idx']} on Machine {op['machine_id']}"
                    })
            job_priority = self.jobs[op['job_idx']]['priority']
            if job_priority < 5:
                insights.append({
                    'type': 'low_priority_on_critical_path',
                    'job_idx': op['job_idx'],
                    'operation_idx': op['operation_idx'],
                    'machine_idx': op['machine_idx'],
                    'time': op['start_time'],
                    'priority': job_priority,
                    'severity': (5 - job_priority) / 5,
                    'message': f"Low priority ({job_priority}) on critical path: Job {op['job_idx']} Operation {op['operation_idx']}"
                })
        non_critical_ops = [op for op in self.operation_history if op not in critical_path]
        for op in non_critical_ops:
            job_priority = self.jobs[op['job_idx']]['priority']
            if job_priority > 7:
                insights.append({
                    'type': 'high_priority_not_on_critical_path',
                    'job_idx': op['job_idx'],
                    'operation_idx': op['operation_idx'],
                    'machine_idx': op['machine_idx'],
                    'time': op['start_time'],
                    'priority': job_priority,
                    'severity': (job_priority - 7) / 3,
                    'message': f"High priority ({job_priority}) not on critical path: Job {op['job_idx']} Operation {op['operation_idx']}"
                })
        return insights
    
    def update_reward_function(self, insights):
        """
        Updates the reward function based on insights from the critical path analysis.
        """
        self.placement_insights = {}
        for insight in insights:
            key = (insight['job_idx'], insight['operation_idx'], insight['machine_idx'])
            if key not in self.placement_insights:
                self.placement_insights[key] = []
            self.placement_insights[key].append(insight)
        if self.enable_logging and insights:
            self.logger.info(f"Reward function updated with {len(insights)} insights")
            for insight in insights:
                self.logger.info(f"  {insight['message']} (Severity: {insight['severity']:.2f})")
    
    def _calculate_reward(self, job_idx, job_completed, setup_time, prev_time, current_time, model):
        """
        Calculate the reward for the current action.
        """
        # Initialize reward components dictionary if it doesn't exist
        if not hasattr(self, 'reward_components'):
            self.reward_components = {
                'makespan_reward': 0.0,
                'setup_reward': 0.0,
                'idle_penalty': 0.0,
                'deadline_reward': 0.0,
                'priority_reward': 0.0,
                'critical_job_reward': 0.0,
                'global_progress_reward': 0.0,
                'placement_reward': 0.0,
                'lookahead_reward': 0.0,
                'timeliness_reward': 0.0,  # TIMELINESS: Neue Komponente für die Pünktlichkeit
                'machine_idle_penalty': 0.0,  # MASCHINENSTILLSTAND: Neue Komponente für Maschinenstillstand
                'credit_assignment_penalty': 0.0  # CREDIT_ASSIGNMENT_PROBLEM: Neue Komponente für Credit Assignment
            }
            
        # TIMELINES: Initialize cumulative reward components dictionary if it doesn't exist
        if not hasattr(self, 'cumulative_reward_components'):
            self.cumulative_reward_components = {key: 0.0 for key in self.reward_components}
        # TIMELINES: Make sure any new keys in reward_components are also in cumulative_reward_components
        elif any(key not in self.cumulative_reward_components for key in self.reward_components):
            for key in self.reward_components:
                if key not in self.cumulative_reward_components:
                    self.cumulative_reward_components[key] = 0.0
        
        # CREDIT_ASSIGNMENT_PROBLEM: Initialize action history if it doesn't exist
        if not hasattr(self, 'action_history'):
            self.action_history = []
            
        # CREDIT_ASSIGNMENT_PROBLEM: Initialize pending penalties if it doesn't exist
        if not hasattr(self, 'pending_penalties'):
            self.pending_penalties = []
            
        # Reset reward components for this step
        for key in self.reward_components:
            self.reward_components[key] = 0.0
        
        # Calculate makespan reward - penalize longer makespans
        makespan_value = -0.01 * (current_time - prev_time)
        makespan_reward = makespan_value / 2.0 if makespan_value < 0 else makespan_value * 2.0
        
        # Calculate setup time penalty
        setup_value = -0.5 * setup_time if setup_time > 0 else 0.0
        setup_reward = setup_value / 2.0 if setup_value < 0 else setup_value * 2.0
        
        # Get the processing time for the current operation
        op_idx = self.job_progress[job_idx] - 1  # The operation that was just completed
        if op_idx >= 0:
            processing_time = self.jobs[job_idx]["operations"][op_idx]["processingTime"]
        else:
            processing_time = 0
        
        # Calculate idle time penalty
        idle_time = max(0, current_time - prev_time - setup_time - processing_time)
        idle_value = -0.2 * idle_time if idle_time > 0 else 0.0
        idle_penalty = idle_value / 2.0 if idle_value < 0 else idle_value * 2.0
        
        # MASCHINENSTILLSTAND: Calculate machine idle penalty - penalize machines that are idle
        machine_idle_penalty = 0.0
        if op_idx >= 0:
            # MASCHINENSTILLSTAND: Get the current machine
            machine_id = self.jobs[job_idx]["operations"][op_idx]["machineId"]
            machine_idx = self.machine_id_to_idx[machine_id]
            
            # MASCHINENSTILLSTAND: Calculate total idle time across all machines
            total_machine_idle_time = 0.0
            for m_idx in range(self.num_machines):
                # MASCHINENSTILLSTAND: Skip the current machine as it's being used
                if m_idx == machine_idx:
                    continue
                
                # MASCHINENSTILLSTAND: Calculate how long this machine has been idle
                machine_idle_time = max(0, current_time - self.machine_times[m_idx])
                total_machine_idle_time += machine_idle_time
            
            # MASCHINENSTILLSTAND: Calculate average idle time per machine (excluding current machine)
            if self.num_machines > 1:
                avg_machine_idle_time = total_machine_idle_time / (self.num_machines - 1)
                # MASCHINENSTILLSTAND: Penalty increases with longer average idle time
                machine_idle_value = -0.3 * avg_machine_idle_time
                machine_idle_penalty = machine_idle_value / 2.0  # MASCHINENSTILLSTAND: Negative value, so divide
            
            # MASCHINENSTILLSTAND: Additional penalty for machines that have been idle for too long (e.g., > 20% of current time)
            if current_time > 0:
                long_idle_machines = 0
                for m_idx in range(self.num_machines):
                    machine_idle_time = max(0, current_time - self.machine_times[m_idx])
                    if machine_idle_time > 0.2 * current_time:
                        long_idle_machines += 1
                
                if long_idle_machines > 0:
                    long_idle_penalty = -0.5 * long_idle_machines
                    machine_idle_penalty += long_idle_penalty / 2.0  # MASCHINENSTILLSTAND: Negative value, so divide
        
        # Calculate deadline reward
        job_id = self.idx_to_job_id[job_idx]
        deadline_reward = 0.0
        if job_completed:
            deadline = self.jobs[job_idx]["deadline"]
            if current_time <= deadline:
                deadline_value = 10.0  # Bonus for meeting deadline
                deadline_reward = deadline_value * 2.0
            else:
                deadline_value = -5.0 * (current_time - deadline) / deadline  # Penalty for missing deadline
                deadline_reward = deadline_value / 2.0
        
        # Calculate priority reward
        priority_reward = 0.0
        if job_completed:
            priority = self.jobs[job_idx]["priority"]
            priority_value = 2.0 * priority  # Higher priority jobs give more reward
            priority_reward = priority_value * 2.0  # Always positive, so multiply
        
        # Calculate critical job reward
        critical_job_reward = 0.0
        if job_completed:
            priority = self.jobs[job_idx]["priority"]
            if priority >= 8:  # Consider jobs with priority >= 8 as critical
                critical_job_value = 15.0
                critical_job_reward = critical_job_value * 2.0  # Always positive, so multiply
        
        # Calculate global progress reward
        global_progress_reward = 0.0
        if self.num_jobs > 0:
            progress_ratio = self.completed_jobs / self.num_jobs
            global_progress_value = 5.0 * progress_ratio
            global_progress_reward = global_progress_value * 2.0  # Always positive, so multiply
        
        # TIMELINESS: Berechne die Pünktlichkeitsbelohnung basierend auf dem Verhältnis von Makespan und Deadlines
        timeliness_reward = 0.0
        if self.num_jobs > 0:
            # TIMELINESS: Berechne das Verhältnis zwischen aktueller Zeit und durchschnittlicher Deadline
            avg_deadline = sum(job["deadline"] for job in self.jobs) / self.num_jobs
            if avg_deadline > 0:
                timeliness_factor = 1.0 - min(1.0, current_time / avg_deadline)
                # TIMELINESS: Positive Belohnung für gute Pünktlichkeit, negative für Verzögerungen
                timeliness_value = 8.0 * timeliness_factor
                timeliness_reward = timeliness_value * 2.0 if timeliness_value >= 0 else timeliness_value / 2.0
            
            # TIMELINESS: Zusätzliche Belohnung für die aktuelle Operation
            if op_idx >= 0:
                op_deadline = self.jobs[job_idx]["deadline"]
                op_expected_completion = op_deadline * (op_idx + 1) / len(self.jobs[job_idx]["operations"])
                if current_time <= op_expected_completion:
                    # TIMELINESS: Belohnung für frühzeitige Fertigstellung der Operation
                    additional_value = 2.0 * (1.0 - current_time / op_expected_completion)
                    timeliness_reward += additional_value * 2.0  # Always positive, so multiply
        
        # Calculate placement reward - reward for good operation placement
        placement_reward = 0.0
        # Überprüfe, ob die aktuelle Operation in den Placement-Insights enthalten ist
        op_idx = self.job_progress[job_idx] - 1  # Die gerade abgeschlossene Operation
        if op_idx >= 0:
            key = (job_idx, op_idx, self.machine_id_to_idx[self.jobs[job_idx]["operations"][op_idx]["machineId"]])
            if hasattr(self, 'placement_insights') and key in self.placement_insights:
                # Wenn die Operation in den Insights enthalten ist, bestrafe sie basierend auf der Schwere
                insights = self.placement_insights[key]
                severity_sum = sum(insight['severity'] for insight in insights)
                placement_value = -5.0 * severity_sum
                placement_reward = placement_value / 2.0  # Negative value, so divide
            else:
                # Belohne Operationen, die nicht in den Insights enthalten sind
                placement_value = 1.0
                placement_reward = placement_value * 2.0  # Positive value, so multiply
                
                # Zusätzliche Belohnung für Operationen mit hoher Priorität
                if self.jobs[job_idx]["priority"] >= 7:
                    placement_reward += 2.0 * 2.0  # Positive value, so multiply
                
                # Zusätzliche Belohnung für Operationen ohne Materialwechsel
                if setup_time <= self.setupTimes[self.jobs[job_idx]["operations"][op_idx]["machineId"]]['standard']:
                    placement_reward += 1.5 * 2.0  # Positive value, so multiply
        
        # Dieser Teil ersetzt den bestehenden Lookahead-Reward-Teil in der _calculate_reward Methode
        # Calculate lookahead reward - reward for actions that enable future good decisions
        lookahead_reward = 0.0
        if model is not None and self.completed_jobs < self.num_jobs:
            try:
                # Simulate future schedule with the current model
                sim_info = self.simulate_future_schedule(model, max_steps=min(20, self.num_jobs - self.completed_jobs))
                
                # Get the simulated makespan
                simulated_makespan = sim_info["makespan"]
                
                # Compare with previous episode's makespan (if available)
                if hasattr(self, 'previous_episode_makespan') and self.previous_episode_makespan > 0:
                    # Calculate improvement ratio
                    if simulated_makespan < self.previous_episode_makespan:
                        # Current action leads to better makespan than previous episode
                        improvement = (self.previous_episode_makespan - simulated_makespan) / self.previous_episode_makespan
                        lookahead_value = 3.0 * improvement  # Reward proportional to improvement
                        lookahead_reward += lookahead_value * 2.0  # Positive value, so multiply
                    else:
                        # Current action leads to worse makespan than previous episode
                        deterioration = (simulated_makespan - self.previous_episode_makespan) / self.previous_episode_makespan
                        lookahead_value = -2.0 * deterioration  # Penalty proportional to deterioration
                        lookahead_reward += lookahead_value / 2.0  # Negative value, so divide
                    
                    if self.enable_logging:
                        self.logger.info(f"Lookahead comparison: Current sim makespan: {simulated_makespan}, Previous episode makespan: {self.previous_episode_makespan}, Reward: {lookahead_reward:.2f}")
                else:
                    # No previous episode to compare with, use existing reward logic
                    # Reward based on completed jobs in simulation
                    completion_ratio = sim_info["completed_jobs"] / self.num_jobs if self.num_jobs > 0 else 0
                    completion_value = 2.0 * completion_ratio
                    lookahead_reward += completion_value * 2.0  # Positive value, so multiply
                    
                    # Reward based on met deadlines in simulation
                    if sim_info["completed_jobs"] > 0:
                        deadline_ratio = sim_info["met_deadlines"] / sim_info["completed_jobs"]
                        deadline_value = 3.0 * deadline_ratio
                        lookahead_reward += deadline_value * 2.0  # Positive value, so multiply
                    
                    # Penalty for suboptimal placements identified in simulation
                    if "suboptimal_placements" in sim_info and sim_info["completed_jobs"] == self.num_jobs:
                        suboptimal_ratio = sim_info["suboptimal_placements"] / len(sim_info.get("critical_path", [1]))
                        suboptimal_value = -2.0 * suboptimal_ratio
                        lookahead_reward += suboptimal_value / 2.0  # Negative value, so divide
                
            except Exception as e:
                print(f"Error in lookahead reward calculation: {e}")
                lookahead_reward = 0.0

        # Update previous episode makespan for the next iteration    
        
        # CREDIT_ASSIGNMENT_PROBLEM: Apply any pending penalties for this step
        credit_assignment_penalty = 0.0
        if hasattr(self, 'pending_penalties') and self.pending_penalties:
            current_step = self.episode_steps
            applicable_penalties = [p for p in self.pending_penalties if p['step'] == current_step]
            for penalty in applicable_penalties:
                credit_assignment_penalty += penalty['value']
                if self.enable_logging:
                    self.logger.info(f"CREDIT_ASSIGNMENT_PROBLEM: Applied delayed penalty of {penalty['value']:.2f} from issue: {penalty['issue']}")
            # Remove applied penalties
            self.pending_penalties = [p for p in self.pending_penalties if p['step'] != current_step]
        
        # CREDIT_ASSIGNMENT_PROBLEM: Record current action for future credit assignment
        if op_idx >= 0:
            action_record = {
                'step': self.episode_steps,
                'job_idx': job_idx,
                'operation_idx': op_idx,
                'machine_idx': self.machine_id_to_idx[self.jobs[job_idx]["operations"][op_idx]["machineId"]],
                'time': current_time
            }
            self.action_history.append(action_record)
        
        # CREDIT_ASSIGNMENT_PROBLEM: Check for issues that should trigger credit assignment
        if job_completed:
            deadline = self.jobs[job_idx]["deadline"]
            # If job missed deadline, assign penalties to past actions
            if current_time > deadline:
                delay_amount = current_time - deadline
                severity = delay_amount / deadline if deadline > 0 else 1.0
                base_penalty = -5.0 * severity
                
                # CREDIT_ASSIGNMENT_PROBLEM: Distribute penalties to past actions related to this job
                self._distribute_penalties(job_idx, base_penalty, "missed_deadline")
        
        # CREDIT_ASSIGNMENT_PROBLEM: Check for machine idle time issues
        if op_idx >= 0:
            machine_id = self.jobs[job_idx]["operations"][op_idx]["machineId"]
            machine_idx = self.machine_id_to_idx[machine_id]
            
            # If there was significant idle time, assign penalties to past actions
            idle_time = max(0, current_time - prev_time - setup_time - processing_time)
            if idle_time > 0.2 * processing_time:  # Significant idle time
                severity = idle_time / processing_time
                base_penalty = -2.0 * severity
                
                # CREDIT_ASSIGNMENT_PROBLEM: Distribute penalties to past actions related to this machine
                self._distribute_penalties(job_idx, base_penalty, "excessive_idle_time", machine_idx=machine_idx)
        
        # Store each reward component
        self.reward_components['makespan_reward'] = makespan_reward
        self.reward_components['setup_reward'] = setup_reward
        self.reward_components['idle_penalty'] = idle_penalty
        self.reward_components['deadline_reward'] = deadline_reward
        self.reward_components['priority_reward'] = priority_reward
        self.reward_components['critical_job_reward'] = critical_job_reward
        self.reward_components['global_progress_reward'] = global_progress_reward
        self.reward_components['placement_reward'] = placement_reward
        self.reward_components['lookahead_reward'] = lookahead_reward
        self.reward_components['timeliness_reward'] = timeliness_reward  # TIMELINESS: Speichere die Pünktlichkeitsbelohnung
        self.reward_components['machine_idle_penalty'] = machine_idle_penalty  # MASCHINENSTILLSTAND: Speichere die Maschinenstillstandsstrafe
        self.reward_components['credit_assignment_penalty'] = credit_assignment_penalty  # CREDIT_ASSIGNMENT_PROBLEM: Speichere die Credit-Assignment-Strafe
        
        # Update cumulative reward components
        for key in self.reward_components:
            self.cumulative_reward_components[key] += self.reward_components[key]
        
        # Calculate total reward
        total_reward = (makespan_reward + setup_reward + idle_penalty + deadline_reward + 
                        priority_reward + critical_job_reward + global_progress_reward + 
                        placement_reward + lookahead_reward + timeliness_reward +
                        machine_idle_penalty + credit_assignment_penalty)  # CREDIT_ASSIGNMENT_PROBLEM: Füge Credit-Assignment-Strafe zur Gesamtbelohnung hinzu
        
        return total_reward
    
    # CREDIT_ASSIGNMENT_PROBLEM: Neue Methode zur Verteilung von Strafen auf vergangene Aktionen
    def _distribute_penalties(self, current_job_idx, base_penalty, issue_type, machine_idx=None):
        """
        Distributes penalties to past actions that contributed to the current issue.
        
        Args:
            current_job_idx: Index of the current job
            base_penalty: Base penalty value to distribute
            issue_type: Type of issue that triggered the penalty
            machine_idx: Optional machine index for machine-specific issues
        """
        if not hasattr(self, 'action_history') or not self.action_history:
            return
            
        # Get relevant past actions (up to 10 steps back)
        current_step = self.episode_steps
        max_lookback = 10
        decay_factor = 0.7  # Exponential decay factor
        
        # Filter relevant actions based on the issue type
        relevant_actions = []
        if issue_type == "missed_deadline":
            # For missed deadlines, consider past actions on the same job
            relevant_actions = [a for a in self.action_history 
                               if a['job_idx'] == current_job_idx and 
                               a['step'] > current_step - max_lookback]
        elif issue_type == "excessive_idle_time" and machine_idx is not None:
            # For idle time issues, consider past actions on the same machine
            relevant_actions = [a for a in self.action_history 
                               if a['machine_idx'] == machine_idx and 
                               a['step'] > current_step - max_lookback]
        else:
            # For other issues, consider all recent actions
            relevant_actions = [a for a in self.action_history 
                               if a['step'] > current_step - max_lookback]
        
        # Sort actions by recency
        relevant_actions.sort(key=lambda a: a['step'], reverse=True)
        
        # Distribute penalties with exponential decay
        for i, action in enumerate(relevant_actions):
            # Skip the current action (it already gets the full penalty through other means)
            if action['step'] == current_step:
                continue
                
            # Calculate decayed penalty
            penalty_factor = decay_factor ** i  # Exponential decay
            penalty_value = base_penalty * penalty_factor
            
            # Add to pending penalties
            self.pending_penalties.append({
                'step': action['step'],
                'value': penalty_value,
                'issue': f"{issue_type} at step {current_step}"
            })
            
            if self.enable_logging:
                self.logger.info(f"CREDIT_ASSIGNMENT_PROBLEM: Scheduled penalty of {penalty_value:.2f} for step {action['step']} due to {issue_type} at step {current_step}")

    def _save_state(self):
        """
        Saves the current state of the environment for simulation.

        Returns:
            dict: A dictionary representing the saved state.
        """
        return {
            'job_progress': self.job_progress.copy(),
            'machine_times': self.machine_times.copy(),
            'current_time': self.current_time,
            'completed_jobs': self.completed_jobs,
            'current_machine_material': self.current_machine_material.copy(),
            'machine_material_idx': self.machine_material_idx.copy(),
            'episode_steps': self.episode_steps,
            'episode_reward': self.episode_reward,
            'episode_makespan': self.episode_makespan,
            'episode_completed_jobs': self.episode_completed_jobs,
            'episode_met_deadlines': self.episode_met_deadlines,
            'operation_history': self.operation_history.copy(),
            'machine_utilization': [util.copy() for util in self.machine_utilization],
            'material_changes': self.material_changes.copy(),
            'job_completion_times': self.job_completion_times.copy()
        }
    
    def _restore_state(self, saved_state):
        """
        Restores a previously saved state.

        Args:
            saved_state: The state dictionary to restore.
        """
        self.job_progress = saved_state['job_progress']
        self.machine_times = saved_state['machine_times']
        self.current_time = saved_state['current_time']
        self.completed_jobs = saved_state['completed_jobs']
        self.current_machine_material = saved_state['current_machine_material']
        self.machine_material_idx = saved_state['machine_material_idx']
        self.episode_steps = saved_state['episode_steps']
        self.episode_reward = saved_state['episode_reward']
        self.episode_makespan = saved_state['episode_makespan']
        self.episode_completed_jobs = saved_state['episode_completed_jobs']
        self.episode_met_deadlines = saved_state['episode_met_deadlines']
        self.operation_history = saved_state['operation_history']
        self.machine_utilization = saved_state['machine_utilization']
        self.material_changes = saved_state['material_changes']
        self.job_completion_times = saved_state['job_completion_times']

    def simulate_future_schedule(self, model, max_steps=50):
        """
        Simulates the remainder of the schedule using a pre-trained model.

        Args:
            model: The pre-trained model to use for simulation.
            max_steps: Maximum number of simulation steps.

        Returns:
            dict: A dictionary containing simulation details.
        """
        saved_state = self._save_state()
        
        # Speichere die ursprünglichen Reward-Komponenten
        original_reward_components = None
        original_cumulative_reward_components = None
        if hasattr(self, 'reward_components'):
            original_reward_components = self.reward_components.copy()
        if hasattr(self, 'cumulative_reward_components'):
            original_cumulative_reward_components = self.cumulative_reward_components.copy()
        
        # Speichere auch action_history und pending_penalties
        original_action_history = None
        original_pending_penalties = None
        if hasattr(self, 'action_history'):
            original_action_history = self.action_history.copy()
        if hasattr(self, 'pending_penalties'):
            original_pending_penalties = self.pending_penalties.copy()
        
        sim_steps = 0
        sim_rewards = []
        done = False
        observation = self._get_observation()
        
        try:
            while not done and sim_steps < max_steps:
                if np.sum(observation['valid_actions_mask']) == 0:
                    break
                model_action, _ = model.select_action(observation)
                # Pass None as the model to avoid infinite recursion
                observation, reward, done, info = self.step(model_action, model=None)
                sim_rewards.append(reward)
                sim_steps += 1
            
            final_makespan = max(self.machine_times)
            completed_jobs = self.completed_jobs
            met_deadlines = self.episode_met_deadlines
            
            critical_path_info = {}
            if completed_jobs == self.num_jobs:
                critical_path = self.analyze_critical_path()
                insights = self.identify_suboptimal_placements(critical_path)
                critical_path_info = {
                    "critical_path_length": len(critical_path),
                    "suboptimal_placements": len(insights),
                    "critical_path": critical_path
                }
            
            # Stelle die ursprünglichen Reward-Komponenten wieder her
            if original_reward_components is not None:
                self.reward_components = original_reward_components
            if original_cumulative_reward_components is not None:
                self.cumulative_reward_components = original_cumulative_reward_components
            
            # Stelle auch action_history und pending_penalties wieder her
            if original_action_history is not None:
                self.action_history = original_action_history
            if original_pending_penalties is not None:
                self.pending_penalties = original_pending_penalties
            
            self._restore_state(saved_state)
            
            simulation_info = {
                "makespan": final_makespan,
                "completed_jobs": completed_jobs,
                "met_deadlines": met_deadlines,
                "simulation_steps": sim_steps,
                "cumulative_reward": sum(sim_rewards)
            }
            simulation_info.update(critical_path_info)
            return simulation_info
        except Exception as e:
            print(f"Error in simulate_future_schedule: {e}")
            
            # Stelle die ursprünglichen Reward-Komponenten wieder her
            if original_reward_components is not None:
                self.reward_components = original_reward_components
            if original_cumulative_reward_components is not None:
                self.cumulative_reward_components = original_cumulative_reward_components
            
            # Stelle auch action_history und pending_penalties wieder her
            if original_action_history is not None:
                self.action_history = original_action_history
            if original_pending_penalties is not None:
                self.pending_penalties = original_pending_penalties
                
            self._restore_state(saved_state)
            return {
                "makespan": max(self.machine_times),
                "completed_jobs": 0,
                "met_deadlines": 0,
                "simulation_steps": 0,
                "cumulative_reward": 0,
                "error": str(e)
            }         
    def get_machine_utilization_stats(self):
        """
        Returns detailed machine utilization statistics.

        Returns:
            dict: Dictionary containing machine utilization statistics.
        """
        stats = {}
        for machine_idx in range(self.num_machines):
            machine_id = self.idx_to_machine_id[machine_idx]
            machine_records = self.machine_utilization[machine_idx]
            if not machine_records:
                stats[machine_id] = {
                    'utilization': 0.0,
                    'setup_time_ratio': 0.0,
                    'idle_time_ratio': 0.0,
                    'processing_time_ratio': 0.0
                }
                continue
            total_busy_time = sum(record['busy_time'] for record in machine_records)
            total_setup_time = sum(record['setup_time'] for record in machine_records)
            total_idle_time = sum(record['idle_time'] for record in machine_records)
            if self.current_time > 0:
                utilization = self.machine_times[machine_idx] / self.current_time
                setup_time_ratio = total_setup_time / self.current_time
                idle_time_ratio = total_idle_time / self.current_time
                processing_time_ratio = total_busy_time / self.current_time
            else:
                utilization = 0.0
                setup_time_ratio = 0.0
                idle_time_ratio = 0.0
                processing_time_ratio = 0.0
            stats[machine_id] = {
                'utilization': utilization,
                'setup_time_ratio': setup_time_ratio,
                'idle_time_ratio': idle_time_ratio,
                'processing_time_ratio': processing_time_ratio,
                'total_busy_time': total_busy_time,
                'total_setup_time': total_setup_time,
                'total_idle_time': total_idle_time
            }
        return stats

    def get_material_change_stats(self):
        """
        Returns statistics about material changes.

        Returns:
            dict: Dictionary containing material change statistics.
        """
        stats = {}
        for machine_idx in range(self.num_machines):
            machine_id = self.idx_to_machine_id[machine_idx]
            machine_changes = [change for change in self.material_changes if change['machine_idx'] == machine_idx]
            material_counts = {}
            for change in machine_changes:
                new_material = change['new_material']
                material_counts[new_material] = material_counts.get(new_material, 0) + 1
            total_setup_time = sum(change['setup_time'] for change in machine_changes)
            stats[machine_id] = {
                'total_changes': len(machine_changes),
                'total_setup_time': total_setup_time,
                'material_counts': material_counts
            }
        return stats

    def get_job_completion_stats(self):
        """
        Returns statistics about job completions.

        Returns:
            dict: Dictionary containing job completion statistics.
        """
        if self.job_completion_times:
            avg_completion_time = sum(job['completion_time'] for job in self.job_completion_times.values()) / len(self.job_completion_times)
        else:
            avg_completion_time = 0.0
        met_deadlines = sum(1 for job in self.job_completion_times.values() if job['deadline_met'])
        deadline_ratio = met_deadlines / len(self.job_completion_times) if self.job_completion_times else 0.0
        priority_weighted_completion = sum(job['completion_time'] * job['priority'] for job in self.job_completion_times.values()) if self.job_completion_times else 0.0
        total_priority = sum(job['priority'] for job in self.job_completion_times.values()) if self.job_completion_times else 0.0
        priority_weighted_avg = priority_weighted_completion / total_priority if total_priority > 0 else 0.0
        high_priority_jobs = {job_id: job for job_id, job in self.job_completion_times.items() if job['priority'] >= 7}
        high_priority_met = sum(1 for job in high_priority_jobs.values() if job['deadline_met'])
        high_priority_ratio = high_priority_met / len(high_priority_jobs) if high_priority_jobs else 0.0
        return {
            'completed_jobs': len(self.job_completion_times),
            'avg_completion_time': avg_completion_time,
            'met_deadlines': met_deadlines,
            'deadline_ratio': deadline_ratio,
            'priority_weighted_avg_completion': priority_weighted_avg,
            'high_priority_met_ratio': high_priority_ratio
        }
    
    def get_reward_stats(self):
        """
        Returns statistics about rewards.

        Returns:
            dict: Dictionary containing reward statistics.
        """
        if not hasattr(self, 'cumulative_reward_components'):
            return {}
        
        # Kopiere die kumulierten Reward-Komponenten, um sie zurückzugeben
        reward_stats = self.cumulative_reward_components.copy()
        
        # Füge die Summe aller Komponenten hinzu
        reward_stats['total'] = sum(reward_stats.values())
        
        return reward_stats

    
    