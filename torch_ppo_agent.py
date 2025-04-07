import torch
import torch.nn as nn
import numpy as np
from jsp_graph import create_jsp_graph
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

class TorchPPOAgent:
    def __init__(self, num_jobs, jsp_data):
        self.num_jobs = num_jobs
        self.jsp_data = jsp_data
        
        # Erstelle Mapping von Job-IDs zu Indizes und umgekehrt
        self.job_id_to_idx = {job["id"]: idx for idx, job in enumerate(jsp_data["jobs"])}
        self.idx_to_job_id = {idx: job["id"] for idx, job in enumerate(jsp_data["jobs"])}
        
        # Erstelle Mapping von Maschinen-IDs zu Indizes und umgekehrt
        self.machine_id_to_idx = {machine["id"]: idx for idx, machine in enumerate(jsp_data["machines"])}
        self.idx_to_machine_id = {idx: machine["id"] for idx, machine in enumerate(jsp_data["machines"])}
        
        # JSP-Graph erstellen und vorbereiten
        self.G, _ = create_jsp_graph(jsp_data)
        self.G = self.prepare_graph_for_learning(self.G)
        
        # Erstelle Mapping von Knoten-IDs zu Indizes für den Graph Transformer
        self.node_id_to_idx = {}
        idx = 0
        for job in jsp_data["jobs"]:
            for op in job["operations"]:
                node_id = f"{job['id']}:{op['id']}"
                self.node_id_to_idx[node_id] = idx
                idx += 1
        
        # Anzahl der Features pro Knoten
        node_features = 7
        
        # Parameter für Embedding und Transformer
        self.embedding_dim = 64   # Dimension des initialen Node-Embeddings
        self.transformer_layers = 2  # Anzahl der Transformer-Encoder-Schichten
        self.nhead = 4  # Anzahl der Attention-Köpfe im Transformer
        
        # Node-Embedding-Layer: wandelt Rohfeatures in einen kontinuierlichen Vektor um
        self.node_embedding = nn.Linear(node_features, self.embedding_dim)
        
        # Statt TransformerConv verwenden wir GATConv (Graph Attention Network)
        # Dies ist besser geeignet für unseren Anwendungsfall und hat weniger Probleme mit der Dimensionalität
        self.graph_transformer_layers = nn.ModuleList([
            pyg_nn.GATConv(
                in_channels=self.embedding_dim,
                out_channels=self.embedding_dim // self.nhead,
                heads=self.nhead,
                dropout=0.1,
                edge_dim=2  # Für edge_type (1 oder 2)
            ) for _ in range(self.transformer_layers)
        ])
        
        # Output-Layer: Wandelt den globalen Zustandsvektor in Logits (für Job-Aktionswahrscheinlichkeiten) um
        self.output_layer = nn.Linear(self.embedding_dim, num_jobs)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.node_embedding.parameters()) +
            list(self.graph_transformer_layers.parameters()) +
            list(self.output_layer.parameters()),
            lr=0.001, weight_decay=1e-5
        )
        
        # PPO-Parameter und weitere Einstellungen (wie bisher)
        self.epsilon = 0.3
        self.gamma = 0.99
        self.exploration_rate = 0.8
        self.exploration_decay = 0.9998
        self.min_exploration_rate = 0.15
        self.temperature = 2.5
        self.experiences = []
        
        # Materialtypen extrahieren (wie bisher)
        self.materials = set()
        for job in jsp_data["jobs"]:
            for op in job["operations"]:
                self.materials.add(self._extract_material_type(op["material"]))
        self.materials = list(self.materials)
    
    def _extract_material_type(self, material_string):
        if "_" in material_string:
            return material_string.split("_")[0]
        return material_string
    
    def _get_material_index(self, material_string):
        material_type = self._extract_material_type(material_string)
        if material_type in self.materials:
            return self.materials.index(material_type)
        return 0
    
    def prepare_graph_for_learning(self, G):
        for u, v, data in G.edges(data=True):
            if data.get('color') == 'blue':
                G.edges[u, v]['edge_type'] = 1  # Konjunktive Kante (gleicher Job)
            elif data.get('color') == 'red':
                G.edges[u, v]['edge_type'] = 2  # Disjunktive Kante (gleiche Maschine)
        return G
    
    def create_pyg_graph(self, node_ids, node_features, job_indices):
        """
        Erstellt einen PyTorch Geometric Graphen aus den gegebenen Knoten und Features.
        """
        # Erstelle Edge-Index und Edge-Attribute für PyG
        edge_index = []
        edge_attr = []
        
        # Füge Kanten aus dem JSP-Graphen hinzu
        for i, node_i in enumerate(node_ids):
            for j, node_j in enumerate(node_ids):
                if self.G.has_edge(node_i, node_j):
                    edge_index.append([i, j])
                    edge_type = self.G.edges[node_i, node_j].get('edge_type', 0)
                    edge_attr.append([float(edge_type), 0.0])  # Zweites Feature ist Platzhalter
        
        if not edge_index:  # Falls keine Kanten vorhanden sind
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # Transponieren für PyG-Format
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Erstelle PyG Data-Objekt
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            job_indices=torch.tensor(job_indices, dtype=torch.long)
        )
        
        return data
    
    def state_to_tensor(self, state):
        """
        Konvertiert den Zustand in einen globalen Zustandsvektor mithilfe eines Graph Transformers.
        Berücksichtigt die Struktur des disjunktiven Graphen bei der Transformation.
        """
        job_progress = state['job_progress']
        
        # Sammle alle aktiven Knoten und ihre Features
        all_nodes = []
        all_node_ids = []
        all_job_indices = []  # Speichere, zu welchem Job jeder Knoten gehört
        all_node_features = []
        
        for job_idx, job_id in self.idx_to_job_id.items():
            job = self.jsp_data["jobs"][job_idx]
            for op_idx, operation in enumerate(job["operations"]):
                node_id = f"{job_id}:{operation['id']}"
                all_node_ids.append(node_id)
                all_job_indices.append(job_idx)
                
                # Berechne Features
                machine_id = operation["machineId"]
                machine_idx = self.machine_id_to_idx[machine_id]
                material = operation["material"]
                material_idx = self._get_material_index(material)
                progress = job_progress[job_idx]
                completed = 1.0 if progress > op_idx else 0.0
                
                normalized_feature = [
                    job_idx / self.num_jobs,                                # Job-Index
                    op_idx / max(1, len(job["operations"])),                # Operation-Index
                    machine_idx / len(self.jsp_data["machines"]),           # Maschinen-Index
                    operation["processingTime"] / 100.0,                     # Bearbeitungszeit
                    job["priority"] / 10.0,                                  # Priorität
                    job["deadline"] / 200.0,                                 # Deadline
                    material_idx / max(1, len(self.materials))               # Materialtyp
                ]
                
                feature_tensor = torch.tensor(normalized_feature, dtype=torch.float32)
                all_node_features.append(feature_tensor)
        
        if not all_node_features:
            return torch.zeros(self.embedding_dim)
        
        # Stapel alle Knoten-Features
        node_features = torch.stack(all_node_features)
        
        # Erstelle initialen Node-Embedding
        node_embeddings = self.node_embedding(node_features)
        
        # Erstelle PyG-Graph
        graph = self.create_pyg_graph(all_node_ids, node_embeddings, all_job_indices)
        
        # Wende Graph Transformer Layer an
        x = graph.x
        for layer in self.graph_transformer_layers:
            # GATConv erwartet andere Parameter als TransformerConv
            if graph.edge_index.size(1) > 0:  # Nur wenn Kanten vorhanden sind
                x = layer(x, graph.edge_index, edge_attr=graph.edge_attr)
            else:
                # Wenn keine Kanten vorhanden sind, behalte die Features bei
                pass
            x = torch.nn.functional.relu(x)
        
        # Aggregiere Knoten pro Job
        job_embeddings = []
        for job_idx in range(self.num_jobs):
            # Finde alle Knoten, die zu diesem Job gehören
            job_mask = (graph.job_indices == job_idx)
            if job_mask.any():
                job_nodes = x[job_mask]
                job_embedding = torch.mean(job_nodes, dim=0)
                job_embeddings.append(job_embedding)
        
        # Globales Zustands-Embedding
        if job_embeddings:
            global_state = torch.mean(torch.stack(job_embeddings), dim=0)
        else:
            global_state = torch.zeros(self.embedding_dim)
        
        return global_state
    
    def select_action(self, state):
        """
        Wählt eine Aktion basierend auf dem aktuellen Zustand unter Verwendung des Transformer-basierten Zustandsvektors.
        Die Entscheidungsfindung nutzt neben explorativen Strategien auch den Output-Layer, der auf den transformierten Zustand angewandt wird.
        """
        # Zunächst wird der globale Zustand mittels des Transformers berechnet
        state_embedding = self.state_to_tensor(state)
        
        # Weiterverarbeitung durch den Output-Layer
        logits = self.output_layer(state_embedding)
        probs = torch.nn.functional.softmax(logits, dim=0)
        
        # Bestimme gültige Aktionen anhand des valid_actions_mask aus dem Zustand (wie bisher)
        valid_jobs = []
        if 'valid_actions_mask' in state:
            for job_idx, is_valid in enumerate(state['valid_actions_mask']):
                if is_valid == 1:
                    valid_jobs.append(job_idx)
        else:
            for job_idx in range(self.num_jobs):
                if state['job_progress'][job_idx] < len(self.jsp_data["jobs"][job_idx]["operations"]):
                    valid_jobs.append(job_idx)
        
        # Maskiere ungültige Aktionen
        valid_probs = torch.tensor([probs[i].item() if i in valid_jobs else 0.0 for i in range(self.num_jobs)])
        if torch.sum(valid_probs) > 0:
            valid_probs = valid_probs / torch.sum(valid_probs)
        else:
            valid_probs = torch.tensor([1.0/len(valid_jobs) if i in valid_jobs else 0.0 for i in range(self.num_jobs)])
        
        action = torch.multinomial(valid_probs, 1).item()
        action_prob = valid_probs[action].item()
        return action, action_prob
    
    # store_experience, update, get_makespan_reward, save_model, load_model und parameters
    # bleiben weitgehend unverändert.
    
    def store_experience(self, state, action, action_prob, reward, next_state, done):
        self.experiences.append({
            'state': state,
            'action': action,
            'action_prob': action_prob,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def update(self, batch_size=32):
        # Update-Logik (PPO) – hier bleibt der Großteil der Logik erhalten.
        if len(self.experiences) < batch_size:
            return 0.0
        
        rewards = [exp['reward'] for exp in self.experiences]
        dones = [exp['done'] for exp in self.experiences]
        returns = []
        advantages = []
        lambda_gae = 0.95
        gae = 0
        next_value = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                delta = r - 0
                gae = delta
            else:
                delta = r + self.gamma * next_value - 0
                gae = delta + self.gamma * lambda_gae * gae
            next_value = 0 if done else r
            returns.insert(0, gae)
            advantages.insert(0, gae)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        epochs = 10
        total_loss = 0.0
        for _ in range(epochs):
            indices = torch.randperm(len(self.experiences))
            epoch_loss = 0.0
            for start_idx in range(0, len(self.experiences), batch_size):
                batch_indices = indices[start_idx:min(start_idx + batch_size, len(indices))]
                batch_states = []
                batch_actions = []
                batch_old_probs = []
                batch_advantages = []
                for idx in batch_indices:
                    exp = self.experiences[idx.item()]
                    batch_states.append(self.state_to_tensor(exp['state']))
                    batch_actions.append(exp['action'])
                    batch_old_probs.append(exp['action_prob'])
                    batch_advantages.append(advantages[idx.item()])
                batch_actions = torch.tensor(batch_actions, dtype=torch.long)
                batch_old_probs = torch.tensor(batch_old_probs, dtype=torch.float32)
                batch_advantages = torch.tensor(batch_advantages, dtype=torch.float32)
                
                batch_logits = []
                for state_embedding in batch_states:
                    logits = self.output_layer(state_embedding)
                    batch_logits.append(logits)
                
                batch_new_probs = []
                batch_entropies = []
                for i, logits in enumerate(batch_logits):
                    probs = torch.nn.functional.softmax(logits, dim=0)
                    batch_new_probs.append(probs[batch_actions[i]])
                    batch_entropies.append(-torch.sum(probs * torch.log(probs + 1e-10)))
                
                batch_new_probs = torch.stack(batch_new_probs)
                batch_entropies = torch.stack(batch_entropies)
                ratio = batch_new_probs / (batch_old_probs + 1e-10)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = batch_entropies.mean()
                loss = actor_loss - 0.1 * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_indices)
            total_loss += epoch_loss / len(self.experiences)
        
        total_loss /= epochs
        self.experiences = []
        return total_loss / len(rewards)
    
    def get_makespan_reward(self, state, action, next_state):
        # Belohnungsberechnung (wie bisher)
        current_makespan = max(state['machine_times'])
        next_makespan = max(next_state['machine_times'])
        makespan_diff = next_makespan - current_makespan
        if 'current_time' in next_state:
            if isinstance(next_state['current_time'], (list, np.ndarray)):
                current_time = next_state['current_time'][0]
            else:
                current_time = next_state['current_time']
        else:
            current_time = next_makespan
        machine_times = next_state['machine_times']
        if next_makespan > 0:
            mean_time = sum(machine_times) / len(machine_times)
            variance = sum((t - mean_time) ** 2 for t in machine_times) / len(machine_times)
            std_dev = variance ** 0.5
            total_util = sum(machine_times) / (current_time * len(machine_times)) if current_time > 0 else 0
            balance_util = 1.0 / (1.0 + std_dev / mean_time) if mean_time > 0 else 0
            machine_util = (total_util * 0.7) + (balance_util * 0.3)
        else:
            machine_util = 0
        job_progress = next_state['job_progress']
        total_operations = 0
        weighted_progress = 0
        completed_jobs = 0
        for job_idx, progress in enumerate(job_progress):
            job = self.jsp_data["jobs"][job_idx]
            job_priority = job["priority"]
            job_ops_count = len(job["operations"])
            total_operations += job_ops_count
            weighted_progress += progress * job_priority
            if progress >= job_ops_count:
                completed_jobs += 1
        total_priority = sum(job["priority"] for job in self.jsp_data["jobs"])
        progress_ratio = weighted_progress / (total_operations * total_priority / self.num_jobs) if total_operations > 0 else 0
        job_completed = False
        job_priority = 0
        job_deadline = 0
        deadline_exceeded = False
        remaining_time = 0
        if action < self.num_jobs:
            job_idx = action
            job = self.jsp_data["jobs"][job_idx]
            if state['job_progress'][job_idx] < len(job["operations"]) and next_state['job_progress'][job_idx] >= len(job["operations"]):
                job_completed = True
                job_priority = job["priority"]
                job_deadline = job["deadline"]
                deadline_exceeded = current_time > job_deadline
                if deadline_exceeded:
                    remaining_time = current_time - job_deadline
        met_deadlines = 0
        for job_idx, progress in enumerate(job_progress):
            job = self.jsp_data["jobs"][job_idx]
            if progress >= len(job["operations"]) and current_time <= job["deadline"]:
                met_deadlines += 1
        deadline_ratio = met_deadlines / max(1, completed_jobs) if completed_jobs > 0 else 0
        makespan_reward = -makespan_diff * 3.0 if makespan_diff > 0 else 10.0
        utilization_reward = machine_util * 8.0
        progress_reward = progress_ratio * 5.0
        deadline_overall_reward = deadline_ratio * 7.0
        priority_reward = 0.0
        deadline_job_reward = 0.0
        if job_completed:
            priority_reward = job_priority * 3.0
            if deadline_exceeded:
                deadline_job_reward = -15.0 - (remaining_time / 5.0)
            else:
                time_before_deadline = job_deadline - current_time
                deadline_job_reward = 20.0 + (time_before_deadline / 3.0)
        setup_reward = 0.0
        if 'setup_time' in next_state:
            setup_time = next_state['setup_time']
            setup_reward = -setup_time / 3.0
        critical_path_reward = 0.0
        if 'critical_path' in next_state:
            critical_path = next_state['critical_path']
            if 'critical_path' in state:
                critical_path_diff = state['critical_path'] - critical_path
                critical_path_reward = critical_path_diff * 2.0
        reward = (makespan_reward + utilization_reward + progress_reward + priority_reward +
                  deadline_job_reward + deadline_overall_reward + setup_reward + critical_path_reward)
        if 'job_completed' in next_state and next_state['job_completed']:
            reward += 15.0
        reward = max(min(reward, 75.0), -75.0)
        return reward
    
    def save_model(self, path):
        model_state = {
            'node_embedding': self.node_embedding.state_dict(),
            'graph_transformer_layers': [layer.state_dict() for layer in self.graph_transformer_layers],
            'output_layer': self.output_layer.state_dict()
        }
        torch.save(model_state, path)
    
    def load_model(self, path):
        model_state = torch.load(path)
        self.node_embedding.load_state_dict(model_state['node_embedding'])
        for i, layer_state in enumerate(model_state['graph_transformer_layers']):
            self.graph_transformer_layers[i].load_state_dict(layer_state)
        self.output_layer.load_state_dict(model_state['output_layer'])
        
    def parameters(self):
        return list(self.node_embedding.parameters()) + \
               list(self.graph_transformer_layers.parameters()) + \
               list(self.output_layer.parameters())
