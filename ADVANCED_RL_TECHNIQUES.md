# Advanced RL Techniques and Agent Improvement Strategies

This document outlines advanced reinforcement learning techniques to improve your soccer agent beyond basic reward shaping.

## Overview

Your agents are trained using **Proximal Policy Optimization (PPO)**, a state-of-the-art policy gradient algorithm. The following techniques can further enhance performance:

---

## 1. Curriculum Learning

**What it is**: Gradually increasing task difficulty during training.

**How it helps**: 
- Start with easier opponents (stationary targets, random actions)
- Progress to harder opponents as agent improves
- Prevents agent from getting stuck in local minima

**Implementation in this codebase**:
```bash
python train_ray_curriculum.py
```

**Modification ideas**:
```python
# In your training config:
"callbacks": {
    "on_train_result": lambda algorithm, result: 
        update_difficulty(algorithm, result)
}
```

**Advanced approach**: Use **Dynamic Difficulty Adjustment (DDA)**
- Measure agent win rate
- If win_rate > 0.8, increase opponent skill
- If win_rate < 0.3, decrease opponent skill
- Keeps agent in "learning zone"

---

## 2. Self-Play Training

**What it is**: Agent plays against previous versions of itself.

**How it helps**:
- Creates increasingly difficult opponents
- Agent learns to counter strategies it knows well
- Enables continuous improvement without external opponents

**Implementation**:
```bash
python train_ray_selfplay.py
```

**Key parameters**:
```python
config = {
    "policy_graphs": {
        "main_policy": ...,
        "opponent_policy": ...  # Older checkpoint
    },
    "callbacks": {
        "on_checkpoint": update_opponent_from_main_checkpoint
    }
}
```

**Advanced variations**:
- **League Play**: Maintain pool of agents, tournament selection for opponents
- **Population Based Training (PBT)**: Automatically adjust hyperparameters during training

---

## 3. Multi-Agent Learning Enhancements

**A. Communication Protocols**

Agents can learn to "communicate" actions:

```python
class CommunicativeAgent:
    def forward(self, obs, teammate_communication):
        # Encode intent
        intent = self.intent_encoder(obs)
        
        # Receive teammate communication
        combined = torch.cat([obs, teammate_communication], dim=-1)
        
        # Predict action
        action = self.policy(combined)
        return action, intent
```

**B. Attention Mechanisms**

Learn to focus on relevant agents:

```python
# Use Multi-Head Attention to weight teammate/opponent positions
class AttentionAgent(nn.Module):
    def __init__(self, obs_size, hidden_size=512):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.fc = nn.Linear(hidden_size, 4)  # Output 4 actions
    
    def forward(self, obs):
        # obs shape: [batch_size, num_entities * entity_dim]
        x = self.fc_in(obs)
        
        # Apply attention
        x, _ = self.attention(x, x, x)
        
        action = self.fc(x)
        return action
```

**C. Cooperation Loss**

Add term to reward function promoting team coordination:

```python
cooperation_loss = -torch.sum(
    teammate_action_similarity(action_1, action_2)
)
total_loss = rl_loss + 0.1 * cooperation_loss
```

---

## 4. Imitation Learning from Baseline

**What it is**: Learning from expert demonstrations (the baseline agent).

**How it helps**:
- Bootstrap agent with good strategies
- Faster initial learning
- Combines behavioral cloning with RL

**Implementation**:

```python
# Step 1: Collect trajectories from baseline agent
def collect_baseline_trajectories(num_episodes=100):
    env = soccer_twos.make(...)
    baseline_agent = load_baseline_agent()
    
    trajectories = []
    for _ in range(num_episodes):
        obs = env.reset()
        trajectory = {'obs': [], 'actions': []}
        
        while True:
            action = baseline_agent.act({0: obs})
            trajectory['obs'].append(obs)
            trajectory['actions'].append(action)
            obs, _, done, _ = env.step(action)
            if done: break
        
        trajectories.append(trajectory)
    return trajectories

# Step 2: Pre-train policy with behavioral cloning
class ImitationAgent(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 4)  # 4 action dimensions
    
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Step 3: Train with imitation + RL loss
def compute_loss(policy_action, expert_action, rl_reward):
    imitation_loss = F.mse_loss(policy_action, expert_action)
    rl_loss = -rl_reward * torch.log(policy_probs + 1e-8)
    
    # Weighted combination
    total_loss = 0.7 * imitation_loss + 0.3 * rl_loss
    return total_loss
```

**In Ray Tune**:
```python
from ray.rllib.contrib.imitation import BC

# Combine BC (Behavioral Cloning) with PPO
config = {
    "algo": "PPO",
    "behavioral_cloning_loss_coeff": 0.5,  # Weight of BC loss
}
```

---

## 5. Advanced Reward Shaping Techniques

### A. Potential-Based Reward Shaping

Theoretically guaranteed to preserve optimal policy:

```python
def potential_reward(obs, next_obs, base_reward, phi_fn):
    """
    Shaped reward = base_reward + gamma * phi(next_obs) - phi(obs)
    where phi() is a potential function (e.g., distance to goal)
    """
    gamma = 0.99
    
    def phi(obs_state):
        # Distance to opponent goal
        agent_pos = obs_state[:3]
        goal_pos = obs_state[-3:]
        return -np.linalg.norm(agent_pos - goal_pos)
    
    shaped_reward = (
        base_reward + 
        gamma * phi(next_obs) - 
        phi(obs)
    )
    return shaped_reward
```

### B. Intrinsic Motivation (Curiosity)

Encourage exploration of novel states:

```python
class CuriosityModule(nn.Module):
    def __init__(self, obs_size, hidden_size=512):
        super().__init__()
        # Forward model: predict next obs from current
        self.forward_model = nn.Sequential(
            nn.Linear(obs_size + 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, obs_size)
        )
        
        # Inverse model: predict action from obs change
        self.inverse_model = nn.Sequential(
            nn.Linear(obs_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)
        )
    
    def forward(self, obs, action, next_obs):
        # Prediction errors
        predicted_next = self.forward_model(
            torch.cat([obs, action], dim=-1)
        )
        forward_error = F.mse_loss(predicted_next, next_obs)
        
        predicted_action = self.inverse_model(
            torch.cat([obs, next_obs], dim=-1)
        )
        inverse_error = F.mse_loss(predicted_action, action)
        
        # Curiosity reward: higher when prediction error is large
        intrinsic_reward = forward_error.detach()
        
        return intrinsic_reward, forward_error + inverse_error

# Total reward = extrinsic + alpha * intrinsic
total_reward = base_reward + 0.1 * curiosity_reward
```

---

## 6. Network Architecture Improvements

### A. Recurrent Networks (LSTM/GRU)

Useful for partially observable environments:

```python
class RecurrentPolicy(nn.Module):
    def __init__(self, obs_size, hidden_size=256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=obs_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 4)
    
    def forward(self, obs_sequence, hidden_state=None):
        # obs_sequence: [batch, seq_len, obs_size]
        lstm_out, hidden_state = self.lstm(obs_sequence, hidden_state)
        
        # Take last timestep
        action_logits = self.fc(lstm_out[:, -1, :])
        return action_logits, hidden_state

# In training config:
config = {
    "model": {
        "use_lstm": True,
        "lstm_cell_size": 256,
        "max_seq_len": 20,
    }
}
```

### B. Dueling Architecture

Separate value and advantage streams:

```python
class DuelingPolicy(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # 4 actions
        )
    
    def forward(self, obs):
        shared = self.shared(obs)
        
        value = self.value_stream(shared)
        advantages = self.advantage_stream(shared)
        
        # Combine: Q = V + (A - mean(A))
        action_values = value + (advantages - advantages.mean())
        return action_values
```

---

## 7. Hyperparameter Tuning

Use **Population Based Training (PBT)** for automatic tuning:

```bash
# In your training config:
from ray import tune

config = {
    "lr": tune.loguniform(1e-6, 1e-3),
    "gamma": tune.choice([0.95, 0.99, 0.999]),
    "entropy_coeff": tune.loguniform(0.001, 0.1),
}

# PBT scheduler will explore and exploit good hyperparameters
pbt_scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=100,
    hyperparam_mutations={
        "lr": [1e-6, 1e-5, 1e-4, 1e-3],
        "entropy_coeff": [0.001, 0.01, 0.1],
    }
)

tune.run(
    "PPO",
    config=config,
    scheduler=pbt_scheduler,
)
```

---

## 8. Efficient Data Collection

### A. Prioritized Experience Replay

Focus on important experiences:

```python
# Higher priority for large TD-errors
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.priorities = []
        self.experiences = []
    
    def add(self, experience, td_error):
        priority = (abs(td_error) + 1e-6) ** self.alpha
        self.priorities.append(priority)
        self.experiences.append(experience)
        
        if len(self) > self.capacity:
            self.priorities.pop(0)
            self.experiences.pop(0)
    
    def sample(self, batch_size):
        # Sample proportional to priority
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self), batch_size, p=probs)
        return [self.experiences[i] for i in indices]
```

### B. Distributed Training

Already built into Ray Tune:

```python
config = {
    "num_workers": 32,  # 32 parallel workers
    "num_envs_per_worker": 4,  # 4 envs per worker
    "num_gpus": 4,  # Distribute across 4 GPUs
    # Total: 32 * 4 = 128 environments in parallel
}
```

---

## 9. Evaluation and Testing Strategy

### Before Training
1. Test environment setup: `python example_random_players.py`
2. Verify baseline loads: `python -m soccer_twos.watch -m1 example_player_agent -m2 ceia_baseline_agent`

### During Training
1. Monitor TensorBoard: episode reward mean, policy loss, value loss
2. Check for divergence or instability
3. Adjust hyperparameters if needed

### After Training
1. **Test vs Random Agent**: Should win 9/10
2. **Test vs Baseline Agent**: Target 9/10 wins
3. **Test vs TA Agent**: When released (stretch goal)
4. **Analyze failures**: Record videos of losses to understand weaknesses

### Analysis Tips
```python
# Extract best checkpoints
best_ckpt = analysis.get_best_checkpoint(
    trial=best_trial,
    metric="episode_reward_mean",
    mode="max"
)

# Load and replay
from ray.rllib.algorithms import PPO
algo = PPO.from_checkpoint(best_ckpt)
```

---

## 10. Quick Start: Combining Techniques

Here's a recommended approach:

```python
# 1. Start with Strategy 5 (Coordinated Teamplay)
python train_strategies.py --strategy 5 --workers 16

# 2. Use self-play to improve
# Modify train_ray_selfplay.py to load best checkpoint
python train_ray_selfplay.py

# 3. Optionally add curriculum learning
python train_ray_curriculum.py

# 4. Fine-tune with imitation of baseline
# Collect baseline demos and add behavioral cloning loss

# 5. Evaluate against baselines
python -m soccer_twos.watch -m1 <your_checkpoint> -m2 ceia_baseline_agent
```

---

## Resources and References

1. **PPO Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. **Ray Tune Documentation**: [ray.io/tune](https://docs.ray.io/en/latest/tune/)
3. **Multi-Agent RL**: [Open-ended Learning Leads to Generally Capable Agents](https://arxiv.org/abs/2107.12808)
4. **Reward Shaping**: [Potential-based Reward Shaping](https://www.jmlr.org/papers/v37/devlin15.pdf)
5. **Curiosity-Driven Learning**: [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)

---

## Summary Table

| Technique | Difficulty | Training Time | Performance Gain | Implementation |
|-----------|-----------|---------------|-----------------|-----------------|
| Reward Shaping | ⭐⭐ | 1x | 30-50% | Already implemented |
| Curriculum Learning | ⭐⭐ | 1.5x | 20-40% | Modify training config |
| Self-Play | ⭐⭐⭐ | 2x | 40-60% | Use existing script + modifications |
| Imitation Learning | ⭐⭐⭐ | 1.5x | 50-70% | Requires demo collection |
| Attention Mechanisms | ⭐⭐⭐ | 1x | 10-30% | Modify network architecture |
| LSTM Networks | ⭐⭐ | 1x | 15-30% | Ray config change |
| Curiosity-Driven | ⭐⭐⭐ | 1.5x | 20-40% | Add curiosity module |
| PBT Hyperparameter Tuning | ⭐⭐ | 2x | 10-25% | Ray scheduler |

**Recommendation**: Start with **Strategy 5 + Self-Play + Curriculum** for best results in limited time.
