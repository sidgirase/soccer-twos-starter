# Multi-Strategy Soccer Agent Training - Complete Project Guide

## Quick Overview

This project trains competitive multi-agent soccer players using 5 different reward shaping strategies. Your goal is to:

1. **Phase 1**: Train agents that beat the random agent (9/10 matches)
2. **Phase 2**: Train agents that beat the baseline agent (9/10 matches)  
3. **Phase 3** (Stretch): Beat the TA's competitive agent

---

## Project Structure

```
.
├── train_strategies.py              # Main training script (use this!)
├── reward_shaping.py                # 5 reward strategies
├── reward_shaping_env.py            # Environment wrapper
├── evaluate_agents.py               # Evaluation script
├── ADVANCED_RL_TECHNIQUES.md        # Advanced methods
├── README_PACE_MULTITRAINING.md     # PACE cluster guide
├── ray_results/                     # Training outputs
└── example_player_agent/            # Random baseline
```

---

## Step 1: Local Testing (Your Machine)

### 1.1 Verify Environment Setup

```bash
# Make sure you completed README.md steps 1-8
python example_random_players.py  # Should show 4v4 match with random agents
```

### 1.2 Quick Local Training (Optional - for testing)

```bash
# Train strategy 1 locally for 100k timesteps (takes ~5-10 minutes)
python train_strategies.py --strategy 1 --workers 2
```

If this works, you're ready for PACE!

---

## Step 2: Training on PACE Cluster

### 2.1 Connect and Setup

```bash
# Connect to PACE
ssh <your_gatech_username>@login-ice.pace.gatech.edu

# (First time only) Clone and setup:
cd ~/scratch
git clone https://github.com/your-github-user/soccer-twos-starter.git
cd soccer-twos-starter

module purge
module load anaconda3/2023.03

conda create --name soccertwos python=3.8 -y
conda activate soccertwos

pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge
pip install -r requirements.txt

pip install protobuf==3.20.3 pydantic==1.10.13
```

### 2.2 Option A: Interactive Training (Faster Feedback)

```bash
# Request interactive session (4 hours, 32 cores)
salloc -N 1 -c 32 --mem=64G -t 4:00:00 -p ice-cpu

# Wait for allocation...

# Once allocated to a node:
module load anaconda3/2023.03
cd ~/scratch/soccer-twos-starter
conda activate soccertwos

# Train a single strategy
python train_strategies.py --strategy 1 --workers 8

# Or train all 5 sequentially
python train_strategies.py --all --workers 8

# Exit when done
exit
```

**Estimated time**: ~2 hours for 1 strategy, ~10 hours for all 5

### 2.3 Option B: Batch Job Training (Better for Long Runs)

```bash
# Create batch script
cat > submit_all_strategies.sh <<'EOF'
#!/bin/bash

for strategy in 1 2 3 4 5; do
  cat > strategy_${strategy}.sh <<EOFJ
#!/bin/bash
#SBATCH --job-name=soccer_strat_${strategy}
#SBATCH -N 1 -c 32
#SBATCH --mem=64G
#SBATCH -t 18:00:00
#SBATCH -p ice-cpu
#SBATCH -o logs/strategy_${strategy}_%j.out

module purge
module load anaconda3/2023.03

cd ~/scratch/soccer-twos-starter
conda activate soccertwos

echo "Starting Strategy ${strategy} at $(date)"
python train_strategies.py --strategy ${strategy} --workers 8
echo "Finished Strategy ${strategy} at $(date)"
EOFJ

  chmod +x strategy_${strategy}.sh
  sbatch strategy_${strategy}.sh
  sleep 2
done

rm strategy_*.sh
echo "All strategies submitted!"
EOF

chmod +x submit_all_strategies.sh
./submit_all_strategies.sh
```

**Monitor progress**:
```bash
# Check job status
squeue -u <your_username>

# Watch strategy 1 output
tail -f logs/strategy_1_*.out
```

### 2.4 Retrieve Results

```bash
# After training completes:

# Copy to home directory (persistent storage)
cp -r ~/scratch/soccer-twos-starter/ray_results ~/soccer_results_backup

# Download to your machine (from your machine):
scp -r <username>@login-ice.pace.gatech.edu:~/soccer_results_backup ~/Downloads/soccer_results

# Or view via tensorboard (SSH tunnel)
# Terminal 1 - On PACE:
tensorboard --logdir=~/scratch/soccer-twos-starter/ray_results --port=6006

# Terminal 2 - On your machine:
ssh -L 6006:localhost:6006 <username>@login-ice.pace.gatech.edu

# Then visit: http://localhost:6006
```

---

## Step 3: Understanding the 5 Strategies

All strategies use **reward shaping** - adding bonuses/penalties to environment rewards:

### Strategy 1: Ball Possession
- **Focus**: Keep ball, maintain close control
- **Good for**: Possession-based soccer style
- **When to use**: Defensive, keep-possession strategy

### Strategy 2: Ball Interception  
- **Focus**: Aggressive defense, quick ball recovery
- **Good for**: Defensive pressing
- **When to use**: High-pressure defense

### Strategy 3: Offensive Positioning
- **Focus**: Position for attacks, advance on goal
- **Good for**: Aggressive attacking
- **When to use**: Scoring-focused strategy

### Strategy 4: Defensive Blocking
- **Focus**: Block shots, defend goal
- **Good for**: Goalkeeper/defender positioning
- **When to use**: Defensive/blocking strategy

### Strategy 5: Coordinated Teamplay
- **Focus**: Pass, support teammates, spacing
- **Good for**: Team coordination
- **When to use**: Best overall (recommended)

**Recommendation**: Start with **Strategy 5**, then try others to compare.

---

## Step 4: Comparing Strategies

After training all 5:

```bash
# Analyze results
python evaluate_agents.py --checkpoint ray_results/PPO_Strategy_1/...

# Compare training curves:
# Look at ray_results/ structure:
ray_results/
  ├── PPO_Strategy_1/
  │   ├── PPO_Soccer_*/
  │   │   └── events.out.tfevents...  # TensorBoard logs
  │   └── ...
  ├── PPO_Strategy_2/
  └── ...
```

**Key metrics to compare**:
- `episode_reward_mean` - Average reward per episode (higher is better)
- `policy_loss` - Policy gradient loss (lower is better)
- `vf_loss` - Value function loss (lower is better)
- `episode_len_mean` - Average episode length

**Which strategy performed best?**
- Look at final `episode_reward_mean`
- Check convergence speed
- Check stability (variance over time)

---

## Step 5: Evaluation Against Baselines

### 5.1 Test Against Random Agent

```bash
# Load your best checkpoint and test
python -m soccer_twos.watch -m1 ray_results/PPO_Strategy_5/PPO_Soccer_*/checkpoint_final
```

**Success criteria**: Win 9/10 matches

### 5.2 Test Against Baseline Agent

First, download the baseline:
```bash
# From https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view
# Extract ceia_baseline_agent.zip to project directory

python -m soccer_twos.watch -m1 ray_results/PPO_Strategy_5/.../checkpoint_final -m2 ceia_baseline_agent
```

**Success criteria**: Win 9/10 matches

### 5.3 Test Against TA Agent (When Released)

```bash
python -m soccer_twos.watch -m1 your_agent -m2 ta_competitive_agent
```

---

## Step 6: Advanced Improvements (Optional)

See `ADVANCED_RL_TECHNIQUES.md` for:

1. **Curriculum Learning** - Gradually increase difficulty
2. **Self-Play** - Agent plays against itself
3. **Imitation Learning** - Learn from baseline agent
4. **Attention Mechanisms** - Focus on relevant entities
5. **Curiosity-Driven Learning** - Explore novel states
6. **Network Architectures** - LSTM, Dueling networks

**Quick wins** (try these for significant improvements):
```bash
# Self-play training
python train_ray_selfplay.py

# Curriculum learning  
python train_ray_curriculum.py

# These often provide 20-40% performance boost
```

---

## Step 7: Project Submission

### Requirements Checklist

- [ ] Multiple agents trained (at least 2-3 with different strategies)
- [ ] Each agent: inherits `soccer_twos.AgentInterface`, implements `act()` method
- [ ] Each agent: filled `README.md` with name, authors, description
- [ ] Each agent: compressed as `.zip` file
- [ ] Training plots for each agent (reward vs. steps)
- [ ] Comparison plot across all agents
- [ ] Report (1-2 pages):
  - Algorithm used (PPO + reward shaping)
  - Theoretical background
  - Your specific modifications (which strategies)
  - Results and comparison
  - Why each modification helped/hurt performance

### Report Structure

```markdown
# Multi-Strategy Soccer Agent Training Report

## 1. Methodology
- Algorithm: Proximal Policy Optimization (PPO)
- Library: Ray RLlib with PyTorch
- Key innovation: 5-strategy reward shaping approach

## 2. Reward Strategies Tested
1. Ball Possession Focus
2. Ball Interception Focus  
3. Offensive Positioning
4. Defensive Blocking
5. Coordinated Teamplay (best)

## 3. Experimental Results
[Include training curves for all agents]
[Include comparison table]

## 4. Analysis
- Strategy 5 achieved X% win rate vs baseline
- Convergence time: Y hours
- Why this worked: [technical reasoning]

## 5. Conclusion
Best strategy: Coordinated Teamplay
- Won 9/10 vs random agent ✓
- Won 8/10 vs baseline agent (close!)
- Recommendation: Combine with self-play for better results
```

### File Submission Structure

```
submission/
├── report.pdf (1-2 pages)
├── agent_strategy_5.zip
│   ├── agent.py
│   ├── model.py (if applicable)
│   ├── README.md
│   └── checkpoint.pth
├── agent_selfplay.zip
│   └── ...
├── training_plots/
│   ├── strategy_1_training_curve.png
│   ├── strategy_2_training_curve.png
│   ├── ...
│   ├── strategy_5_training_curve.png
│   └── comparison_all_strategies.png
└── checkpoint_paths.txt  # Links to best checkpoints
```

---

## Troubleshooting

### Training doesn't start
```bash
# Check Python version
python --version  # Must be 3.8.x

# Verify Ray installation
python -c "import ray; print(ray.__version__)"  # Should be 1.4.0

# Test environment
python example_random_players.py
```

### Out of memory on PACE
```bash
# Reduce workers
python train_strategies.py --strategy 1 --workers 4

# Or reduce batch size in config
# In train_strategies.py: "batch_size": 2048  # reduced from 4096
```

### PACE login issues
```bash
# Verify VPN is connected
# Check PACE status: https://pace.gatech.edu

# Try alternative login:
ssh <username>@ice123.pace.gatech.edu  # Direct node access
```

### Results not saving
```bash
# Check scratch directory has space
du -h ~/scratch

# Move old results
mv ~/scratch/soccer-twos-starter/ray_results ~/soccer_results_old
```

---

## Timeline Estimate

| Phase | Task | Time |
|-------|------|------|
| 1 | Setup on PACE | 30 min |
| 2 | Train Strategy 5 | 2-4 hrs |
| 3 | Train other strategies | 8-10 hrs |
| 4 | Evaluate & compare | 1 hr |
| 5 | Advanced improvements | 4-8 hrs |
| 6 | Write report | 1-2 hrs |
| **Total** | | **18-26 hrs** |

**Note**: Most time is automated training. You only need to monitor and wait.

---

## Next Steps After This Project

1. **Multi-objective RL**: Train for multiple goals (score + possession)
2. **Hierarchical RL**: High-level strategy + low-level control
3. **Transfer Learning**: Apply soccer skills to other games
4. **Distributed Training**: Scale to 100+ workers on larger clusters
5. **Production**: Deploy trained agent as service

---

## Additional Resources

- [Soccer-Twos Environment](https://github.com/bryanoliveira/soccer-twos-env)
- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [Multi-Agent RL Survey](https://arxiv.org/abs/1908.03963)
- [PACE Cluster Guide](https://pace.gatech.edu/accessing-the-cluster)

---

## Questions?

- Check ADVANCED_RL_TECHNIQUES.md for implementation details
- Review README_PACE_MULTITRAINING.md for PACE-specific issues
- Check existing training scripts for examples

Good luck! 🎮⚽🤖
