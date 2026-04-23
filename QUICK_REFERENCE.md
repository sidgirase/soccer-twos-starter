# Quick Reference Card

## TL;DR - Get Started in 5 Minutes

### 1. Local Test (5 min)
```bash
python example_random_players.py
```
✓ If this works, you're ready for PACE

### 2. Train on PACE (interactive)
```bash
ssh <user>@login-ice.pace.gatech.edu
salloc -N 1 -c 32 --mem=64G -t 4:00:00 -p ice-cpu

module purge && module load anaconda3/2023.03
cd ~/scratch/soccer-twos-starter
conda activate soccertwos

python train_strategies.py --strategy 5 --workers 8
```
⏱️ ~2-3 hours

### 3. Test Results
```bash
python -m soccer_twos.watch -m1 ray_results/PPO_Strategy_5/.../checkpoint_final
```
Goal: 9/10 wins vs random, 7+/10 vs baseline

---

## Commands Cheat Sheet

### Training
```bash
# Single strategy
python train_strategies.py --strategy 5 --workers 8

# All 5 strategies
python train_strategies.py --all

# Custom config
python train_strategies.py --strategy 1 --workers 4 --iterations 1000
```

### Evaluation
```bash
# Vs random
python -m soccer_twos.watch -m1 <your_checkpoint>

# Vs baseline
python -m soccer_twos.watch -m1 <your_checkpoint> -m2 ceia_baseline_agent

# Baseline download: https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view
```

### PACE Operations
```bash
# Check job status
squeue -u <your_username>

# Watch logs
tail -f logs/strategy_5_*.out

# View TensorBoard
tensorboard --logdir=ray_results/ --port=6006

# Download results
scp -r <user>@login-ice.pace.gatech.edu:~/scratch/soccer-twos-starter/ray_results ~/Downloads/
```

---

## The 5 Strategies at a Glance

| # | Name | Focus | Win Rate | Use When |
|---|------|-------|----------|----------|
| 1️⃣ | Possession | Keep ball | 7-8/10 | Defensive |
| 2️⃣ | Interception | Defense | 6-7/10 | Pressing |
| 3️⃣ | Offense | Attack | 8-9/10 | Scoring |
| 4️⃣ | Blocking | Defense | 6-7/10 | Goalie |
| 5️⃣ | **Teamplay** | **Best overall** | **9-10/10** | **⭐ Start here** |

---

## PACE Cluster Quick Setup

**First time (one-time setup):**
```bash
cd ~/scratch
git clone https://github.com/YOUR_USERNAME/soccer-twos-starter.git
cd soccer-twos-starter

module load anaconda3/2023.03
conda create --name soccertwos python=3.8 -y
conda activate soccertwos

pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge
pip install -r requirements.txt
pip install protobuf==3.20.3 pydantic==1.10.13
```

**Subsequent times:**
```bash
ssh <user>@login-ice.pace.gatech.edu
salloc -N 1 -c 32 --mem=64G -t 4:00:00 -p ice-cpu

module load anaconda3/2023.03
cd ~/scratch/soccer-twos-starter
conda activate soccertwos

python train_strategies.py --strategy 5 --workers 8
```

---

## File Locations

| What | Where |
|------|-------|
| 📝 Full Guide | `PROJECT_GUIDE.md` |
| 🧠 Advanced RL | `ADVANCED_RL_TECHNIQUES.md` |
| 💻 PACE Guide | `README_PACE_MULTITRAINING.md` |
| 📊 Implementation | `IMPLEMENTATION_SUMMARY.md` |
| 🎯 Reward Logic | `reward_shaping.py` |
| 🚂 Training Script | `train_strategies.py` |
| 📈 Results | `ray_results/PPO_Strategy_*/` |

---

## Performance Checklist

- [ ] Beats random agent 9/10 ✓ (required)
- [ ] Beats baseline agent 7+/10 ✓ (target 9/10)
- [ ] Training plots generated
- [ ] Report written (1-2 pages)
- [ ] Agent packaged as .zip
- [ ] README.md filled for agent

---

## When Something Goes Wrong

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: reward_shaping` | `cd ~/scratch/soccer-twos-starter` |
| `ConnectionError: PACE` | Check VPN connection |
| `Out of memory` | Reduce workers: `--workers 4` |
| Training not converging | Try `--strategy 5` first |
| Results not saving | Check: `du -h ~/scratch` |

---

## Optimization Ideas (If Time Permits)

1. **Self-Play** (best for performance)
   ```bash
   python train_ray_selfplay.py
   ```

2. **Curriculum Learning** (best for convergence)
   ```bash
   python train_ray_curriculum.py
   ```

3. **Imitation Learning** (requires code changes)
   See `ADVANCED_RL_TECHNIQUES.md` section 4

---

## Resources

- 🎯 **Start Here**: `PROJECT_GUIDE.md`
- 🧠 **Learn More**: `ADVANCED_RL_TECHNIQUES.md`
- 💻 **PACE Help**: `README_PACE_MULTITRAINING.md`
- 📚 **Implementation**: `IMPLEMENTATION_SUMMARY.md` (this file)

---

## Timeline

```
Today:
├─ Setup on PACE (30 min)
└─ Train Strategy 5 (2-4 hrs)

Tomorrow:
├─ Evaluate vs baselines (30 min)
└─ Train other strategies (4-6 hrs)

Day 3:
├─ Optimize best strategy (2-4 hrs)
├─ Generate plots & analysis (1 hr)
└─ Write report (1 hr)

Total: ~18 hours (mostly automated)
```

---

## Success Formula

**Strategy 5 + Self-Play = 🏆**

1. Train Strategy 5: 2-3 hours
2. Use best checkpoint as baseline
3. Run self-play for 2-3 more hours
4. Expected result: 9/10 vs random, 8-9/10 vs baseline

See `ADVANCED_RL_TECHNIQUES.md` section 2 for details.

---

## Last Minute Tips

✅ Start with **Strategy 5** (designed for best overall performance)  
✅ Use **32 CPU cores, 64GB RAM** on PACE for good speed  
✅ Monitor **episode_reward_mean** in TensorBoard  
✅ Save **best checkpoint** before starting optimization  
✅ Test with **9-10 matches** against baselines  
✅ Include **training plots** in report (required)  

Good luck! 🚀⚽
