# Running Multiple Soccer-Twos RL Training Jobs on PACE (Georgia Tech)

This guide provides everything you need to request compute resources on the PACE cluster and run multiple RL agent training jobs efficiently. Unlike the original VLM project, this workload is **CPU-intensive** rather than GPU-intensive, allowing you to spawn many parallel training jobs.

---

## Step 1: Log in to PACE
SSH into the PACE interactive login node from your terminal:
```bash
ssh <your_gatech_username>@login-ice.pace.gatech.edu
```

*Note: You must be connected to the campus VPN if you are off-campus.*

---

## Step 2: Request a CPU-Optimized Interactive Session
Instead of requesting a GPU, request a **high-core CPU node** for interactive testing and development.
Run this SLURM command to request 32 CPU cores and 64GB of RAM for 2 hours:

```bash
salloc -N 1 -c 32 --mem=64G -t 2:00:00 -p ice-cpu
```

Wait until you see your command prompt change from `login-ice` to something like `ice123` (this means you have been allocated a node).

*Note: If you prefer even more cores for larger experiments, you can request up to 64 cores:*
```bash
salloc -N 1 -c 64 --mem=128G -t 18:00:00 -p ice-cpu
```

*Alternative: For faster allocation, try smaller requests on different partitions:*
```bash
# Smaller allocation on pace-cpu partition
salloc -N 1 -c 16 --mem=32G -t 10:00:00 -p pace-cpu

# Or try coc-cpu partition
salloc -N 1 -c 16 --mem=32G -t 10:00:00 -p coc-cpu

# Minimal allocation for testing
salloc -N 1 -c 8 --mem=16G -t 4:00:00 -p ice-cpu
```

---

## Step 3: Load Necessary Modules
PACE uses a module system. Load Python (Anaconda) to set up your environment:

```bash
# Clear any loaded modules
module purge

# Load Anaconda
module load anaconda3/2023.03
```

---

## Step 4: Clone Your Project and Navigate to Workspace
Navigate to your scratch directory (better space and speed) and clone the repository:

```bash
# Go to your scratch directory
cd ~/scratch

# Clone your repo
git clone https://github.com/sidgirase/soccer-twos-starter.git

cd soccer-twos-starter/
```

---

## Step 5: Create and Activate Conda Environment
Create and activate a Python 3.8 virtual environment:

```bash
conda create --name soccertwos python=3.8 -y

conda activate soccertwos
```

---

## Step 6: Downgrade Build Tools for Compatibility
```bash
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4

pip cache purge
```

---

## Step 7: Install Requirements
```bash
pip install -r requirements.txt
```

---

## Step 8: Fix Protobuf and Pydantic Compatibility
```bash
pip install protobuf==3.20.3

pip install pydantic==1.10.13
```

---

## Step 9: Run a Test Training Job
Before running multiple jobs, verify that your environment works with a quick test run:

```bash
# Run a quick training job to verify the setup
python example_ray_ppo_sp_still.py --stop-iters=1
```

If this completes without errors, your environment is ready.

*Troubleshooting: If you get "ModuleNotFoundError: No module named 'ray'"*
```bash
# Make sure you're using Python 3.8
python --version  # Should show: Python 3.8.x

# If not, recreate the environment with Python 3.8:
conda create --name soccertwos python=3.8 -y
conda activate soccertwos

# Reinstall requirements
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge
pip install -r requirements.txt
pip install protobuf==3.20.3 pydantic==1.10.13

# Or install Ray specifically (should work with Python 3.8)
pip install ray==1.4.0 ray[tune]==1.4.0 ray[rllib]==1.4.0
```

---

## Step 10: Running Multiple Training Jobs

### Option A: Interactive Parallel Execution
You can run multiple training jobs in parallel within a single `salloc` session using Ray Tune's built-in distributed training.

For example, to run multiple curriculum-based training configurations:
```bash
# Run multiple experiments concurrently
python train_ray_curriculum.py &
python example_ray_team_vs_random.py &
python train_ray_selfplay.py &

# Wait for all background jobs to complete
wait
```

Ray Tune will automatically distribute the work across your allocated CPU cores.

### Option B: Submit Multiple Batch Jobs
For long-running training, you can submit multiple `sbatch` scripts simultaneously. Each job will queue independently and run when resources become available.

Create `train_job_1.sh`:
```bash
nano train_job_1.sh
```

Paste this template:
```bash
#!/bin/bash
#SBATCH --job-name=soccer_training_1
#SBATCH -N 1 -c 32
#SBATCH --mem=64G
#SBATCH -t 18:00:00
#SBATCH -p ice-cpu
#SBATCH -o logs/soccer_train_1_%j.out

module purge
module load anaconda3/2023.03

conda activate soccertwos
cd ~/scratch/soccer-twos-starter

python train_ray_curriculum.py
```

Save it (`Ctrl+O`, `Enter`, `Ctrl+X`), then repeat for other training scripts:

```bash
# Create Job 2
nano train_job_2.sh
# (Paste same content but change --job-name to soccer_training_2 and python command to example_ray_team_vs_random.py)

# Create Job 3
nano train_job_3.sh
# (Paste same content but change --job-name to soccer_training_3 and python command to train_ray_selfplay.py)

# Create the logs directory
mkdir -p logs

# Submit all jobs
sbatch train_job_1.sh
sbatch train_job_2.sh
sbatch train_job_3.sh
```

Check on all your jobs:
```bash
squeue -u <your_gatech_username>
```

### Connect to Running Jobs
To see which compute node your job is running on and connect to it:

```bash
# Check job details including node allocation
squeue -u <your_gatech_username> -o "%.8i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# SSH into the compute node (replace NODE_NAME with the actual node from above)
ssh <your_gatech_username>@NODE_NAME.pace.gatech.edu

# Example: ssh sgirase3@ice123.pace.gatech.edu
```

*Note: You can only SSH into nodes where you have active jobs running.*

---

## Step 11: Monitoring and Results

### Watch Live Output
```bash
tail -f logs/soccer_train_1_<job_id>.out
```

### Check Ray Results
Ray Tune automatically saves results to `ray_results/`. To monitor training in real-time:

```bash
# Inside your allocated session, open a new terminal window and:
cd ~/scratch/soccer-twos-starter
tensorboard --logdir=ray_results/
```

Then visit `http://localhost:6006` (or port 6007 if 6006 is in use).

### Retrieve Results After Job Completion
```bash
# Copy results from scratch to your home directory
cp -r ~/scratch/soccer-twos-starter/ray_results/ ~/results_backup/
```

---

## Step 12: Advanced: Bulk Submit Multiple Configurations

To run a grid of hyperparameter combinations, create a master submission script:

```bash
nano submit_all_experiments.sh
```

```bash
#!/bin/bash

CONFIGS=(
  "train_ray_curriculum.py"
  "train_ray_selfplay.py"
  "example_ray_team_vs_random.py"
  "example_ray_ppo_sp_still.py"
)

for config in "${CONFIGS[@]}"; do
  # Create a job script for each config
  cat > temp_job.sh <<EOF
#!/bin/bash
#SBATCH --job-name=soccer_$(basename $config .py)
#SBATCH -N 1 -c 32
#SBATCH --mem=64G
#SBATCH -t 18:00:00
#SBATCH -p ice-cpu
#SBATCH -o logs/soccer_$(basename $config .py)_%j.out

module purge
module load anaconda3/2023.03

conda activate soccertwos
cd ~/scratch/soccer-twos-starter

python $config
EOF

  chmod +x temp_job.sh
  sbatch temp_job.sh
  sleep 1  # Small delay to avoid overloading the scheduler
done

rm temp_job.sh
echo "All training jobs submitted!"
```

Make it executable and run it:
```bash
chmod +x submit_all_experiments.sh
./submit_all_experiments.sh
```

---

## Resource Guidelines

- **Per Job**: 32 CPU cores with 64GB RAM is sufficient for most RL training workloads
- **Max Cores**: PACE ice-cpu partition allows up to 64 cores per job
- **Time Limit**: Set to 18 hours (max available on ice-cpu partition); adjust based on your experiment
- **Memory per Core**: ~2GB per core is a safe ratio for this workload
- **Partitions Available**: `ice-cpu` (default, 18h max), `pace-cpu` (18h max), `coc-cpu` (18h max)

---

## Tips for Efficient Multi-Job Training

1. **Stagger Job Submissions**: If submitting many jobs, add small delays to avoid overwhelming the scheduler
2. **Monitor Resource Usage**: Use `htop` on your allocated node to ensure Ray is distributing work across cores
3. **Use Ray Tune's parallelization**: Ray can automatically parallelize trials; set `num_samples` in your config
4. **Checkpoint Frequently**: Ray auto-saves checkpoints; you can resume interrupted jobs
5. **Log Everything**: Keep detailed logs in `logs/` directory for later analysis

---

## Troubleshooting

**Issue**: "No resources available"  
**Solution**: Reduce CPU request (e.g., -c 16 instead of -c 32) or increase time limit

**Issue**: Ray worker crashes  
**Solution**: Increase memory per CPU: try `--mem-per-cpu=4G` instead of relying on total `--mem`

**Issue**: Out of disk space on scratch  
**Solution**: Move old results: `mv ray_results/* ~/results_old/` and sync to persistent storage

**Issue**: SSH connection drops during long job  
**Solution**: Always use `sbatch` for runs longer than 2 hours instead of interactive `salloc`

**Issue**: "Invalid partition specification"  
**Solution**: Run `sinfo -o "%20P %.5a %.10l %.6D"` to see available partitions. Use `-p ice-cpu` for CPU workloads or `-p pace-cpu` as an alternative

**Issue**: "Could not find a version that satisfies the requirement ray==1.4.0"  
**Solution**: This repo requires Python 3.8. Make sure you're using `conda create --name soccertwos python=3.8 -y` instead of the default Python version

---

## Next Steps

- Modify training scripts to accept hyperparameter arguments for grid searches
- Use Ray's experiment tracking features for systematic hyperparameter tuning
- Set up automated post-processing of results from `ray_results/`
