# Soccer-Twos Starter Kit

Example training/testing scripts for the Soccer-Twos environment. This starter code is modified from the example code provided in https://github.com/bryanoliveira/soccer-twos-starter.

Environment-level specification code can be found at https://github.com/bryanoliveira/soccer-twos-env, which may also be useful to reference.

## Requirements

- Python 3.8
- See [requirements.txt](requirements.txt)

## Usage

### 1. Fork this repository

git clone https://github.com/your-github-user/soccer-twos-starter.git

cd soccer-twos-starter/

### 2. Create and activate conda environment
conda create --name soccertwos python=3.8 -y

conda activate soccertwos

### 3. Downgrade build tools for compatibility
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4

pip cache purge

### 4. Install requirements
pip install -r requirements.txt

### 5. Fix protobuf and pydantic compatibility
pip install protobuf==3.20.3

pip install pydantic==1.10.13

### 5. Run `python example_random.py` to watch a random agent play the game
python example_random_players.py

### 6. Train using any of the example scripts
python example_ray_ppo_sp_still.py

python example_ray_team_vs_random.py

etc.

## 🚀 Multi-Strategy Agent Training (Final Project)

This project includes a comprehensive **5-strategy reward shaping approach** for training competitive soccer agents. See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for complete instructions.

### Quick Start

**Local training (testing)**:
```bash
python train_strategies.py --strategy 1 --workers 2
```

**PACE cluster training (recommended)**:
```bash
# See README_PACE_MULTITRAINING.md for detailed PACE instructions

# Train a single strategy
python train_strategies.py --strategy 1 --workers 8

# Train all 5 strategies
python train_strategies.py --all --workers 8
```

### The 5 Strategies

| ID | Strategy | Focus | Best For |
|---|----------|-------|----------|
| 1 | Ball Possession | Close control, possession | Possession-based play |
| 2 | Ball Interception | Aggressive defense | Defensive pressing |
| 3 | Offensive Positioning | Attack/goal proximity | Scoring |
| 4 | Defensive Blocking | Shot blocking | Defense/goalkeeping |
| 5 | Coordinated Teamplay | Passes, team spacing | Overall performance ✓ |

**Recommendation**: Start with Strategy 5, which combines team coordination with smart positioning.

### Training On PACE

```bash
# Setup (first time only)
cd ~/scratch
git clone https://github.com/your-username/soccer-twos-starter.git
cd soccer-twos-starter

module load anaconda3/2023.03
conda create --name soccertwos python=3.8 -y
conda activate soccertwos

pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge
pip install -r requirements.txt
pip install protobuf==3.20.3 pydantic==1.10.13

# Interactive training
salloc -N 1 -c 32 --mem=64G -t 4:00:00 -p ice-cpu
python train_strategies.py --strategy 5 --workers 8
exit
```

See [README_PACE_MULTITRAINING.md](README_PACE_MULTITRAINING.md) for batch job submission and [ADVANCED_RL_TECHNIQUES.md](ADVANCED_RL_TECHNIQUES.md) for advanced improvements.

## Agent Packaging

To receive full credit on the assignment and ensure the teaching staff can properly compile your code, you must follow these instructions:

- Implement a class that inherits from `soccer_twos.AgentInterface` and implements an `act` method. Examples are located under the `example_player_agent/` or `example_team_agent/` directories.
- Fill in your agent's information in the `README.md` file (agent name, authors & emails, and description)
- Compress each agent's module folder as `.zip`.

*Submission Policy*: Students must submit multiple trained agents to meet all assignment requirements. In both the agent desription and the report, clearly identify which agent file corresponds to each evaluation criterion (e.g., Agent1 – policy performance, Agent2 – reward modification, Agent3 – imitation learning, etc.). 

Training plots are required for every agent that is discussed or submitted. Additionally, include a direct performance comparison across agents, such as overlaid learning curves, to support your analysis.


## Testing/Evaluating

Use the environment's rollout tool to test the example agent module:

`python -m soccer_twos.watch -m example_player_agent`

Similarly, you can test your own agent by replacing `example_player_agent` with the name of your agent directory.

The baseline agent is located here: [pre-trained baseline (download)](https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view?usp=sharing).
To examine the baseline agent, you must extract the `ceia_baseline_agent` folder to this project's folder. For instance you can run, 

`python -m soccer_twos.watch -m1 example_player_agent -m2 ceia_baseline_agent`

, to examine the random agent vs. the baseline agent.



