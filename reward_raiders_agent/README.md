# Reward Raiders Agent

**Agent name:** Reward_Raiders_Balanced

**Author(s):** Kun-Lin Hsieh (khsieh37@gatech.edu), Siddhesh Girase (sgirase3@gatech.edu)

## Description

An agent trained with PPO via multi-agent self-play using Ray RLLib, utilizing a Balanced Strategy reward structure. This agent combines proximity control with a slight time penalty to encourage rapid goal-scoring, forcing agents to learn spatial coordination and passing lanes.

This was the final submission agent in the CS 8803 DRL final project, achieving competitive performance against the random baseline with 6 wins, 3 ties, and 1 loss over 10 matches (8 goals scored vs 2 conceded).

## Training Strategy

- **Algorithm:** Proximal Policy Optimization (PPO)
- **Training Method:** Multi-agent self-play using Ray RLLib
- **Reward Structure:** Balanced approach combining:
  - Proximity rewards for being near the ball
  - Time penalty to encourage rapid action
  - Coordination incentives for team play
- **Hyperparameters:** Tuned for stability and convergence

## Performance

- **vs Random Agent:** 6W - 3T - 1L (8 goals : 2 goals)
- **vs CEIA Baseline:** 0W - 0T - 10L (27 goals : 178 goals)
- **vs TA Agent:** 0W - 1T - 9L (6 goals : 94 goals)

## Key Insight

Despite initial attempts at more complex reward shaping approaches, this simpler balanced strategy proved to be the most robust competitor, demonstrating that less is often more in reward shaping.
