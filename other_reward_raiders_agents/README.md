# Other Reward Raiders Agents

**Experimental Agents:** Role-Based MARL, Velocity-Based, Conventional Rewards Baseline

**Author(s):** Kun-Lin Hsieh (khsieh37@gatech.edu), Siddhesh Girase (sgirase3@gatech.edu)

# Run Instructions

Uncomment the checkpoint paths as necessary!

## Description

This directory contains experimental agent implementations from the CS 8803 DRL final project that explored various advanced reward shaping and multi-agent coordination strategies. While these agents represent theoretically sound approaches, they ultimately underperformed compared to simpler heuristics.

## Experimental Approaches

### 1. Role-Based MARL Agent

**Concept:** Assign distinct brains (Attacker/Defender Policies) using multi-agent role assignment inspired by RoboCup domain research.

**Approach:**
- Extract exact coordinates from environment
- Provide static positional rewards
- Multi-agent policy mapping for specialization

**Hyperparameters:** LR 3e-4, Train Batch Size 4000, Entropy 0.01

**Result:** **FAILED** - Massive reward hacking. Agents realized that scoring ended the game and stopped the free positional points, then simply stood still behind the ball to farm infinite points.

---

### 2. Velocity-Based Agent

**Concept:** Implement pure Relative Velocity Gradients inspired by DeepMind's simulated football environments to avoid specification gaming.

**Approach:**
- Strip static positional rewards
- Implement continuous frame-by-frame penalties (-0.0002 per frame)
- Reward agents only when velocity vector points toward ball or ball velocity points toward enemy goal
- Locomotion-based control similar to real-world soccer physics

**Result:** **FAILED** - While it solved the spinning issue, agents became hyper-aggressive but entirely uncoordinated. They essentially chased the ball wildly, failing to work as a team.

---

### 3. Conventional Rewards Baseline

**Concept:** Classic reward structure combining time penalty, proximity rewards, and ball momentum gradient.

**Approach:**
- Tiny time penalty per step
- Gentle proximity reward (within 2.0 units of ball)
- Passive gradient pulling ball toward enemy net

**Result:** **FAILED** - Agent learned too slowly and was overly passive, failing to beat the random baseline in the limited training time window.

---

## Key Learnings

These experimental agents demonstrate critical lessons in Deep Reinforcement Learning:

1. **Specification Gaming:** Agents will exploit any reward structure loophole, especially static positional rewards
2. **Infrastructure Matters:** Complex reward shaping requires extensive training time (24+ hours) and rapid iteration capability
3. **Simplicity Wins:** Theoretically sound but complex approaches often underperformed the simpler Balanced Strategy
4. **Logging Critical:** Real-time monitoring of training progress is essential—unbuffered logs on remote clusters revealed that agents often converged within the first hour

## Future Directions

These experimental agents provide a foundation for future work exploring:
- **Curriculum Self-Play:** Rather than static reward shapes, have agents play against older checkpoints of themselves
- **Emergent Coordination:** Allow natural arms race dynamics to force learned teamwork without manual reward engineering
- **Iterative Refinement:** With better computational infrastructure, rapid iteration on these approaches could yield better results
