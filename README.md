# MATLAB PPO Reinforcement Learning Framework

This is a MATLAB-based reinforcement learning framework that implements Proximal Policy Optimization (PPO) algorithm and its multi-agent extension (MAPPO), supporting GPU acceleration and parallel computing, suitable for control system research and engineering applications.

> For Chinese version, please refer to [README_zh.md](README_zh.md)

## Algorithm Introduction

### PPO (Proximal Policy Optimization)
- Policy gradient-based reinforcement learning algorithm with clipped objective function to limit policy updates
- Supports both continuous and discrete action spaces
- Provides stable and efficient training process

### MAPPO (Multi-Agent PPO)
- Multi-agent extension of PPO
- Uses "centralized training with decentralized execution" architecture
- Provides solutions for tasks requiring agent collaboration

## Supported Environments

The framework includes four control scenarios:

1. **CartPole**
   - Classic control problem balancing an inverted pendulum
   - Example of discrete action space

2. **DC Motor Control**
   - Regulates motor speed to track target velocity
   - Example of continuous action space
   - Includes electrical and mechanical system dynamics

3. **AC Motor FOC Control**
   - Implements Field-Oriented Control for AC induction motors
   - Example of advanced industrial control problem
   - Involves complex coordinate transformations and dynamics

4. **Double Inverted Pendulum System**
   - Multi-agent cooperative control example
   - Demonstrates advantages of MAPPO algorithm
   - Two connected pendulums require coordinated control to maintain balance

## Quick Start

### Requirements
- MATLAB R2019b or later
- Deep Learning Toolbox
- Parallel Computing Toolbox (optional, for parallel data collection)
- GPU support (optional but recommended for faster training)

### Usage Steps

#### 1. Run Examples

```matlab
% Train CartPole controller
cd examples
train_cartpole  % Train model
test_cartpole   % Test trained model

% Train DC motor controller
train_dcmotor
test_dcmotor

% Train AC motor FOC controller
train_acmotor
test_acmotor

% Train multi-agent double pendulum controller
train_doublependulum
test_doublependulum
```

#### 2. Custom Configuration

```matlab
% Create and modify configuration
config = PPOConfig();
config.gamma = 0.99;             % Discount factor
config.epsilon = 0.2;           % Clip parameter
config.actorLearningRate = 3e-4; % Policy network learning rate
config.useGPU = true;            % Enable GPU acceleration

% Train with custom configuration
agent = PPOAgent(env, config);
agent.train();
```

#### 3. Save and Load Models

```matlab
% Save trained model
agent.save('my_trained_model.mat');

% Load model
agent = PPOAgent.load('my_trained_model.mat', env);
```

## Directory Structure

```
Matlab_PPO/
├── core/               # Core algorithm implementation
├── environments/       # Environment implementations
├── config/             # Configuration files
├── utils/              # Utility functions
├── examples/           # Example scripts
└── logs/               # Logs and model saving directory
```

## Documentation

- [TUTORIAL.md](TUTORIAL.md) - Detailed algorithm tutorial and implementation details
- API documentation available via MATLAB `help` command, e.g.:
  ```matlab
  help PPOAgent
  help Environment
  help DCMotorEnv
  ```

## Contribution

Welcome to submit improvements and code contributions via Issues and Pull Requests.

## License

MIT License
