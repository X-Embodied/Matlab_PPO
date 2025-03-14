# Matlab PPO 强化学习框架

这是一个基于MATLAB的强化学习框架，包含近端策略优化（Proximal Policy Optimization, PPO）算法及其多智能体扩展版本（MAPPO），支持GPU加速和并行计算，适用于控制系统研究和工程应用。

> 详细的算法原理、实现细节和高级功能请参考 [TUTORIAL.md](TUTORIAL.md)

## 算法简介

### PPO (近端策略优化)
- 基于策略梯度的强化学习算法，使用裁剪目标函数限制策略更新幅度
- 支持连续和离散动作空间
- 提供稳定、高效的训练过程

### MAPPO (多智能体PPO)
- PPO的多智能体扩展版本
- 采用"集中训练，分散执行"架构
- 为需要智能体协同合作的任务提供解决方案

## 支持的环境

框架内置四种控制场景：

1. **倒立摆 (CartPole)**
   - 经典控制问题，通过左右移动小车平衡竖直摆杆
   - 离散动作空间示例

2. **直流电机控制**
   - 调节电机转速以跟踪目标速度
   - 连续动作空间示例
   - 包含电气和机械系统动态模型

3. **交流电机FOC控制**
   - 实现交流感应电机的磁场定向控制
   - 高级工业控制问题示例
   - 涉及复杂的坐标变换和动态模型

4. **双倒立摆系统**
   - 多智能体协同控制示例
   - 展示MAPPO算法的优势
   - 两个连接的摆杆需要协同控制以保持平衡

## 快速入门

### 安装要求
- MATLAB R2019b 或更高版本
- Deep Learning Toolbox
- Parallel Computing Toolbox (可选，用于并行数据收集)
- GPU 支持 (可选，但推荐用于加速训练)

### 使用步骤

#### 1. 运行示例

```matlab
% 训练倒立摆控制器
cd examples
train_cartpole  % 训练模型
test_cartpole   % 测试训练结果

% 训练直流电机控制器
train_dcmotor
test_dcmotor

% 训练交流电机FOC控制器
train_acmotor
test_acmotor

% 训练多智能体双倒立摆控制器
train_doublependulum
test_doublependulum
```

#### 2. 自定义配置

```matlab
% 创建和修改配置
config = PPOConfig();
config.gamma = 0.99;             % 折扣因子
config.epsilon = 0.2;            % 裁剪参数
config.actorLearningRate = 3e-4; % 策略网络学习率
config.useGPU = true;            % 启用GPU加速

% 使用自定义配置训练
agent = PPOAgent(env, config);
agent.train();
```

#### 3. 保存和加载模型

```matlab
% 保存训练好的模型
agent.save('my_trained_model.mat');

% 加载模型
agent = PPOAgent.load('my_trained_model.mat', env);
```

## 目录结构

```
Matlab_PPO/
├── core/               # 核心算法实现
├── environments/       # 环境实现
├── config/             # 配置文件
├── utils/              # 工具函数
├── examples/           # 示例脚本
└── logs/               # 日志和模型保存目录
```

## 文档

- [TUTORIAL.md](TUTORIAL.md) - 详细的算法教程和实现细节
- API文档可通过MATLAB的`help`命令获取，例如：
  ```matlab
  help PPOAgent
  help Environment
  help DCMotorEnv
  ```

## 贡献

欢迎通过Issue和Pull Request提交改进建议和贡献代码。

## 许可证

MIT许可证
