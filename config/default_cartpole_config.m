% default_cartpole_config.m
% 倒立摆环境的默认PPO配置文件

% 创建配置对象
config = PPOConfig();

% 环境配置
config.envName = 'CartPoleEnv';

% 网络配置
config.actorLayerSizes = [64, 64];      % Actor网络隐藏层大小
config.criticLayerSizes = [64, 64];     % Critic网络隐藏层大小

% 算法超参数
config.gamma = 0.99;                    % 折扣因子
config.lambda = 0.95;                   % GAE参数
config.epsilon = 0.2;                   % PPO裁剪参数
config.entropyCoef = 0.01;              % 熵正则化系数
config.vfCoef = 0.5;                    % 价值函数系数
config.maxGradNorm = 0.5;               % 梯度裁剪

% 优化器配置
config.actorLearningRate = 3e-4;        % Actor学习率
config.criticLearningRate = 3e-4;       % Critic学习率
config.momentum = 0.9;                  % 动量

% 训练配置
config.numIterations = 100;             % 训练迭代次数
config.batchSize = 64;                  % 批次大小
config.epochsPerIter = 4;               % 每次迭代的训练轮数
config.trajectoryLen = 200;             % 轨迹长度
config.numTrajectories = 10;            % 每次迭代收集的轨迹数量

% 硬件配置
config.useGPU = true;                   % 是否使用GPU

% 日志配置
config.logDir = 'logs/cartpole';        % 日志保存目录
config.evalFreq = 5;                    % 评估频率 (迭代次数)
config.numEvalEpisodes = 10;            % 评估时的回合数
config.saveModelFreq = 10;              % 保存模型频率 (迭代次数)

% 保存配置
saveDir = 'config';
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

configStruct = config.toStruct();
save('config/cartpole_config.mat', '-struct', 'configStruct');
fprintf('倒立摆环境的默认配置已保存到: config/cartpole_config.mat\n');
