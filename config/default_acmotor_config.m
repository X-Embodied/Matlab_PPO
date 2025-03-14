% default_acmotor_config.m
% 交流感应电机环境的默认PPO配置文件

% 创建配置对象
config = PPOConfig();

% 环境配置
config.envName = 'ACMotorEnv';

% 网络配置 - 对于复杂的交流电机控制，使用更深更宽的网络
config.actorLayerSizes = [256, 256, 128, 64];    % Actor网络隐藏层大小
config.criticLayerSizes = [256, 256, 128, 64];   % Critic网络隐藏层大小

% 算法超参数
config.gamma = 0.99;                    % 折扣因子
config.lambda = 0.95;                   % GAE参数
config.epsilon = 0.1;                   % PPO裁剪参数（减小以更加保守地学习）
config.entropyCoef = 0.001;             % 熵正则化系数（交流电机控制更精确，更低的熵系数）
config.vfCoef = 0.5;                    % 价值函数系数
config.maxGradNorm = 0.5;               % 梯度裁剪

% 优化器配置
config.actorLearningRate = 5e-5;        % Actor学习率（交流电机更复杂，使用更小的学习率）
config.criticLearningRate = 5e-5;       % Critic学习率
config.momentum = 0.9;                  % 动量

% 训练配置
config.numIterations = 400;             % 训练迭代次数（复杂系统需要更多迭代）
config.batchSize = 256;                 % 批次大小
config.epochsPerIter = 15;              % 每次迭代的训练轮数（增加以提高样本利用效率）
config.trajectoryLen = 500;             % 轨迹长度（交流电机采样率高，需要更长轨迹）
config.numTrajectories = 32;            % 每次迭代收集的轨迹数量（增加以提高稳定性）

% 硬件配置
config.useGPU = true;                   % 是否使用GPU

% 日志配置
config.logDir = 'logs/acmotor';         % 日志保存目录
config.evalFreq = 10;                   % 评估频率 (迭代次数)
config.numEvalEpisodes = 5;             % 评估时的回合数
config.saveModelFreq = 40;              % 保存模型频率 (迭代次数)

% 保存配置
saveDir = 'config';
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

save('config/acmotor_config.mat', '-struct', 'config.toStruct()');
fprintf('交流感应电机环境的默认配置已保存到: config/acmotor_config.mat\n');
