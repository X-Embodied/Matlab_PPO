% default_dcmotor_config.m
% 直流电机环境的默认PPO配置文件

% 创建配置对象
config = PPOConfig();

% 环境配置
config.envName = 'DCMotorEnv';

% 网络配置 - 使用更深的网络处理连续控制问题
config.actorLayerSizes = [128, 128, 64];    % Actor网络隐藏层大小
config.criticLayerSizes = [128, 128, 64];   % Critic网络隐藏层大小

% 算法超参数
config.gamma = 0.99;                    % 折扣因子
config.lambda = 0.95;                   % GAE参数
config.epsilon = 0.2;                   % PPO裁剪参数
config.entropyCoef = 0.005;             % 熵正则化系数（由于连续控制需要更精确，降低熵系数）
config.vfCoef = 0.5;                    % 价值函数系数
config.maxGradNorm = 0.5;               % 梯度裁剪

% 优化器配置
config.actorLearningRate = 1e-4;        % Actor学习率（对连续控制使用较小的学习率）
config.criticLearningRate = 1e-4;       % Critic学习率
config.momentum = 0.9;                  % 动量

% 训练配置
config.numIterations = 200;             % 训练迭代次数
config.batchSize = 128;                 % 批次大小
config.epochsPerIter = 10;              % 每次迭代的训练轮数（增加以提高样本利用效率）
config.trajectoryLen = 250;             % 轨迹长度
config.numTrajectories = 16;            % 每次迭代收集的轨迹数量

% 硬件配置
config.useGPU = true;                   % 是否使用GPU

% 日志配置
config.logDir = 'logs/dcmotor';         % 日志保存目录
config.evalFreq = 5;                    % 评估频率 (迭代次数)
config.numEvalEpisodes = 10;            % 评估时的回合数
config.saveModelFreq = 20;              % 保存模型频率 (迭代次数)

% 保存配置
saveDir = 'config';
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

save('config/dcmotor_config.mat', '-struct', 'config.toStruct()');
fprintf('直流电机环境的默认配置已保存到: config/dcmotor_config.mat\n');
