function config = default_doublependulum_config()
    % default_doublependulum_config 双倒立摆环境的默认MAPPO配置
    
    % 创建MAPPO配置对象
    config = MAPPOConfig();
    
    % 设置环境
    config.envName = 'DoublePendulumEnv';
    config.numAgents = 2;  % 两个智能体，各控制一个摆杆
    
    % 网络配置 - 使用较大的网络以处理复杂的动力学
    config.actorLayerSizes = [128, 64];
    config.criticLayerSizes = [256, 128];
    
    % 算法超参数
    config.gamma = 0.99;      % 折扣因子
    config.lambda = 0.95;     % GAE参数
    config.epsilon = 0.2;     % PPO裁剪参数
    config.entropyCoef = 0.01; % 熵正则化系数，鼓励探索
    config.vfCoef = 0.5;      % 价值函数系数
    config.maxGradNorm = 0.5; % 梯度裁剪阈值
    
    % 优化器配置
    config.actorLearningRate = 3e-4;
    config.criticLearningRate = 1e-3;
    config.momentum = 0.9;
    
    % 训练配置
    config.numIterations = 200;     % 训练迭代次数
    config.batchSize = 64;          % 批次大小
    config.epochsPerIter = 4;       % 每次迭代的训练轮数
    config.trajectoryLen = 200;     % 轨迹长度
    config.numTrajectories = 20;    % 每次迭代收集的轨迹数量
    
    % 硬件配置
    config.useGPU = true;
    
    % 日志配置
    config.logDir = 'logs/doublependulum';
    config.evalFreq = 10;            % 每10次迭代评估一次
    config.numEvalEpisodes = 5;      % 每次评估5个回合
    config.saveModelFreq = 20;       % 每20次迭代保存一次模型
end
