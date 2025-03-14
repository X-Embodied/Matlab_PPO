% train_cartpole.m
% 训练倒立摆环境示例脚本

% 添加路径
addpath('../');
addpath('../core');
addpath('../environments');
addpath('../config');
addpath('../utils');

% 创建日志目录
logDir = '../logs/cartpole';
if ~exist(logDir, 'dir')
    mkdir(logDir);
end

% 加载配置
config = PPOConfig();

% 环境配置
config.envName = 'CartPoleEnv';

% 网络配置
config.actorLayerSizes = [64, 64];
config.criticLayerSizes = [64, 64];

% 算法超参数
config.gamma = 0.99;
config.lambda = 0.95;
config.epsilon = 0.2;
config.entropyCoef = 0.01;
config.vfCoef = 0.5;
config.maxGradNorm = 0.5;

% 优化器配置
config.actorLearningRate = 3e-4;
config.criticLearningRate = 3e-4;
config.momentum = 0.9;

% 训练配置
config.numIterations = 100;
config.batchSize = 64;
config.epochsPerIter = 4;
config.trajectoryLen = 200;

% 硬件配置
config.useGPU = true;

% 日志配置
config.logDir = logDir;
config.evalFreq = 5;
config.numEvalEpisodes = 10;
config.saveModelFreq = 10;

% 创建PPO代理
agent = PPOAgent(config);

% 训练代理
fprintf('开始训练倒立摆环境...\n');
agent.train(config.numIterations);

% 训练完成后评估
fprintf('训练完成，开始评估...\n');
evalResult = agent.evaluate(20);

% 显示评估结果
fprintf('评估结果:\n');
fprintf('  平均回报: %.2f ± %.2f\n', evalResult.meanReturn, evalResult.stdReturn);
fprintf('  最小回报: %.2f\n', evalResult.minReturn);
fprintf('  最大回报: %.2f\n', evalResult.maxReturn);
fprintf('  平均回合长度: %.2f\n', evalResult.meanLength);

% 可视化一个回合
fprintf('可视化一个回合...\n');
env = feval(config.envName);
obs = env.reset();
done = false;

while ~done
    % 转换为dlarray并根据需要迁移到GPU
    if agent.useGPU
        dlObs = dlarray(single(obs), 'CB');
        dlObs = gpuArray(dlObs);
    else
        dlObs = dlarray(single(obs), 'CB');
    end
    
    % 采样动作
    [action, ~] = agent.actorNet.sampleAction(dlObs);
    
    % 转换为CPU并提取数值
    if agent.useGPU
        action = gather(extractdata(action));
    else
        action = extractdata(action);
    end
    
    % 执行动作
    [obs, reward, done, ~] = env.step(action);
    
    % 渲染环境
    env.render();
    pause(0.01);
end

fprintf('演示完成\n');
