% train_dcmotor.m
% 训练直流电机控制系统的PPO代理

% 添加路径
addpath('../');
addpath('../core');
addpath('../environments');
addpath('../config');
addpath('../utils');

% 创建日志目录
logDir = '../logs/dcmotor';
if ~exist(logDir, 'dir')
    mkdir(logDir);
end

% 加载配置
config = PPOConfig();

% 环境配置
config.envName = 'DCMotorEnv';

% 网络配置 - 连续控制问题推荐使用更深的网络
config.actorLayerSizes = [128, 128, 64];
config.criticLayerSizes = [128, 128, 64];

% 算法超参数
config.gamma = 0.99;
config.lambda = 0.95;
config.epsilon = 0.2;
config.entropyCoef = 0.005;  % 连续控制问题通常使用较小的熵系数
config.vfCoef = 0.5;
config.maxGradNorm = 0.5;

% 优化器配置
config.actorLearningRate = 1e-4;  % 连续控制问题通常使用较小的学习率
config.criticLearningRate = 1e-4;
config.momentum = 0.9;

% 训练配置
config.numIterations = 200;
config.batchSize = 128;
config.epochsPerIter = 10;
config.trajectoryLen = 250;

% 硬件配置
config.useGPU = true;

% 日志配置
config.logDir = logDir;
config.evalFreq = 5;
config.numEvalEpisodes = 10;
config.saveModelFreq = 20;

% 创建PPO代理
fprintf('正在初始化PPO代理...\n');
agent = PPOAgent(config);

% 训练代理
fprintf('开始训练直流电机控制系统...\n');
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
totalReward = 0;

figure('Name', '直流电机控制测试', 'Position', [100, 100, 800, 600]);
subplot(3, 1, 1);
anglePlot = plot(0, 0, 'b-', 'LineWidth', 1.5);
hold on;
targetPlot = plot(0, 0, 'r--', 'LineWidth', 1.5);
title('角度跟踪');
xlabel('时间步');
ylabel('角度 (rad)');
legend('实际角度', '目标角度');
grid on;

subplot(3, 1, 2);
speedPlot = plot(0, 0, 'g-', 'LineWidth', 1.5);
title('角速度');
xlabel('时间步');
ylabel('角速度 (rad/s)');
grid on;

subplot(3, 1, 3);
actionPlot = plot(0, 0, 'm-', 'LineWidth', 1.5);
title('控制信号');
xlabel('时间步');
ylabel('电压比例 [-1,1]');
grid on;

% 记录数据
timeSteps = [];
angles = [];
targets = [];
speeds = [];
actions = [];

step = 0;
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
    [obs, reward, done, info] = env.step(action);
    totalReward = totalReward + reward;
    
    % 记录数据
    step = step + 1;
    timeSteps(end+1) = step;
    angles(end+1) = env.state(1);
    targets(end+1) = info.targetAngle;
    speeds(end+1) = env.state(2);
    actions(end+1) = action;
    
    % 更新图形
    anglePlot.XData = timeSteps;
    anglePlot.YData = angles;
    targetPlot.XData = timeSteps;
    targetPlot.YData = targets;
    speedPlot.XData = timeSteps;
    speedPlot.YData = speeds;
    actionPlot.XData = timeSteps;
    actionPlot.YData = actions;
    
    % 调整X轴范围
    for i = 1:3
        subplot(3, 1, i);
        xlim([1, max(1, step)]);
    end
    subplot(3, 1, 1);
    ylim([0, 2*pi]);
    
    % 渲染环境
    env.render();
    
    pause(0.01);
end

fprintf('演示完成，总奖励: %.2f\n', totalReward);
