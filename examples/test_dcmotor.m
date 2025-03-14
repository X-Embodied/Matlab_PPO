% test_dcmotor.m
% 测试已训练的直流电机控制系统PPO模型

% 添加路径
addpath('../');
addpath('../core');
addpath('../environments');
addpath('../config');
addpath('../utils');

% 模型路径
modelPath = '../logs/dcmotor/model_iter_200.mat';
if ~exist(modelPath, 'file')
    % 如果找不到指定迭代次数的模型，尝试找到目录中的任意模型
    logDir = '../logs/dcmotor';
    files = dir(fullfile(logDir, 'model_iter_*.mat'));
    if ~isempty(files)
        [~, idx] = max([files.datenum]);
        modelPath = fullfile(logDir, files(idx).name);
        fprintf('未找到指定模型，将使用最新模型: %s\n', modelPath);
    else
        error('找不到任何训练好的模型');
    end
end

% 加载配置
config = PPOConfig();
config.envName = 'DCMotorEnv';
config.actorLayerSizes = [128, 128, 64];
config.criticLayerSizes = [128, 128, 64];
config.useGPU = true; % 根据需要设置是否使用GPU

% 创建PPO代理
agent = PPOAgent(config);

% 加载模型
fprintf('加载模型: %s\n', modelPath);
agent.loadModel(modelPath);

% 测试参数
numEpisodes = 5;      % 测试回合数
renderTest = true;    % 是否可视化测试过程
changingTarget = true; % 是否在回合中改变目标角度

% 测试已训练的模型
fprintf('开始测试，共%d回合...\n', numEpisodes);
totalReward = 0;
totalSteps = 0;
successCount = 0;

% 创建环境
env = feval(config.envName);

% 创建记录数据的结构
recordData = struct();
recordData.episodes = cell(numEpisodes, 1);

for episode = 1:numEpisodes
    % 重置环境
    obs = env.reset();
    episodeReward = 0;
    steps = 0;
    done = false;
    
    % 记录当前回合数据
    episodeData = struct();
    episodeData.time = [];
    episodeData.angles = [];
    episodeData.targets = [];
    episodeData.angleDiffs = [];
    episodeData.speeds = [];
    episodeData.currents = [];
    episodeData.actions = [];
    episodeData.rewards = [];
    
    % 设定固定目标改变时间点（如果changingTarget为true）
    targetChangePoints = [100, 200, 300];
    nextTargetChange = 1;
    
    while ~done
        % 转换为dlarray并根据需要迁移到GPU
        if agent.useGPU
            dlObs = dlarray(single(obs), 'CB');
            dlObs = gpuArray(dlObs);
        else
            dlObs = dlarray(single(obs), 'CB');
        end
        
        % 使用确定性策略（使用均值，无探索）
        action = agent.actorNet.getMeanAction(dlObs);
        
        % 转换为CPU并提取数值
        if agent.useGPU
            action = gather(extractdata(action));
        else
            action = extractdata(action);
        end
        
        % 执行动作
        [obs, reward, done, info] = env.step(action);
        
        % 更新统计
        episodeReward = episodeReward + reward;
        steps = steps + 1;
        
        % 记录数据
        episodeData.time(end+1) = steps;
        episodeData.angles(end+1) = env.state(1);
        episodeData.targets(end+1) = env.targetAngle;
        episodeData.angleDiffs(end+1) = obs(1); % 观察中的角度差
        episodeData.speeds(end+1) = env.state(2);
        episodeData.currents(end+1) = env.state(3);
        episodeData.actions(end+1) = action;
        episodeData.rewards(end+1) = reward;
        
        % 如果需要随机改变目标角度
        if changingTarget && nextTargetChange <= length(targetChangePoints) && steps == targetChangePoints(nextTargetChange)
            env.resetTarget();
            fprintf('回合 %d: 在步骤 %d 改变目标角度为 %.2f rad\n', episode, steps, env.targetAngle);
            nextTargetChange = nextTargetChange + 1;
        end
        
        % 如果需要渲染
        if renderTest
            env.render();
            pause(0.01); % 控制渲染速度
        end
    end
    
    % 保存回合数据
    recordData.episodes{episode} = episodeData;
    
    % 判断是否成功完成任务
    finalDistance = info.distance;
    if finalDistance < 0.1 % 使用较小的阈值判断是否达到目标
        successCount = successCount + 1;
    end
    
    % 更新总统计
    totalReward = totalReward + episodeReward;
    totalSteps = totalSteps + steps;
    
    fprintf('回合 %d: 奖励 = %.2f, 步数 = %d, 最终角度误差 = %.4f rad\n', ...
        episode, episodeReward, steps, finalDistance);
end

% 打印测试结果摘要
fprintf('\n测试结果摘要:\n');
fprintf('平均奖励: %.2f\n', totalReward / numEpisodes);
fprintf('平均步数: %.2f\n', totalSteps / numEpisodes);
fprintf('成功率: %.1f%%\n', successCount / numEpisodes * 100);

% 绘制测试结果汇总图表
figure('Name', '直流电机控制测试结果', 'Position', [100, 100, 1000, 800]);

% 1. 角度跟踪性能
subplot(2, 2, 1);
hold on;
for i = 1:numEpisodes
    plot(recordData.episodes{i}.time, recordData.episodes{i}.angleDiffs, 'LineWidth', 1.5);
end
title('角度误差');
xlabel('时间步');
ylabel('角度误差 (rad)');
grid on;
legend(arrayfun(@(x) sprintf('回合 %d', x), 1:numEpisodes, 'UniformOutput', false));

% 2. 控制信号
subplot(2, 2, 2);
hold on;
for i = 1:numEpisodes
    plot(recordData.episodes{i}.time, recordData.episodes{i}.actions, 'LineWidth', 1.5);
end
title('控制信号 (归一化电压)');
xlabel('时间步');
ylabel('控制信号 [-1,1]');
grid on;

% 3. 角速度
subplot(2, 2, 3);
hold on;
for i = 1:numEpisodes
    plot(recordData.episodes{i}.time, recordData.episodes{i}.speeds, 'LineWidth', 1.5);
end
title('角速度');
xlabel('时间步');
ylabel('角速度 (rad/s)');
grid on;

% 4. 电流
subplot(2, 2, 4);
hold on;
for i = 1:numEpisodes
    plot(recordData.episodes{i}.time, recordData.episodes{i}.currents, 'LineWidth', 1.5);
end
title('电机电流');
xlabel('时间步');
ylabel('电流 (A)');
grid on;

% 如果想要显示更详细的单回合性能，可以单独绘制
if numEpisodes > 0
    bestEpisode = 1; % 可以根据奖励选择最佳回合
    
    figure('Name', sprintf('直流电机控制 - 回合 %d 详细分析', bestEpisode), 'Position', [100, 100, 1000, 600]);
    
    % 1. 角度跟踪
    subplot(3, 1, 1);
    plot(recordData.episodes{bestEpisode}.time, recordData.episodes{bestEpisode}.angles, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(recordData.episodes{bestEpisode}.time, recordData.episodes{bestEpisode}.targets, 'r--', 'LineWidth', 1.5);
    title('角度跟踪');
    xlabel('时间步');
    ylabel('角度 (rad)');
    legend('实际角度', '目标角度');
    grid on;
    
    % 2. 角速度
    subplot(3, 1, 2);
    plot(recordData.episodes{bestEpisode}.time, recordData.episodes{bestEpisode}.speeds, 'g-', 'LineWidth', 1.5);
    title('角速度');
    xlabel('时间步');
    ylabel('角速度 (rad/s)');
    grid on;
    
    % 3. 控制信号与奖励
    subplot(3, 1, 3);
    yyaxis left;
    plot(recordData.episodes{bestEpisode}.time, recordData.episodes{bestEpisode}.actions, 'm-', 'LineWidth', 1.5);
    ylabel('控制信号 [-1,1]');
    
    yyaxis right;
    plot(recordData.episodes{bestEpisode}.time, recordData.episodes{bestEpisode}.rewards, 'k-', 'LineWidth', 1);
    ylabel('奖励');
    
    title('控制信号与奖励');
    xlabel('时间步');
    grid on;
    legend('控制信号', '奖励');
end

fprintf('测试完成，分析图表已生成\n');
