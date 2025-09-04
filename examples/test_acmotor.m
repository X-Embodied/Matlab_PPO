% test_acmotor.m
% 测试已训练的交流感应电机FOC控制系统PPO模型

% 添加路径
addpath('../');
addpath('../core');
addpath('../environments');
addpath('../config');
addpath('../utils');

% 模型路径
modelPath = '../logs/acmotor/model_final.mat';
if ~exist(modelPath, 'file')
    % 如果找不到最终模型，尝试找到目录中的任意模型
    logDir = '../logs/acmotor';
    files = dir(fullfile(logDir, 'model_*.mat'));
    if ~isempty(files)
        [~, idx] = max([files.datenum]);
        modelPath = fullfile(logDir, files(idx).name);
        fprintf('未找到最终模型，将使用最新模型: %s\n', modelPath);
    else
        error('找不到任何训练好的模型，请先运行train_acmotor.m训练模型');
    end
end

% 加载配置
config = PPOConfig();
config.envName = 'ACMotorEnv';
config.actorLayerSizes = [256, 256, 128, 64];
config.criticLayerSizes = [256, 256, 128, 64];
config.useGPU = true; % 根据需要设置是否使用GPU

% 创建PPO代理
agent = PPOAgent(config);

% 加载模型
fprintf('加载模型: %s\n', modelPath);
agent.loadModel(modelPath);

% 测试参数
numEpisodes = 3;              % 测试回合数
renderTest = true;            % 是否可视化测试过程
testDuration = 5000;          % 测试步数（由于交流电机采样率高）
saveResults = true;           % 是否保存测试结果
testScenarios = {
    '正常负载运行',           % 标准工作负载
    '突加负载响应',           % 负载突增
    '速度阶跃响应'            % 速度突变
};

% 创建结果目录
resultsDir = '../results/acmotor';
if saveResults && ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

% 测试已训练的模型
fprintf('开始测试，将执行%d个测试场景...\n', length(testScenarios));

% 创建结果结构
testResults = struct();

for scenarioIdx = 1:length(testScenarios)
    scenarioName = testScenarios{scenarioIdx};
    fprintf('\n开始场景 %d: %s\n', scenarioIdx, scenarioName);
    
    % 创建环境
    env = ACMotorEnv();
    
    % 为不同场景设置不同的参数
    switch scenarioIdx
        case 1 % 正常负载运行
            % 使用默认参数
            env.loadProfile = [
                0.3 * env.nominalTorque;  % 30%额定负载
                0.3 * env.nominalTorque;
                0.3 * env.nominalTorque;
                0.3 * env.nominalTorque;
                0.3 * env.nominalTorque;
            ];
            targetSpeed = 0.8 * env.nominalSpeed; % 80%额定速度
            env.targetSpeed = targetSpeed;
            
        case 2 % 突加负载响应
            % 设置突加负载测试
            env.loadProfile = [
                0.2 * env.nominalTorque;  % 初始轻载
                0.8 * env.nominalTorque;  % 突加到80%负载
                0.8 * env.nominalTorque;
                0.8 * env.nominalTorque;
                0.2 * env.nominalTorque;  % 恢复到轻载
            ];
            env.loadChangeTime = [1000, 4000]; % 在这些步骤改变负载
            targetSpeed = 0.7 * env.nominalSpeed; % 70%额定速度
            env.targetSpeed = targetSpeed;
            
        case 3 % 速度阶跃响应
            % 设置速度阶跃测试
            env.loadProfile = [
                0.4 * env.nominalTorque;  % 40%额定负载
                0.4 * env.nominalTorque;
                0.4 * env.nominalTorque;
                0.4 * env.nominalTorque;
                0.4 * env.nominalTorque;
            ];
            % 初始速度设置
            initialSpeed = 0.4 * env.nominalSpeed;
            targetSpeed = 0.9 * env.nominalSpeed; % 将在测试中改变目标
            env.targetSpeed = initialSpeed;
            % 我们将在第1000步改变速度设定值
            speedChangeStep = 1000;
    end
    
    % 重置环境
    obs = env.reset();
    
    % 初始化性能指标
    totalReward = 0;
    speedErrors = [];
    
    % 创建记录数据的结构
    recordData = struct();
    recordData.time = [];
    recordData.speed = [];
    recordData.targetSpeed = [];
    recordData.id = [];
    recordData.iq = [];
    recordData.Te = [];
    recordData.Tl = [];
    recordData.Vd = [];
    recordData.Vq = [];
    recordData.reward = [];
    
    % 开始测试循环
    for step = 1:testDuration
        % 如果是速度阶跃测试，在指定步骤改变目标速度
        if scenarioIdx == 3 && step == speedChangeStep
            env.targetSpeed = targetSpeed;
            fprintf('在步骤 %d 将目标速度从 %.1f rad/s 改变到 %.1f rad/s\n', 
                    step, initialSpeed, targetSpeed);
        end
        
        % 转换为dlarray并根据需要迁移到GPU
        if agent.useGPU
            dlObs = dlarray(single(obs), 'CB');
            dlObs = gpuArray(dlObs);
        else
            dlObs = dlarray(single(obs), 'CB');
        end
        
        % 使用确定性策略（使用均值）
        action = agent.actorNet.getMeanAction(dlObs);
        
        % 转换为CPU并提取数值
        if agent.useGPU
            action = gather(extractdata(action));
        else
            action = extractdata(action);
        end
        
        % 执行动作
        [obs, reward, done, info] = env.step(action);
        totalReward = totalReward + reward;
        
        % 记录性能指标
        speedError = abs(info.targetSpeed - info.speed) / env.maxSpeed;
        speedErrors(end+1) = speedError;
        
        % 记录数据用于分析
        recordData.time(end+1) = step * env.dt;
        recordData.speed(end+1) = info.speed;
        recordData.targetSpeed(end+1) = info.targetSpeed;
        recordData.id(end+1) = info.id;
        recordData.iq(end+1) = info.iq;
        recordData.Te(end+1) = info.Te;
        recordData.Tl(end+1) = info.Tl;
        recordData.Vd(end+1) = action(1) * env.maxVoltage;
        recordData.Vq(end+1) = action(2) * env.maxVoltage;
        recordData.reward(end+1) = reward;
        
        % 渲染环境（如需要）
        if renderTest && mod(step, 10) == 0
            env.render();
            pause(0.001); % 降低渲染频率以避免过多图形更新
        end
    end
    
    % 计算性能指标
    avgSpeedError = mean(speedErrors);
    maxSpeedError = max(speedErrors);
    steadyStateError = mean(speedErrors(end-500:end)); % 最后500点的稳态误差
    
    % 计算上升时间和调节时间（仅对速度阶跃响应场景）
    if scenarioIdx == 3
        % 找到速度阶跃后的数据
        stepIndex = find(recordData.time >= speedChangeStep * env.dt, 1);
        postStepSpeed = recordData.speed(stepIndex:end);
        postStepTime = recordData.time(stepIndex:end) - recordData.time(stepIndex);
        
        % 计算上升时间（从10%到90%的响应时间）
        speedChange = targetSpeed - initialSpeed;
        tenPercent = initialSpeed + 0.1 * speedChange;
        ninetyPercent = initialSpeed + 0.9 * speedChange;
        
        tenPercentIndex = find(postStepSpeed >= tenPercent, 1);
        ninetyPercentIndex = find(postStepSpeed >= ninetyPercent, 1);
        
        if ~isempty(tenPercentIndex) && ~isempty(ninetyPercentIndex)
            riseTime = postStepTime(ninetyPercentIndex) - postStepTime(tenPercentIndex);
        else
            riseTime = NaN;
        end
        
        % 计算调节时间（达到并维持在最终值±5%之内）
        fivePercent = 0.05 * speedChange;
        steadyBand = [targetSpeed - fivePercent, targetSpeed + fivePercent];
        
        for i = 1:length(postStepSpeed)
            if postStepSpeed(i) >= steadyBand(1) && postStepSpeed(i) <= steadyBand(2)
                % 检查是否之后的所有点都在稳态带内
                if all(postStepSpeed(i:end) >= steadyBand(1) & postStepSpeed(i:end) <= steadyBand(2))
                    settlingTime = postStepTime(i);
                    break;
                end
            end
        end
        
        if ~exist('settlingTime', 'var')
            settlingTime = NaN;
        end
    else
        riseTime = NaN;
        settlingTime = NaN;
    end
    
    % 打印性能指标
    fprintf('场景 %d: %s 完成\n', scenarioIdx, scenarioName);
    fprintf('  总奖励: %.2f\n', totalReward);
    fprintf('  平均速度误差: %.4f\n', avgSpeedError);
    fprintf('  最大速度误差: %.4f\n', maxSpeedError);
    fprintf('  稳态误差: %.4f\n', steadyStateError);
    
    if scenarioIdx == 3
        fprintf('  上升时间 (10%%-90%%): %.3f s\n', riseTime);
        fprintf('  调节时间 (±5%%): %.3f s\n', settlingTime);
    end
    
    % 保存性能指标
    testResults.(sprintf('scenario%d', scenarioIdx)).name = scenarioName;
    testResults.(sprintf('scenario%d', scenarioIdx)).totalReward = totalReward;
    testResults.(sprintf('scenario%d', scenarioIdx)).avgSpeedError = avgSpeedError;
    testResults.(sprintf('scenario%d', scenarioIdx)).maxSpeedError = maxSpeedError;
    testResults.(sprintf('scenario%d', scenarioIdx)).steadyStateError = steadyStateError;
    testResults.(sprintf('scenario%d', scenarioIdx)).riseTime = riseTime;
    testResults.(sprintf('scenario%d', scenarioIdx)).settlingTime = settlingTime;
    testResults.(sprintf('scenario%d', scenarioIdx)).data = recordData;
    
    % 绘制并保存测试结果图
    if saveResults
        % 创建图形
        fig = figure('Name', ['交流电机控制 - ', scenarioName], 'Position', [100, 100, 1200, 900]);
        
        % 1. 速度响应
        subplot(3, 2, 1);
        plot(recordData.time, recordData.speed, 'b-', 'LineWidth', 1.5);
        hold on;
        plot(recordData.time, recordData.targetSpeed, 'r--', 'LineWidth', 1.5);
        title('速度响应');
        xlabel('时间 (s)');
        ylabel('速度 (rad/s)');
        legend('实际速度', '目标速度', 'Location', 'best');
        grid on;
        
        % 2. 速度误差
        subplot(3, 2, 2);
        speedErr = abs(recordData.speed - recordData.targetSpeed);
        plot(recordData.time, speedErr, 'b-', 'LineWidth', 1.5);
        title('速度误差');
        xlabel('时间 (s)');
        ylabel('误差 (rad/s)');
        grid on;
        
        % 3. d-q轴电流
        subplot(3, 2, 3);
        plot(recordData.time, recordData.id, 'b-', 'LineWidth', 1.5);
        hold on;
        plot(recordData.time, recordData.iq, 'r-', 'LineWidth', 1.5);
        title('d-q轴电流');
        xlabel('时间 (s)');
        ylabel('电流 (A)');
        legend('id', 'iq', 'Location', 'best');
        grid on;
        
        % 4. d-q轴电压（控制信号）
        subplot(3, 2, 4);
        plot(recordData.time, recordData.Vd, 'b-', 'LineWidth', 1.5);
        hold on;
        plot(recordData.time, recordData.Vq, 'r-', 'LineWidth', 1.5);
        title('d-q轴电压（控制信号）');
        xlabel('时间 (s)');
        ylabel('电压 (V)');
        legend('Vd', 'Vq', 'Location', 'best');
        grid on;
        
        % 5. 转矩
        subplot(3, 2, 5);
        plot(recordData.time, recordData.Te, 'b-', 'LineWidth', 1.5);
        hold on;
        plot(recordData.time, recordData.Tl, 'r-', 'LineWidth', 1.5);
        title('转矩');
        xlabel('时间 (s)');
        ylabel('转矩 (N·m)');
        legend('电磁转矩', '负载转矩', 'Location', 'best');
        grid on;
        
        % 6. 瞬时奖励
        subplot(3, 2, 6);
        plot(recordData.time, recordData.reward, 'k-', 'LineWidth', 1.5);
        title('瞬时奖励');
        xlabel('时间 (s)');
        ylabel('奖励');
        grid on;
        
        % 保存图形
        if saveResults
            saveName = fullfile(resultsDir, sprintf('scenario%d_%s.fig', scenarioIdx, strrep(scenarioName, ' ', '_')));
            saveas(fig, saveName);
            saveName = fullfile(resultsDir, sprintf('scenario%d_%s.png', scenarioIdx, strrep(scenarioName, ' ', '_')));
            saveas(fig, saveName);
            fprintf('  已保存测试结果图到: %s\n', saveName);
        end
    end
end

% 汇总比较所有场景
if saveResults && length(testScenarios) > 1
    % 创建表格比较所有场景
    scenarioNames = {};
    avgErrors = [];
    maxErrors = [];
    steadyErrors = [];
    riseTimes = [];
    settlingTimes = [];
    totalRewards = [];
    
    for i = 1:length(testScenarios)
        scenarioNames{i} = testResults.(sprintf('scenario%d', i)).name;
        avgErrors(i) = testResults.(sprintf('scenario%d', i)).avgSpeedError;
        maxErrors(i) = testResults.(sprintf('scenario%d', i)).maxSpeedError;
        steadyErrors(i) = testResults.(sprintf('scenario%d', i)).steadyStateError;
        riseTimes(i) = testResults.(sprintf('scenario%d', i)).riseTime;
        settlingTimes(i) = testResults.(sprintf('scenario%d', i)).settlingTime;
        totalRewards(i) = testResults.(sprintf('scenario%d', i)).totalReward;
    end
    
    % 创建比较表格
    comparisonTable = table(scenarioNames', avgErrors', maxErrors', steadyErrors', ...
                           riseTimes', settlingTimes', totalRewards', ...
                           'VariableNames', {'场景', '平均速度误差', '最大速度误差', '稳态误差', ...
                                            '上升时间', '调节时间', '总奖励'});
    
    % 显示表格
    disp('所有场景性能对比:');
    disp(comparisonTable);
    
    % 保存性能比较结果
    save(fullfile(resultsDir, 'test_results.mat'), 'testResults', 'comparisonTable');
    fprintf('测试结果已保存到: %s\n', fullfile(resultsDir, 'test_results.mat'));
    
    % 绘制所有场景的速度响应比较图
    figure('Name', '所有场景速度响应比较', 'Position', [100, 100, 1200, 600]);
    
    % 速度响应
    subplot(1, 2, 1);
    hold on;
    colors = {'b', 'r', 'g', 'm', 'c'};
    for i = 1:length(testScenarios)
        plot(testResults.(sprintf('scenario%d', i)).data.time, ...
             testResults.(sprintf('scenario%d', i)).data.speed, ...
             [colors{mod(i-1, length(colors))+1}, '-'], 'LineWidth', 1.5);
    end
    title('所有场景速度响应');
    xlabel('时间 (s)');
    ylabel('速度 (rad/s)');
    legend(scenarioNames, 'Location', 'best');
    grid on;
    
    % 速度误差
    subplot(1, 2, 2);
    hold on;
    for i = 1:length(testScenarios)
        speedErr = abs(testResults.(sprintf('scenario%d', i)).data.speed - ...
                      testResults.(sprintf('scenario%d', i)).data.targetSpeed);
        plot(testResults.(sprintf('scenario%d', i)).data.time, ...
             speedErr, ...
             [colors{mod(i-1, length(colors))+1}, '-'], 'LineWidth', 1.5);
    end
    title('所有场景速度误差');
    xlabel('时间 (s)');
    ylabel('误差 (rad/s)');
    legend(scenarioNames, 'Location', 'best');
    grid on;
    
    % 保存比较图
    saveName = fullfile(resultsDir, 'scenarios_comparison.png');
    saveas(gcf, saveName);
    fprintf('场景比较图已保存到: %s\n', saveName);
end

fprintf('\n测试完成！\n');
