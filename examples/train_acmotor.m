% train_acmotor.m
% 训练交流感应电机控制系统的PPO代理

% 添加路径
addpath('../');
addpath('../core');
addpath('../environments');
addpath('../config');
addpath('../utils');

% 创建日志目录
logDir = '../logs/acmotor';
if ~exist(logDir, 'dir')
    mkdir(logDir);
end

% 加载配置
config = PPOConfig();

% 环境配置
config.envName = 'ACMotorEnv';

% 网络配置 - 复杂的交流电机系统需要更深更宽的网络
config.actorLayerSizes = [256, 256, 128, 64];
config.criticLayerSizes = [256, 256, 128, 64];

% 算法超参数
config.gamma = 0.99;
config.lambda = 0.95;
config.epsilon = 0.1;
config.entropyCoef = 0.001;
config.vfCoef = 0.5;
config.maxGradNorm = 0.5;

% 优化器配置
config.actorLearningRate = 5e-5;
config.criticLearningRate = 5e-5;
config.momentum = 0.9;

% 训练配置
config.numIterations = 400;
config.batchSize = 256;
config.epochsPerIter = 15;
config.trajectoryLen = 500;
config.numTrajectories = 32;

% 硬件配置
config.useGPU = true;

% 日志配置
config.logDir = logDir;
config.evalFreq = 10;
config.numEvalEpisodes = 5;
config.saveModelFreq = 40;

% 创建PPO代理
fprintf('正在初始化PPO代理...\n');
agent = PPOAgent(config);

% 创建Logger实例
logger = Logger(config.logDir);

% 训练代理
fprintf('开始训练交流感应电机控制系统...\n');
fprintf('训练过程中会模拟工业环境下的负载突变场景\n');

% 训练循环
for iter = 1:config.numIterations
    fprintf('迭代 %d/%d\n', iter, config.numIterations);
    
    % 收集轨迹
    fprintf('  收集轨迹数据...\n');
    trajectories = agent.collectTrajectories(config.trajectoryLen, config.numTrajectories);
    
    % 计算优势和回报
    fprintf('  计算优势估计...\n');
    trajectories = agent.computeAdvantagesAndReturns(trajectories);
    
    % 更新策略
    fprintf('  更新策略...\n');
    [actorLoss, criticLoss, entropy] = agent.updatePolicy(trajectories, config.epochsPerIter, config.batchSize);
    
    % 记录训练信息
    meanReturn = mean([trajectories.return]);
    meanLength = mean([trajectories.length]);
    
    % 打印当前迭代的指标
    fprintf('  平均回报: %.2f\n', meanReturn);
    fprintf('  平均长度: %.2f\n', meanLength);
    fprintf('  Actor损失: %.4f\n', actorLoss);
    fprintf('  Critic损失: %.4f\n', criticLoss);
    fprintf('  策略熵: %.4f\n', entropy);
    
    % 将训练指标记录到logger
    logger.logTrainingMetrics(iter, meanReturn, meanLength, actorLoss, criticLoss, entropy);
    
    % 评估当前策略
    if mod(iter, config.evalFreq) == 0
        fprintf('  评估当前策略...\n');
        evalResult = agent.evaluate(config.numEvalEpisodes);
        fprintf('  评估平均回报: %.2f ± %.2f\n', evalResult.meanReturn, evalResult.stdReturn);
        
        % 记录评估指标
        logger.logEvaluationMetrics(iter, evalResult.meanReturn, evalResult.stdReturn, evalResult.minReturn, evalResult.maxReturn);
        
        % 可视化速度跟踪性能
        visualizePerformance(agent, config.envName);
    end
    
    % 保存模型
    if mod(iter, config.saveModelFreq) == 0
        modelPath = fullfile(config.logDir, sprintf('model_iter_%d.mat', iter));
        fprintf('  保存模型到: %s\n', modelPath);
        agent.saveModel(modelPath);
    end
end

% 训练完成后保存最终模型
finalModelPath = fullfile(config.logDir, 'model_final.mat');
fprintf('训练完成，保存最终模型到: %s\n', finalModelPath);
agent.saveModel(finalModelPath);

% 最终评估
fprintf('进行最终评估...\n');
evalResult = agent.evaluate(20);

% 显示评估结果
fprintf('最终评估结果:\n');
fprintf('  平均回报: %.2f ± %.2f\n', evalResult.meanReturn, evalResult.stdReturn);
fprintf('  最小回报: %.2f\n', evalResult.minReturn);
fprintf('  最大回报: %.2f\n', evalResult.maxReturn);
fprintf('  平均回合长度: %.2f\n', evalResult.meanLength);

% 绘制训练曲线
logger.plotTrainingCurves();

% 可视化控制性能的函数
function visualizePerformance(agent, envName)
    % 创建环境
    env = feval(envName);
    
    % 重置环境
    obs = env.reset();
    done = false;
    
    % 创建图形
    figure('Name', '交流电机控制性能评估', 'Position', [100, 100, 1000, 800]);
    
    % 速度响应子图
    subplot(2, 2, 1);
    speedPlot = plot(0, 0, 'b-', 'LineWidth', 1.5);
    hold on;
    targetPlot = plot(0, 0, 'r--', 'LineWidth', 1.5);
    title('速度响应');
    xlabel('时间 (s)');
    ylabel('速度 (rad/s)');
    legend('实际速度', '目标速度');
    grid on;
    
    % 电流响应子图
    subplot(2, 2, 2);
    idPlot = plot(0, 0, 'b-', 'LineWidth', 1.5);
    hold on;
    iqPlot = plot(0, 0, 'r-', 'LineWidth', 1.5);
    title('d-q轴电流');
    xlabel('时间 (s)');
    ylabel('电流 (A)');
    legend('id', 'iq');
    grid on;
    
    % 控制信号子图
    subplot(2, 2, 3);
    VdPlot = plot(0, 0, 'b-', 'LineWidth', 1.5);
    hold on;
    VqPlot = plot(0, 0, 'r-', 'LineWidth', 1.5);
    title('控制信号 (d-q轴电压)');
    xlabel('时间 (s)');
    ylabel('电压 (V)');
    legend('Vd', 'Vq');
    grid on;
    
    % 转矩子图
    subplot(2, 2, 4);
    tePlot = plot(0, 0, 'b-', 'LineWidth', 1.5);
    hold on;
    tlPlot = plot(0, 0, 'r-', 'LineWidth', 1.5);
    title('转矩');
    xlabel('时间 (s)');
    ylabel('转矩 (N·m)');
    legend('电磁转矩', '负载转矩');
    grid on;
    
    % 数据记录
    timeData = [];
    speedData = [];
    targetSpeedData = [];
    idData = [];
    iqData = [];
    VdData = [];
    VqData = [];
    TeData = [];
    TlData = [];
    
    % 评估最多500步
    maxSteps = 500;
    step = 0;
    
    % 运行一个回合
    while ~done && step < maxSteps
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
        
        % 记录数据
        step = step + 1;
        timeData(end+1) = step * env.dt;
        speedData(end+1) = info.speed;
        targetSpeedData(end+1) = info.targetSpeed;
        idData(end+1) = info.id;
        iqData(end+1) = info.iq;
        VdData(end+1) = action(1) * env.maxVoltage;
        VqData(end+1) = action(2) * env.maxVoltage;
        TeData(end+1) = info.Te;
        TlData(end+1) = info.Tl;
        
        % 每10步更新图形
        if mod(step, 10) == 0
            % 更新速度图
            speedPlot.XData = timeData;
            speedPlot.YData = speedData;
            targetPlot.XData = timeData;
            targetPlot.YData = targetSpeedData;
            
            % 更新电流图
            idPlot.XData = timeData;
            idPlot.YData = idData;
            iqPlot.XData = timeData;
            iqPlot.YData = iqData;
            
            % 更新控制信号图
            VdPlot.XData = timeData;
            VdPlot.YData = VdData;
            VqPlot.XData = timeData;
            VqPlot.YData = VqData;
            
            % 更新转矩图
            tePlot.XData = timeData;
            tePlot.YData = TeData;
            tlPlot.XData = timeData;
            tlPlot.YData = TlData;
            
            % 调整坐标轴
            for i = 1:4
                subplot(2, 2, i);
                xlim([0, max(timeData)]);
            end
            
            % 刷新图形
            drawnow;
        end
    end
    
    % 保存评估图
    saveas(gcf, fullfile(agent.config.logDir, 'performance_evaluation.png'));
    
    % 关闭图形
    pause(1);
    close;
end
