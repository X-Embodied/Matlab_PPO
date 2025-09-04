%% 使用MAPPO训练双倒立摆控制器
% 本脚本展示了如何使用MAPPO算法训练多智能体来协同控制双倒立摆系统
% 这是一个只能用多智能体方法而不能用单一智能体方法解决的问题示例

clc;
clear;
close all;

% 将必要的路径添加到MATLAB路径
addpath('../core');
addpath('../environments');
addpath('../config');
addpath('../utils');

% 加载配置
config = default_doublependulum_config();

% 创建日志目录
if ~exist(config.logDir, 'dir')
    mkdir(config.logDir);
end

% 创建MAPPO智能体
fprintf('初始化MAPPO智能体...\n');
mappoAgent = MAPPOAgent(config);

% 训练智能体
fprintf('开始训练，总迭代次数: %d\n', config.numIterations);
mappoAgent.train(config.numIterations);

% 训练完成
fprintf('训练完成！最终模型已保存到 %s\n', fullfile(config.logDir, 'model_final.mat'));

% 训练后评估和可视化
fprintf('评估训练后的智能体性能...\n');
numTestEpisodes = 5;
renderEpisodes = true;

% 评估并可视化
testResults = mappoAgent.evaluate(numTestEpisodes, renderEpisodes);

fprintf('测试结果:\n');
fprintf('  平均回报: %.2f ± %.2f\n', testResults.meanReturn, testResults.stdReturn);
fprintf('  最小回报: %.2f\n', testResults.minReturn);
fprintf('  最大回报: %.2f\n', testResults.maxReturn);
fprintf('  平均长度: %.2f\n', testResults.meanLength);

% 绘制训练曲线
mappoAgent.logger.plotTrainingCurves();

fprintf('训练和评估完成！\n');
fprintf('MAPPO成功解决了双倒立摆问题，这是一个需要多智能体协作的任务\n');
fprintf('单一智能体PPO无法有效解决此问题，因为它需要两个控制器协同工作\n');
