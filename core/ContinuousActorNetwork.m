classdef ContinuousActorNetwork < handle
    % ContinuousActorNetwork 用于连续动作空间的策略网络
    %   实现了具有高斯分布输出的策略网络
    
    properties
        layerSizes      % 隐藏层大小
        learnables      % 可学习参数
        useGPU          % 是否使用GPU
    end
    
    methods
        function obj = ContinuousActorNetwork(inputSize, outputSize, layerSizes)
            % 构造函数：初始化连续动作空间的Actor网络
            %   inputSize - 输入维度（观察空间大小）
            %   outputSize - 输出维度（动作空间大小）
            %   layerSizes - 隐藏层大小的数组
            
            obj.layerSizes = layerSizes;
            obj.useGPU = false;
            
            % 初始化网络参数
            obj.learnables = struct();
            
            % 初始化第一个隐藏层
            layerIdx = 1;
            obj.learnables.fc1w = dlarray(initializeGlorot(layerSizes(1), inputSize), 'UW');
            obj.learnables.fc1b = dlarray(zeros(layerSizes(1), 1), 'UB');
            
            % 初始化中间隐藏层
            for i = 2:length(layerSizes)
                obj.learnables.(sprintf('fc%dw', i)) = dlarray(initializeGlorot(layerSizes(i), layerSizes(i-1)), 'UW');
                obj.learnables.(sprintf('fc%db', i)) = dlarray(zeros(layerSizes(i), 1), 'UB');
            end
            
            % 初始化均值输出层
            obj.learnables.meanw = dlarray(initializeGlorot(outputSize, layerSizes(end)), 'UW');
            obj.learnables.meanb = dlarray(zeros(outputSize, 1), 'UB');
            
            % 初始化方差输出层 (log of std)
            obj.learnables.logstdw = dlarray(initializeGlorot(outputSize, layerSizes(end)), 'UW');
            obj.learnables.logstdb = dlarray(zeros(outputSize, 1), 'UB');
        end
        
        function [action, logProb, mean, std] = sampleAction(obj, observation)
            % 采样一个动作并计算其对数概率
            %   observation - 当前的观察状态
            %   action - 采样的动作
            %   logProb - 动作的对数概率
            
            % 前向传播，获取动作分布参数
            [mean, logstd] = obj.forward(observation);
            std = exp(logstd);
            
            % 从正态分布采样
            if obj.useGPU
                noise = dlarray(gpuArray(randn(size(mean))), 'CB');
            else
                noise = dlarray(randn(size(mean)), 'CB');
            end
            
            action = mean + std .* noise;
            
            % 计算对数概率
            logProb = -0.5 * sum((action - mean).^2 ./ (std.^2 + 1e-8), 1) - ...
                      sum(logstd, 1) - ...
                      0.5 * size(mean, 1) * log(2 * pi);
        end
        
        function [logProb, entropy] = evaluateActions(obj, observation, action)
            % 评估动作的对数概率和熵
            %   observation - 观察状态
            %   action - 执行的动作
            %   logProb - 动作的对数概率
            %   entropy - 策略的熵
            
            % 前向传播，获取动作分布参数
            [mean, logstd] = obj.forward(observation);
            std = exp(logstd);
            
            % 计算对数概率
            logProb = -0.5 * sum((action - mean).^2 ./ (std.^2 + 1e-8), 1) - ...
                      sum(logstd, 1) - ...
                      0.5 * size(mean, 1) * log(2 * pi);
            
            % 计算熵
            entropy = sum(logstd + 0.5 * log(2 * pi * exp(1)), 1);
        end
        
        function [mean, logstd] = forward(obj, observation)
            % 前向传播，计算动作分布参数
            %   observation - 观察状态
            %   mean - 动作均值
            %   logstd - 动作标准差的对数
            
            % 第一个隐藏层
            x = fullyconnect(observation, obj.learnables.fc1w, obj.learnables.fc1b);
            x = tanh(x);
            
            % 中间隐藏层
            for i = 2:length(obj.layerSizes)
                x = fullyconnect(x, obj.learnables.(sprintf('fc%dw', i)), obj.learnables.(sprintf('fc%db', i)));
                x = tanh(x);
            end
            
            % 均值输出层
            mean = fullyconnect(x, obj.learnables.meanw, obj.learnables.meanb);
            
            % 方差输出层 (log of std)
            logstd = fullyconnect(x, obj.learnables.logstdw, obj.learnables.logstdb);
            
            % 限制logstd范围，防止数值不稳定
            logstd = max(min(logstd, 2), -20);
        end
        
        function action = getMeanAction(obj, observation)
            % 获取确定性动作（均值）
            %   observation - 观察状态
            %   action - 确定性动作
            
            % 前向传播，获取动作分布的均值
            [mean, ~] = obj.forward(observation);
            action = mean;
        end
        
        function toGPU(obj)
            % 将网络参数转移到GPU
            obj.useGPU = true;
            
            % 将所有参数移至GPU
            fnames = fieldnames(obj.learnables);
            for i = 1:length(fnames)
                obj.learnables.(fnames{i}) = gpuArray(obj.learnables.(fnames{i}));
            end
        end
        
        function toCPU(obj)
            % 将网络参数转移回CPU
            obj.useGPU = false;
            
            % 将所有参数移回CPU
            fnames = fieldnames(obj.learnables);
            for i = 1:length(fnames)
                obj.learnables.(fnames{i}) = gather(obj.learnables.(fnames{i}));
            end
        end
        
        function params = getParameters(obj)
            % 获取网络参数
            params = struct2cell(obj.learnables);
        end
        
        function setParameters(obj, params)
            % 设置网络参数
            fnames = fieldnames(obj.learnables);
            for i = 1:length(fnames)
                obj.learnables.(fnames{i}) = params{i};
            end
        end
    end
end

function W = initializeGlorot(numOut, numIn)
    % Glorot/Xavier初始化
    stddev = sqrt(2 / (numIn + numOut));
    W = stddev * randn(numOut, numIn);
end
