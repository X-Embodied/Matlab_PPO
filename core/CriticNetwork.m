classdef CriticNetwork < handle
    % CriticNetwork 价值网络
    %   实现了用于估计状态价值的网络
    
    properties
        layerSizes      % 隐藏层大小
        learnables      % 可学习参数
        useGPU          % 是否使用GPU
    end
    
    methods
        function obj = CriticNetwork(inputSize, layerSizes)
            % 构造函数：初始化价值网络
            %   inputSize - 输入维度（观察空间大小）
            %   layerSizes - 隐藏层大小的数组
            
            obj.layerSizes = layerSizes;
            obj.useGPU = false;
            
            % 初始化网络参数
            obj.learnables = struct();
            
            % 初始化第一个隐藏层
            obj.learnables.fc1w = dlarray(initializeGlorot(layerSizes(1), inputSize), 'UW');
            obj.learnables.fc1b = dlarray(zeros(layerSizes(1), 1), 'UB');
            
            % 初始化中间隐藏层
            for i = 2:length(layerSizes)
                obj.learnables.(sprintf('fc%dw', i)) = dlarray(initializeGlorot(layerSizes(i), layerSizes(i-1)), 'UW');
                obj.learnables.(sprintf('fc%db', i)) = dlarray(zeros(layerSizes(i), 1), 'UB');
            end
            
            % 初始化输出层（价值）
            obj.learnables.outw = dlarray(initializeGlorot(1, layerSizes(end)), 'UW');
            obj.learnables.outb = dlarray(zeros(1, 1), 'UB');
        end
        
        function value = getValue(obj, observation)
            % 获取状态价值估计
            %   observation - 观察状态
            %   value - 状态价值估计
            
            % 前向传播
            x = fullyconnect(observation, obj.learnables.fc1w, obj.learnables.fc1b);
            x = tanh(x);
            
            % 中间隐藏层
            for i = 2:length(obj.layerSizes)
                x = fullyconnect(x, obj.learnables.(sprintf('fc%dw', i)), obj.learnables.(sprintf('fc%db', i)));
                x = tanh(x);
            end
            
            % 输出层
            value = fullyconnect(x, obj.learnables.outw, obj.learnables.outb);
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
