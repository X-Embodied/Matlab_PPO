classdef (Abstract) Environment < handle
    % Environment 强化学习环境的基类
    %   所有环境都应该继承这个基类并实现其方法
    
    properties (Abstract)
        observationSize     % 观察空间大小
        actionSize          % 动作空间大小
        isDiscrete          % 是否为离散动作空间
    end
    
    methods (Abstract)
        % 重置环境到初始状态
        %   返回初始观察
        obs = reset(obj)
        
        % 执行动作并返回新的状态
        %   action - 要执行的动作
        %   返回值：
        %     nextObs - 新的观察
        %     reward - 获得的奖励
        %     done - 是否回合结束
        %     info - 附加信息（可选）
        [nextObs, reward, done, info] = step(obj, action)
        
        % 渲染环境（可选）
        render(obj)
    end
    
    methods
        function validateAction(obj, action)
            % 验证动作是否合法
            %   action - 要验证的动作
            
            assert(length(action) == obj.actionSize, ...
                '动作维度不匹配：期望 %d，实际 %d', obj.actionSize, length(action));
        end
        
        function obs = normalizeObservation(obj, obs)
            % 标准化观察（子类可以重写此方法）
            %   obs - 原始观察
            %   返回值：标准化后的观察
        end
        
        function action = normalizeAction(obj, action)
            % 标准化动作（子类可以重写此方法）
            %   action - 原始动作
            %   返回值：标准化后的动作
        end
        
        function actionValues = discreteToBox(obj, action)
            % 将离散动作转换为连续值（对于具有离散动作空间的环境）
            %   action - 离散动作索引
            %   返回值：连续动作值
        end
        
        function actionIndex = boxToDiscrete(obj, action)
            % 将连续动作值转换为离散动作索引（对于具有离散动作空间的环境）
            %   action - 连续动作值
            %   返回值：离散动作索引
        end
        
        function seed(obj, seedValue)
            % 设置随机种子以便结果可复现（子类可以重写此方法）
            %   seedValue - 随机种子值
            rng(seedValue);
        end
        
        function info = getEnvInfo(obj)
            % 获取环境信息
            %   返回值：包含环境信息的结构体
            
            info = struct();
            info.observationSize = obj.observationSize;
            info.actionSize = obj.actionSize;
            info.isDiscrete = obj.isDiscrete;
        end
    end
end
