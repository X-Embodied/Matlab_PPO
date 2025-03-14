classdef DCMotorEnv < Environment
    % DCMotorEnv 直流电机控制环境
    %   模拟直流电机系统的动态特性和控制问题
    
    properties
        % 环境规格
        observationSize = 3     % 状态空间维度：[角度, 角速度, 电流]
        actionSize = 1          % 动作空间维度：施加电压
        isDiscrete = false      % 连续动作空间
        
        % 电机物理参数
        J = 0.01           % 转动惯量 (kg.m^2)
        b = 0.1            % 阻尼系数 (N.m.s)
        K = 0.01           % 电机常数 (N.m/A 或 V.s/rad)
        R = 1.0            % 电阻 (ohm)
        L = 0.5            % 电感 (H)
        
        % 系统参数
        dt = 0.01          % 时间步长 (s)
        maxVoltage = 24.0  % 最大允许电压 (V)
        maxCurrent = 10.0  % 最大允许电流 (A)
        maxSpeed = 50.0    % 最大允许角速度 (rad/s)
        
        % 目标
        targetAngle        % 目标角度 (rad)
        
        % 当前状态
        state              % [角度, 角速度, 电流]
        
        % 回合信息
        steps = 0          % 当前回合步数
        maxSteps = 500     % 最大步数
        
        % 可视化
        renderFig          % 图形句柄
        renderAx           % 坐标轴句柄
        plotHandles        % 图形元素句柄
    end
    
    methods
        function obj = DCMotorEnv()
            % 构造函数：初始化电机环境
            
            % 初始化状态
            obj.state = zeros(3, 1);
            
            % 随机目标角度 (0到2π)
            obj.resetTarget();
            
            % 随机种子
            rng('shuffle');
        end
        
        function resetTarget(obj)
            % 设置新的随机目标角度
            obj.targetAngle = 2 * pi * rand();
        end
        
        function observation = reset(obj)
            % 重置环境到初始状态
            %   返回初始观察
            
            % 随机初始角度 (0到2π)
            initialAngle = 2 * pi * rand();
            
            % 初始状态：[角度, 角速度, 电流]
            obj.state = [initialAngle; 0; 0];
            
            % 重置步数
            obj.steps = 0;
            
            % 重置目标
            obj.resetTarget();
            
            % 返回观察
            observation = obj.getObservation();
        end
        
        function observation = getObservation(obj)
            % 获取当前观察（可以包括状态处理）
            
            % 标准化角度差异（在-π到π之间）
            angleDiff = obj.state(1) - obj.targetAngle;
            angleDiff = atan2(sin(angleDiff), cos(angleDiff));
            
            % 组合观察：[角度差, 角速度, 电流]
            observation = [angleDiff; obj.state(2)/obj.maxSpeed; obj.state(3)/obj.maxCurrent];
        end
        
        function [nextObs, reward, done, info] = step(obj, action)
            % 执行动作并返回新的状态
            %   action - 要执行的动作（施加电压，范围[-1, 1]）
            %   返回值：
            %     nextObs - 新的观察
            %     reward - 获得的奖励
            %     done - 是否回合结束
            %     info - 附加信息
            
            % 验证动作（在电压限制内）
            action = max(-1, min(1, action)); % 裁剪到[-1, 1]
            
            % 将动作转换为电压
            voltage = action * obj.maxVoltage;
            
            % 解包状态
            angle = obj.state(1);
            angularVelocity = obj.state(2);
            current = obj.state(3);
            
            % 物理模型 (电机动力学)
            % dθ/dt = ω
            % dω/dt = (K*i - b*ω) / J
            % di/dt = (V - R*i - K*ω) / L
            
            % 欧拉方法更新状态
            angleNext = angle + obj.dt * angularVelocity;
            angularVelocityNext = angularVelocity + obj.dt * ((obj.K * current - obj.b * angularVelocity) / obj.J);
            currentNext = current + obj.dt * ((voltage - obj.R * current - obj.K * angularVelocity) / obj.L);
            
            % 限制电流
            currentNext = max(-obj.maxCurrent, min(obj.maxCurrent, currentNext));
            
            % 限制角速度
            angularVelocityNext = max(-obj.maxSpeed, min(obj.maxSpeed, angularVelocityNext));
            
            % 角度归一化到[0, 2π)
            angleNext = mod(angleNext, 2 * pi);
            
            % 更新状态
            obj.state = [angleNext; angularVelocityNext; currentNext];
            obj.steps = obj.steps + 1;
            
            % 计算与目标的角度差
            angleDiff = angleNext - obj.targetAngle;
            angleDiff = atan2(sin(angleDiff), cos(angleDiff)); % 归一化到[-π, π]
            
            % 计算与目标的距离（角度误差）
            distance = abs(angleDiff);
            
            % 判断是否完成
            angleThreshold = 0.05;  % 角度误差阈值（弧度）
            speedThreshold = 0.1;   % 速度阈值（rad/s）
            
            targetReached = distance < angleThreshold && abs(angularVelocityNext) < speedThreshold;
            timeout = obj.steps >= obj.maxSteps;
            
            done = targetReached || timeout;
            
            % 计算奖励
            % 角度误差奖励（接近目标给更高奖励）
            angleReward = -distance^2;
            
            % 速度惩罚（过高的速度给予惩罚）
            speedPenalty = -0.01 * (angularVelocityNext^2);
            
            % 电压使用惩罚（鼓励使用较小的控制信号）
            voltagePenalty = -0.01 * (voltage^2 / (obj.maxVoltage^2));
            
            % 电流惩罚（避免过高电流）
            currentPenalty = -0.01 * (currentNext^2 / (obj.maxCurrent^2));
            
            % 目标达成奖励
            successReward = targetReached ? 10.0 : 0.0;
            
            % 总奖励
            reward = angleReward + speedPenalty + voltagePenalty + currentPenalty + successReward;
            
            % 返回结果
            nextObs = obj.getObservation();
            info = struct('steps', obj.steps, 'targetAngle', obj.targetAngle, 'distance', distance);
        end
        
        function render(obj)
            % 渲染当前环境状态
            
            % 如果没有图形，创建一个
            if isempty(obj.renderFig) || ~isvalid(obj.renderFig)
                obj.renderFig = figure('Name', '直流电机控制', 'Position', [100, 100, 1000, 600]);
                
                % 创建两个子图
                subplot(2, 1, 1);
                obj.renderAx = gca;
                title('电机角度');
                hold(obj.renderAx, 'on');
                grid(obj.renderAx, 'on');
                
                % 绘制目标角度
                targetLine = line(obj.renderAx, [0, cos(obj.targetAngle)], [0, sin(obj.targetAngle)], ...
                    'Color', 'g', 'LineWidth', 2, 'LineStyle', '--');
                
                % 绘制电机轴
                motorLine = line(obj.renderAx, [0, cos(obj.state(1))], [0, sin(obj.state(1))], ...
                    'Color', 'b', 'LineWidth', 3);
                
                % 绘制电机本体
                t = linspace(0, 2*pi, 100);
                motorBody = fill(0.2*cos(t), 0.2*sin(t), 'r');
                
                % 设置图形属性
                axis(obj.renderAx, 'equal');
                xlim(obj.renderAx, [-1.5, 1.5]);
                ylim(obj.renderAx, [-1.5, 1.5]);
                
                % 存储图形句柄
                obj.plotHandles = struct('targetLine', targetLine, 'motorLine', motorLine, 'motorBody', motorBody);
                
                % 创建下方子图显示状态信息
                subplot(2, 1, 2);
                infoAx = gca;
                hold(infoAx, 'on');
                grid(infoAx, 'on');
                
                % 时间序列数据
                timeData = zeros(1, 100);
                angleData = zeros(1, 100);
                speedData = zeros(1, 100);
                currentData = zeros(1, 100);
                
                % 绘制时间序列
                anglePlot = plot(infoAx, timeData, angleData, 'b-', 'LineWidth', 1.5);
                speedPlot = plot(infoAx, timeData, speedData, 'r-', 'LineWidth', 1.5);
                currentPlot = plot(infoAx, timeData, currentData, 'g-', 'LineWidth', 1.5);
                
                % 设置图形属性
                title(infoAx, '系统状态');
                xlabel(infoAx, '时间步');
                ylabel(infoAx, '状态值');
                legend(infoAx, {'角度差', '角速度', '电流'}, 'Location', 'best');
                xlim(infoAx, [0, 100]);
                ylim(infoAx, [-1.5, 1.5]);
                
                % 存储时间序列信息
                obj.plotHandles.timeData = timeData;
                obj.plotHandles.angleData = angleData;
                obj.plotHandles.speedData = speedData;
                obj.plotHandles.currentData = currentData;
                obj.plotHandles.anglePlot = anglePlot;
                obj.plotHandles.speedPlot = speedPlot;
                obj.plotHandles.currentPlot = currentPlot;
                obj.plotHandles.infoAx = infoAx;
            end
            
            % 更新顶部电机图
            subplot(2, 1, 1);
            
            % 更新目标线
            obj.plotHandles.targetLine.XData = [0, cos(obj.targetAngle)];
            obj.plotHandles.targetLine.YData = [0, sin(obj.targetAngle)];
            
            % 更新电机轴线
            obj.plotHandles.motorLine.XData = [0, cos(obj.state(1))];
            obj.plotHandles.motorLine.YData = [0, sin(obj.state(1))];
            
            % 更新下方状态图
            subplot(2, 1, 2);
            
            % 获取当前观察
            obs = obj.getObservation();
            
            % 更新时间序列数据
            obj.plotHandles.timeData = [obj.plotHandles.timeData(2:end), obj.steps];
            obj.plotHandles.angleData = [obj.plotHandles.angleData(2:end), obs(1)];
            obj.plotHandles.speedData = [obj.plotHandles.speedData(2:end), obs(2)];
            obj.plotHandles.currentData = [obj.plotHandles.currentData(2:end), obs(3)];
            
            % 更新图形
            obj.plotHandles.anglePlot.XData = obj.plotHandles.timeData;
            obj.plotHandles.anglePlot.YData = obj.plotHandles.angleData;
            obj.plotHandles.speedPlot.XData = obj.plotHandles.timeData;
            obj.plotHandles.speedPlot.YData = obj.plotHandles.speedData;
            obj.plotHandles.currentPlot.XData = obj.plotHandles.timeData;
            obj.plotHandles.currentPlot.YData = obj.plotHandles.currentData;
            
            % 调整X轴范围以始终显示最近的100个时间步
            if obj.steps > 100
                xlim(obj.plotHandles.infoAx, [obj.steps-100, obj.steps]);
            end
            
            % 刷新图形
            drawnow;
        end
    end
end
