classdef ACMotorEnv < Environment
    % ACMotorEnv 交流感应电机控制环境
    %   使用FOC（磁场定向控制）方法模拟三相异步电机的控制系统
    
    properties
        % 环境规格
        observationSize = 6     % 状态空间维度：[速度误差, id, iq, d轴误差, q轴误差, 负载转矩估计]
        actionSize = 2          % 动作空间维度：[Vd, Vq] - FOC控制的d轴和q轴电压
        isDiscrete = false      % 连续动作空间
        
        % 电机物理参数
        Rs = 2.0           % 定子电阻 (ohm)
        Rr = 2.0           % 转子电阻 (ohm)
        Ls = 0.2           % 定子自感 (H)
        Lr = 0.2           % 转子自感 (H)
        Lm = 0.15          % 互感 (H)
        J = 0.02           % 转动惯量 (kg.m^2)
        p = 2              % 极对数
        B = 0.005          % 阻尼系数 (N.m.s)
        
        % 系统参数
        dt = 0.001         % 时间步长 (s)，FOC控制需要较小的时间步长
        fs = 1000          % 采样频率 (Hz)
        maxVoltage = 400.0 % 最大允许电压 (V)
        maxCurrent = 15.0  % 最大允许电流 (A)
        maxSpeed = 157.0   % 最大允许角速度 (rad/s) ~ 1500rpm
        nominalSpeed = 150 % 额定角速度 (rad/s)
        nominalTorque = 10 % 额定转矩 (N.m)
        
        % 目标
        targetSpeed        % 目标速度 (rad/s)
        
        % 当前状态
        speed              % 实际转速 (rad/s)
        position           % 实际位置 (rad)
        id                 % d轴电流 (A)
        iq                 % q轴电流 (A)
        psi_d              % d轴磁链 (Wb)
        psi_q              % q轴磁链 (Wb)
        Te                 % 电磁转矩 (N.m)
        Tl                 % 负载转矩 (N.m)
        
        % 控制参数（磁场定向控制FOC需要）
        id_ref             % d轴电流参考值 (A), 通常为常数
        iq_ref             % q轴电流参考值 (A), 由速度控制环生成
        
        % 回合信息
        steps = 0          % 当前回合步数
        maxSteps = 5000    % 最大步数（由于采样频率高，需要更多步数）
        
        % 速度控制PI参数
        Kp_speed = 0.5
        Ki_speed = 5.0
        speed_error_integral = 0
        
        % 负载模拟
        loadChangeTime     % 负载突变的时间
        loadProfile        % 负载随时间的变化
        
        % 可视化
        renderFig          % 图形句柄
        renderAx           % 坐标轴句柄
        plotHandles        % 图形元素句柄
        
        % 数据记录
        historyData        % 用于记录系统状态的历史数据
    end
    
    methods
        function obj = ACMotorEnv()
            % 构造函数：初始化交流电机环境
            
            % 初始化状态
            obj.speed = 0;
            obj.position = 0;
            obj.id = 0;
            obj.iq = 0;
            obj.psi_d = 0;
            obj.psi_q = 0;
            obj.Te = 0;
            obj.Tl = 0;
            
            % 设置d轴电流参考值（磁通控制）
            obj.id_ref = 3.0;  % 常量，用于建立磁场
            
            % 随机目标速度
            obj.resetTarget();
            
            % 初始化负载变化
            obj.initializeLoadProfile();
            
            % 初始化历史数据记录
            obj.historyData = struct(...
                'time', [], ...
                'speed', [], ...
                'targetSpeed', [], ...
                'id', [], ...
                'iq', [], ...
                'Te', [], ...
                'Tl', [], ...
                'Vd', [], ...
                'Vq', [] ...
            );
            
            % 随机种子
            rng('shuffle');
        end
        
        function resetTarget(obj)
            % 设置新的随机目标速度
            obj.targetSpeed = (0.5 + 0.5 * rand()) * obj.nominalSpeed; % 50%-100%额定速度
        end
        
        function initializeLoadProfile(obj)
            % 初始化负载变化，模拟工业场景的负载突变
            % 设置负载突变的时间点
            obj.loadChangeTime = [1000, 2000, 3000, 4000]; % 在这些步骤改变负载
            
            % 设置负载变化的配置文件
            obj.loadProfile = [
                0.2 * obj.nominalTorque;  % 初始负载 - 轻载
                0.8 * obj.nominalTorque;  % 第一次变化 - 重载
                0.4 * obj.nominalTorque;  % 第二次变化 - 中等负载
                0.9 * obj.nominalTorque;  % 第三次变化 - 接近满载
                0.3 * obj.nominalTorque;  % 最终负载 - 轻载
            ];
        end
        
        function observation = reset(obj)
            % 重置环境到初始状态
            %   返回初始观察
            
            % 重置电机状态
            obj.speed = 0;
            obj.position = 0;
            obj.id = 0;
            obj.iq = 0;
            obj.psi_d = 0;
            obj.psi_q = 0;
            obj.Te = 0;
            obj.Tl = obj.loadProfile(1); % 初始负载
            
            % 重置控制变量
            obj.iq_ref = 0;
            obj.speed_error_integral = 0;
            
            % 重置步数
            obj.steps = 0;
            
            % 重置目标速度
            obj.resetTarget();
            
            % 重置历史数据
            obj.historyData = struct(...
                'time', [], ...
                'speed', [], ...
                'targetSpeed', [], ...
                'id', [], ...
                'iq', [], ...
                'Te', [], ...
                'Tl', [], ...
                'Vd', [], ...
                'Vq', [] ...
            );
            
            % 返回观察
            observation = obj.getObservation();
        end
        
        function observation = getObservation(obj)
            % 获取当前观察（状态处理和归一化）
            
            % 计算速度误差
            speed_error = obj.targetSpeed - obj.speed;
            
            % 计算电流误差
            id_error = obj.id_ref - obj.id;
            iq_error = obj.iq_ref - obj.iq;
            
            % 组合观察（归一化）
            observation = [
                speed_error / obj.maxSpeed;         % 归一化速度误差
                obj.id / obj.maxCurrent;            % 归一化d轴电流
                obj.iq / obj.maxCurrent;            % 归一化q轴电流
                id_error / obj.maxCurrent;          % 归一化d轴电流误差
                iq_error / obj.maxCurrent;          % 归一化q轴电流误差
                obj.Tl / obj.nominalTorque;         % 归一化负载转矩
            ];
        end
        
        function [nextObs, reward, done, info] = step(obj, action)
            % 执行动作并返回新的状态
            %   action - 要执行的动作（d轴和q轴电压，范围[-1, 1]）
            %   返回值：
            %     nextObs - 新的观察
            %     reward - 获得的奖励
            %     done - 是否回合结束
            %     info - 附加信息
            
            % 将动作范围限制在[-1, 1]
            action = max(-1, min(1, action));
            
            % 将动作转换为d轴和q轴电压
            Vd = action(1) * obj.maxVoltage;
            Vq = action(2) * obj.maxVoltage;
            
            % 更新负载（模拟负载变化）
            obj.updateLoad();
            
            % 更新PI控制器计算iq_ref（速度环）
            % 在实际系统中这通常由PI控制器完成，这里我们让PPO学习这个映射
            speed_error = obj.targetSpeed - obj.speed;
            obj.speed_error_integral = obj.speed_error_integral + speed_error * obj.dt;
            obj.iq_ref = obj.Kp_speed * speed_error + obj.Ki_speed * obj.speed_error_integral;
            obj.iq_ref = max(-obj.maxCurrent, min(obj.maxCurrent, obj.iq_ref));
            
            % 解包当前状态
            id = obj.id;
            iq = obj.iq;
            psi_d = obj.psi_d;
            psi_q = obj.psi_q;
            speed = obj.speed;
            Tl = obj.Tl;
            
            % 计算电磁转矩
            Te = 1.5 * obj.p * obj.Lm / obj.Lr * (iq * psi_d - id * psi_q);
            
            % 机械系统方程 - 转速和位置更新
            % dω/dt = (Te - Tl - B*ω) / J
            speed_new = speed + obj.dt * ((Te - Tl - obj.B * speed) / obj.J);
            position_new = obj.position + obj.dt * speed_new;
            
            % 电气系统方程 - FOC模型
            % 模拟交流电机FOC控制下的动态模型
            sigma = 1 - obj.Lm^2 / (obj.Ls * obj.Lr);
            Tr = obj.Lr / obj.Rr;  % 转子时间常数
            
            % 转子磁通动态方程
            psi_d_new = psi_d + obj.dt * ((-psi_d + obj.Lm * id) / Tr - obj.p * speed * psi_q);
            psi_q_new = psi_q + obj.dt * ((-psi_q + obj.Lm * iq) / Tr + obj.p * speed * psi_d);
            
            % 定子电流动态方程
            id_new = id + obj.dt * ((Vd - obj.Rs * id + obj.p * speed * sigma * obj.Ls * iq) / (sigma * obj.Ls));
            iq_new = iq + obj.dt * ((Vq - obj.Rs * iq - obj.p * speed * (sigma * obj.Ls * id + obj.Lm/obj.Lr * psi_d)) / (sigma * obj.Ls));
            
            % 限制电流
            id_new = max(-obj.maxCurrent, min(obj.maxCurrent, id_new));
            iq_new = max(-obj.maxCurrent, min(obj.maxCurrent, iq_new));
            
            % 更新状态
            obj.id = id_new;
            obj.iq = iq_new;
            obj.psi_d = psi_d_new;
            obj.psi_q = psi_q_new;
            obj.speed = speed_new;
            obj.position = position_new;
            obj.Te = Te;
            
            % 更新步数
            obj.steps = obj.steps + 1;
            
            % 记录历史数据
            obj.updateHistory(Vd, Vq);
            
            % 计算奖励
            reward = obj.calculateReward(speed_error, id_new, iq_new, Te, Vd, Vq);
            
            % 判断是否结束
            done = obj.steps >= obj.maxSteps;
            
            % 返回结果
            nextObs = obj.getObservation();
            info = struct(...
                'steps', obj.steps, ...
                'targetSpeed', obj.targetSpeed, ...
                'speed', obj.speed, ...
                'Te', obj.Te, ...
                'Tl', obj.Tl, ...
                'id', obj.id, ...
                'iq', obj.iq ...
            );
        end
        
        function updateLoad(obj)
            % 更新负载转矩（模拟负载变化）
            for i = 1:length(obj.loadChangeTime)
                if obj.steps == obj.loadChangeTime(i)
                    obj.Tl = obj.loadProfile(i+1);
                    break;
                end
            end
        end
        
        function reward = calculateReward(obj, speed_error, id, iq, Te, Vd, Vq)
            % 计算奖励函数
            
            % 速度误差奖励（负的平方误差）
            speed_reward = -0.5 * (speed_error/obj.maxSpeed)^2;
            
            % d轴电流跟踪奖励（保持磁通）
            id_error = obj.id_ref - id;
            id_reward = -0.3 * (id_error/obj.maxCurrent)^2;
            
            % q轴电流奖励 - 转矩生成
            iq_reward = -0.3 * ((obj.iq_ref - iq)/obj.maxCurrent)^2;
            
            % 能量效率奖励（避免过高电压和电流）
            power_reward = -0.05 * ((Vd/obj.maxVoltage)^2 + (Vq/obj.maxVoltage)^2);
            
            % 综合奖励函数
            reward = speed_reward + id_reward + iq_reward + power_reward;
        end
        
        function updateHistory(obj, Vd, Vq)
            % 更新历史数据，用于绘图
            obj.historyData.time(end+1) = obj.steps * obj.dt;
            obj.historyData.speed(end+1) = obj.speed;
            obj.historyData.targetSpeed(end+1) = obj.targetSpeed;
            obj.historyData.id(end+1) = obj.id;
            obj.historyData.iq(end+1) = obj.iq;
            obj.historyData.Te(end+1) = obj.Te;
            obj.historyData.Tl(end+1) = obj.Tl;
            obj.historyData.Vd(end+1) = Vd;
            obj.historyData.Vq(end+1) = Vq;
        end
        
        function render(obj)
            % 渲染当前环境状态
            
            % 如果没有图形，创建一个
            if isempty(obj.renderFig) || ~isvalid(obj.renderFig)
                obj.renderFig = figure('Name', '交流感应电机FOC控制', 'Position', [100, 100, 1200, 800]);
                
                % 创建四个子图
                % 1. 速度响应
                subplot(2, 2, 1);
                hold on;
                speedPlot = plot(0, 0, 'b-', 'LineWidth', 1.5);
                targetSpeedPlot = plot(0, 0, 'r--', 'LineWidth', 1.5);
                title('速度响应');
                xlabel('时间 (s)');
                ylabel('速度 (rad/s)');
                legend('实际速度', '目标速度');
                grid on;
                
                % 2. 电流响应
                subplot(2, 2, 2);
                hold on;
                idPlot = plot(0, 0, 'b-', 'LineWidth', 1.5);
                iqPlot = plot(0, 0, 'r-', 'LineWidth', 1.5);
                idRefPlot = plot(0, 0, 'b--', 'LineWidth', 1);
                iqRefPlot = plot(0, 0, 'r--', 'LineWidth', 1);
                title('d-q轴电流');
                xlabel('时间 (s)');
                ylabel('电流 (A)');
                legend('id', 'iq', 'id-ref', 'iq-ref');
                grid on;
                
                % 3. 电压控制输入
                subplot(2, 2, 3);
                hold on;
                vdPlot = plot(0, 0, 'b-', 'LineWidth', 1.5);
                vqPlot = plot(0, 0, 'r-', 'LineWidth', 1.5);
                title('d-q轴电压');
                xlabel('时间 (s)');
                ylabel('电压 (V)');
                legend('Vd', 'Vq');
                grid on;
                
                % 4. 转矩和负载
                subplot(2, 2, 4);
                hold on;
                tePlot = plot(0, 0, 'b-', 'LineWidth', 1.5);
                tlPlot = plot(0, 0, 'r-', 'LineWidth', 1.5);
                title('转矩');
                xlabel('时间 (s)');
                ylabel('转矩 (N.m)');
                legend('电磁转矩', '负载转矩');
                grid on;
                
                % 存储图形句柄
                obj.plotHandles = struct(...
                    'speedPlot', speedPlot, ...
                    'targetSpeedPlot', targetSpeedPlot, ...
                    'idPlot', idPlot, ...
                    'iqPlot', iqPlot, ...
                    'idRefPlot', idRefPlot, ...
                    'iqRefPlot', iqRefPlot, ...
                    'vdPlot', vdPlot, ...
                    'vqPlot', vqPlot, ...
                    'tePlot', tePlot, ...
                    'tlPlot', tlPlot ...
                );
            end
            
            % 更新图形
            % 获取历史数据
            time = obj.historyData.time;
            
            % 只显示最后windowSize个点，避免图形过于拥挤
            windowSize = 1000;  % 显示1秒的数据
            if length(time) > windowSize
                startIdx = length(time) - windowSize + 1;
            else
                startIdx = 1;
            end
            
            % 更新速度响应图
            obj.plotHandles.speedPlot.XData = time(startIdx:end);
            obj.plotHandles.speedPlot.YData = obj.historyData.speed(startIdx:end);
            obj.plotHandles.targetSpeedPlot.XData = time(startIdx:end);
            obj.plotHandles.targetSpeedPlot.YData = obj.historyData.targetSpeed(startIdx:end);
            
            % 更新电流图
            obj.plotHandles.idPlot.XData = time(startIdx:end);
            obj.plotHandles.idPlot.YData = obj.historyData.id(startIdx:end);
            obj.plotHandles.iqPlot.XData = time(startIdx:end);
            obj.plotHandles.iqPlot.YData = obj.historyData.iq(startIdx:end);
            
            % 更新参考电流线
            obj.plotHandles.idRefPlot.XData = time(startIdx:end);
            obj.plotHandles.idRefPlot.YData = ones(size(time(startIdx:end))) * obj.id_ref;
            obj.plotHandles.iqRefPlot.XData = time(startIdx:end);
            obj.plotHandles.iqRefPlot.YData = ones(size(time(startIdx:end))) * obj.iq_ref;
            
            % 更新电压图
            obj.plotHandles.vdPlot.XData = time(startIdx:end);
            obj.plotHandles.vdPlot.YData = obj.historyData.Vd(startIdx:end);
            obj.plotHandles.vqPlot.XData = time(startIdx:end);
            obj.plotHandles.vqPlot.YData = obj.historyData.Vq(startIdx:end);
            
            % 更新转矩图
            obj.plotHandles.tePlot.XData = time(startIdx:end);
            obj.plotHandles.tePlot.YData = obj.historyData.Te(startIdx:end);
            obj.plotHandles.tlPlot.XData = time(startIdx:end);
            obj.plotHandles.tlPlot.YData = obj.historyData.Tl(startIdx:end);
            
            % 调整所有子图的X轴范围
            for i = 1:4
                subplot(2, 2, i);
                if ~isempty(time)
                    if length(time) > windowSize
                        xlim([time(end) - (windowSize-1)*obj.dt, time(end)]);
                    else
                        xlim([0, max(time(end), 0.1)]);
                    end
                end
            end
            
            % 速度图Y轴
            subplot(2, 2, 1);
            ylim([0, max(max(obj.historyData.targetSpeed) * 1.2, obj.maxSpeed * 0.5)]);
            
            % 电流图Y轴
            subplot(2, 2, 2);
            maxCurrent = max(max(abs(obj.historyData.id)), max(abs(obj.historyData.iq)));
            ylim([-maxCurrent * 1.2, maxCurrent * 1.2]);
            
            % 电压图Y轴
            subplot(2, 2, 3);
            maxVoltage = max(max(abs(obj.historyData.Vd)), max(abs(obj.historyData.Vq)));
            ylim([-maxVoltage * 1.2, maxVoltage * 1.2]);
            
            % 转矩图Y轴
            subplot(2, 2, 4);
            maxTorque = max(max(obj.historyData.Te), max(obj.historyData.Tl));
            ylim([0, maxTorque * 1.5]);
            
            % 刷新图形
            drawnow;
        end
    end
end
