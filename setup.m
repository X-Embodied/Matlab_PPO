function setup()
    % Matlab PPO强化学习框架
    % 版本: v1.0.0
    % 作者: AIResearcherHZ
    % 许可证: MIT
    % GitHub: https://github.com/AIResearcherHZ/Matlab_PPO
    
    % 检查必需的工具箱
    required_toolboxes = {
        'Deep Learning Toolbox',
        'Optimization Toolbox',
        'Control System Toolbox'
    };
    
    % 检查可选的工具箱
    optional_toolboxes = {
        'Parallel Computing Toolbox'
    };
    
    % 验证必需工具箱
    v = ver;
    installed_toolboxes = {v.Name};
    missing_toolboxes = setdiff(required_toolboxes, installed_toolboxes);
    
    if ~isempty(missing_toolboxes)
        error('缺少必需的工具箱：\n%s', strjoin(missing_toolboxes, '\n'));
    end
    
    % 检查可选工具箱并提供警告
    missing_optional = setdiff(optional_toolboxes, installed_toolboxes);
    if ~isempty(missing_optional)
        warning('以下可选工具箱未安装，某些功能可能受限：\n%s', strjoin(missing_optional, '\n'));
    end
    
    % 添加所有子目录到MATLAB路径
    addpath(genpath(fileparts(mfilename('fullpath'))));
    
    fprintf('MATLAB PPO强化学习框架设置完成！\n');
end