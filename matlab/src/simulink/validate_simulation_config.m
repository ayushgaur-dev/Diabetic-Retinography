function [ok, errors] = validate_simulation_config(cfg)
%VALIDATE_SIMULATION_CONFIG Range/type checks. Returns ok + error strings.
    errors = {};
    mustPos = {'annual_patients', 'operating_days_per_year', 'operating_hours_per_day'};
    for k = 1:numel(mustPos)
        if ~isfield(cfg, mustPos{k}) || ~(cfg.(mustPos{k}) > 0)
            errors{end+1} = sprintf('%s must be positive', mustPos{k}); %#ok<AGROW>
        end
    end
    if ~ismember(cfg.arrival_model, {'poisson', 'deterministic'})
        errors{end+1} = 'arrival_model must be poisson|deterministic'; %#ok<AGROW>
    end
    if ~ismember(cfg.service_model, {'exponential', 'deterministic'})
        errors{end+1} = 'service_model must be exponential|deterministic'; %#ok<AGROW>
    end
    qm = [cfg.quality_mix.GOOD, cfg.quality_mix.BORDERLINE, cfg.quality_mix.UNGRADABLE];
    if any(qm < 0) || abs(sum(qm) - 1) > 1e-9
        errors{end+1} = 'quality_mix must be non-negative and sum to 1'; %#ok<AGROW>
    end
    if cfg.enhance_accept < 0 || cfg.enhance_accept > 1
        errors{end+1} = 'enhance_accept must be in [0,1]'; %#ok<AGROW>
    end
    for f = {'capture', 'ai', 'review', 'tele'}
        if cfg.servers.(f{1}) < 1 || floor(cfg.servers.(f{1})) ~= cfg.servers.(f{1})
            errors{end+1} = sprintf('servers.%s must be a positive integer', f{1}); %#ok<AGROW>
        end
        if cfg.queue_cap.(f{1}) < 1
            errors{end+1} = sprintf('queue_cap.%s must be positive', f{1}); %#ok<AGROW>
        end
    end
    if cfg.ai_failure_prob < 0 || cfg.ai_failure_prob > 1
        errors{end+1} = 'ai_failure_prob must be in [0,1]'; %#ok<AGROW>
    end
    ok = isempty(errors);
end
