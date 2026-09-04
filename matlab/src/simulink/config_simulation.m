function cfg = config_simulation()
%CONFIG_SIMULATION Default scenario parameters with provenance labels.
%   Every value is tagged 'measured' (this repo's engineering evidence) or
%   'assumption' (planning placeholder). Nothing here is a clinical fact.
    cfg = struct();
    cfg.annual_patients = 100000;
    cfg.operating_days_per_year = 300;
    cfg.operating_hours_per_day = 8;
    cfg.arrival_model = 'poisson';   % 'poisson' | 'deterministic'
    cfg.seed = 26038;
    % Service pools: servers, mean service seconds, time model.
    % AI grading ~0.15 s measured (Phase 5 runtimes) + margin -> 0.5 s.
    cfg.servers = struct('capture', 2, 'ai', 4, 'review', 2, 'tele', 1);
    cfg.service_s = struct('capture', 120, 'quality', 0.01, 'enhancement', 0.02, ...
        'ai', 0.5, 'review', 240, 'tele', 600, 'network_delay', 30, ...
        'recapture_delay', 300);
    cfg.service_model = 'exponential';  % 'exponential' | 'deterministic'
    % Measured on APTOS test (this repo): GOOD 0.5%, BORDERLINE 86%, UNGRADABLE 13.5%.
    cfg.quality_mix = struct('GOOD', 0.005, 'BORDERLINE', 0.86, 'UNGRADABLE', 0.135);
    cfg.enhance_accept = 0.27;          % measured (Phase 5 ablation)
    % Measured Phase 8 triage mix among graded images (renormalized in engine).
    cfg.triage_mix = struct('ROUTINE', 19, 'REFER', 50, 'URGENT_REVIEW', 54, ...
        'TECHNICAL_REVIEW', 353);
    cfg.tele_escalate = struct('REFER', 0.4, 'URGENT_REVIEW', 0.8, ...
        'TECHNICAL_REVIEW', 0.2);       % assumptions
    cfg.ai_failure_prob = 0.001;        % assumption
    cfg.ai_retries = 1;
    cfg.queue_cap = struct('capture', 200, 'ai', 500, 'review', 200, 'tele', 100);
    cfg.provenance = 'mixed: service/quality/triage rates measured on repo data; staffing, review/tele times, network, capture are ENGINEERING SCENARIO ASSUMPTIONS';
end
