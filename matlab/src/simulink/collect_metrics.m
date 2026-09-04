function M = collect_metrics(P, S, dropped, N, H, cfg)
%COLLECT_METRICS Aggregate simulation counters into the report struct.
%   Pure function of counters (unit-testable with hand-made structs).
%   Queue max/mean are EXACT running counters (not sampled estimates).
%   Throughput/latency/utilization/queues/reliability/workload/bottleneck.
    done = [P.done];
    failed = [P.failed];
    completed = done & ~failed;
    n_done = sum(completed);
    tend = [P.t_end];
    lat = tend(completed) - [P.arrival];
    lat = lat(completed);
    pools = {'capture', 'ai', 'review', 'tele'};
    util = struct();
    for k = 1:numel(pools)
        p = pools{k};
        util.(p) = S.(p).busy / max(H * cfg.servers.(p), eps);
    end
    qmax = struct(); qmean = struct(); qseries = struct();
    for k = 1:numel(pools)
        p = pools{k};
        qmax.(p) = S.(p).qmax;
        if S.(p).qn > 0
            qmean.(p) = S.(p).qsum / S.(p).qn;
        else
            qmean.(p) = 0;
        end
        % Stride-plot series (<=2000 points). Defensive defaults: hand-built
        % structs in tests may omit the plot fields.
        if isfield(S.(p), 'qt'), t = S.(p).qt; else, t = []; end
        if isfield(S.(p), 'ql'), l = S.(p).ql; else, l = []; end
        if numel(t) > 2000
            ix = unique(round(linspace(1, numel(t), 2000)));
            t = t(ix); l = l(ix);
        end
        qseries.(p) = struct('t', t, 'len', l);
    end
    classes = {P(completed).class};
    n_reviewed = sum(ismember(classes, {'REFER', 'URGENT_REVIEW', 'TECHNICAL_REVIEW'}));
    n_tele = sum([P.escalated]);
    rev_hours = S.review.busy / 3600;
    tele_hours = S.tele.busy / 3600;
    [~, bi] = max([util.capture, util.ai, util.review, util.tele]);
    names = {'CAPTURE', 'AI SCREENING', 'HUMAN REVIEW', 'TELE-OPHTHALMOLOGY'};
    waits = struct('capture', mmean([P(completed).wcap]), ...
        'ai', mmean([P(completed).wai]), ...
        'review', mmean([P(completed).wrev]), ...
        'tele', mmean([P(completed).wtele]));
    M = struct('n_arrivals', N, 'n_completed', n_done, ...
        'n_failed', sum(failed), 'n_dropped_overflow', dropped, ...
        'throughput_per_day', n_done / max(cfg.operating_days_per_year, 1), ...
        'throughput_per_year', n_done / max(cfg.operating_days_per_year, 1) * 365, ...
        'annual_volume_target', cfg.annual_patients, ...
        'mean_latency_s', mmean(lat), 'p95_latency_s', mpctl(lat, 95), ...
        'mean_queue_wait_s', waits, ...
        'utilization', struct('capture', r4(util.capture), 'ai', r4(util.ai), ...
            'review', r4(util.review), 'tele', r4(util.tele)), ...
        'queue_max', qmax, 'queue_mean', qmean, 'queue_series', qseries, ...
        'n_reviewed', n_reviewed, 'n_tele', n_tele, ...
        'reviewer_hours', round(rev_hours, 1), 'tele_hours', round(tele_hours, 1), ...
        'retries_total', sum([P.retries]), ...
        'bottleneck', names{bi}, ...
        'bottleneck_utilization', r4(max([util.capture, util.ai, util.review, util.tele])), ...
        'horizon_s', H);
end

function v = mmean(x)
    if isempty(x), v = NaN; else, v = mean(x); end
end

function v = mpctl(x, p)
    if isempty(x), v = NaN; else, v = prctile(x, p); end
end

function v = r4(x)
    v = round(x, 4);
end
