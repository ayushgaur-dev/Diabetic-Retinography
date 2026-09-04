function rep = simulate_workflow(cfg)
%SIMULATE_WORKFLOW Discrete-event telemedicine workflow simulation.
%   Event-calendar DES in base MATLAB only (no SimEvents required):
%   exponential variates via -log(rand) (NOT exprnd: that needs the
%   Statistics Toolbox); binary-heap calendar for scale. Deterministic
%   for a fixed cfg.seed. Returns report struct (see collect_metrics).
    [ok, errors] = validate_simulation_config(cfg);
    assert(ok, 'Invalid config: %s', strjoin(errors, '; '));
    rng(cfg.seed);
    H = cfg.operating_days_per_year * cfg.operating_hours_per_day * 3600;
    lambda = cfg.annual_patients / H;

    % ---- arrivals ----
    if strcmp(cfg.arrival_model, 'poisson')
        t = 0; arr = zeros(1, 0);
        while true
            t = t + (-log(rand)) / lambda;
            if t > H, break; end
            arr(end + 1) = t; %#ok<AGROW>
        end
    else
        n = cfg.annual_patients;
        grid = linspace(0, H, n + 2);
        arr = grid(2:end - 1);
    end
    N = numel(arr);

    % ---- server pools ----
    pools = {'capture', 'ai', 'review', 'tele'};
    S = struct();
    for k = 1:numel(pools)
        p = pools{k};
        S.(p).free = zeros(cfg.servers.(p), 1);
        S.(p).queue = zeros(1, 0);
        S.(p).busy = 0;
        S.(p).qmax = 0; S.(p).qsum = 0; S.(p).qn = 0;
        S.(p).qt = zeros(1, 0); S.(p).ql = zeros(1, 0);  % stride-10 plot samples
    end

    % ---- patient state (struct array, one element per patient) ----
    P = struct('arrival', num2cell(arr), 'done', num2cell(false(1, N)), ...
        'failed', num2cell(false(1, N)), 't_end', num2cell(nan(1, N)), ...
        'class', num2cell(repmat("", 1, N)), 'retries', num2cell(zeros(1, N)), ...
        'escalated', num2cell(false(1, N)), 'wcap', num2cell(zeros(1, N)), ...
        'wai', num2cell(zeros(1, N)), 'wrev', num2cell(zeros(1, N)), ...
        'wtele', num2cell(zeros(1, N)), 'qentry', num2cell(nan(1, N)), ...
        'pool', num2cell(repmat("", 1, N)));

    % ---- binary-heap calendar: 1=arrival, 2=svc_done, 3=recapture_ready ----
    HQ_T = arr(:)'; HQ_K = ones(1, N); HQ_I = 1:N;
    HQ_T = HQ_T(:); HQ_K = HQ_K(:); HQ_I = HQ_I(:);
    for k = floor(numel(HQ_T) / 2):-1:1
        [HQ_T, HQ_K, HQ_I] = siftdown(HQ_T, HQ_K, HQ_I, k);
    end

    TR_T = zeros(1, 0); TR_P = zeros(1, 0); TR_L = zeros(1, 0);
    dropped = 0;
    triage_cdf = cumsum([19, 50, 54, 353] / 476);  % measured mix, renormalized
    triage_names = {'ROUTINE', 'REFER', 'URGENT_REVIEW', 'TECHNICAL_REVIEW'};

    while ~isempty(HQ_T)
        [now, kind, i, HQ_T, HQ_K, HQ_I] = heappop(HQ_T, HQ_K, HQ_I);
        switch kind
            case 1  % arrival
                P(i).pool = 'capture'; P(i).qentry = now;
                [S, P, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L, dropped] = ...
                    enqueue(S, P, i, now, 'capture', cfg, ...
                    HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L, dropped, false);
            case 2  % svc_done
                [S, P, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L, dropped] = ...
                    complete(S, P, i, now, cfg, ...
                    HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L, dropped, ...
                    triage_cdf, triage_names);
            case 3  % recapture_ready
                P(i).retries = P(i).retries + 1;
                P(i).pool = 'capture'; P(i).qentry = now;
                [S, P, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L, dropped] = ...
                    enqueue(S, P, i, now, 'capture', cfg, ...
                    HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L, dropped, false);
        end
    end

    rep = collect_metrics(P, S, dropped, N, H, cfg);
end

function [HQ_T, HQ_K, HQ_I] = siftdown(HQ_T, HQ_K, HQ_I, k)
%SIFTDOWN Restore heap below index k (parallel arrays).
    n = numel(HQ_T);
    while true
        l = 2 * k; r = l + 1; smallest = k;
        if l <= n && HQ_T(l) < HQ_T(smallest), smallest = l; end
        if r <= n && HQ_T(r) < HQ_T(smallest), smallest = r; end
        if smallest == k, break; end
        [HQ_T(k), HQ_T(smallest)] = deal(HQ_T(smallest), HQ_T(k));
        [HQ_K(k), HQ_K(smallest)] = deal(HQ_K(smallest), HQ_K(k));
        [HQ_I(k), HQ_I(smallest)] = deal(HQ_I(smallest), HQ_I(k));
        k = smallest;
    end
end

function [HQ_T, HQ_K, HQ_I] = siftup(HQ_T, HQ_K, HQ_I)
%SIFTUP Restore heap after appending at the end.
    k = numel(HQ_T);
    while k > 1
        p = floor(k / 2);
        if HQ_T(k) >= HQ_T(p), break; end
        [HQ_T(k), HQ_T(p)] = deal(HQ_T(p), HQ_T(k));
        [HQ_K(k), HQ_K(p)] = deal(HQ_K(p), HQ_K(k));
        [HQ_I(k), HQ_I(p)] = deal(HQ_I(p), HQ_I(k));
        k = p;
    end
end

function [HQ_T, HQ_K, HQ_I, t, kind, id] = heappush(HQ_T, HQ_K, HQ_I, t, kind, id)
    HQ_T(end + 1, 1) = t; HQ_K(end + 1, 1) = kind; HQ_I(end + 1, 1) = id; %#ok<AGROW>
    [HQ_T, HQ_K, HQ_I] = siftup(HQ_T, HQ_K, HQ_I);
end

function [now, kind, i, HQ_T, HQ_K, HQ_I] = heappop(HQ_T, HQ_K, HQ_I)
    now = HQ_T(1); kind = HQ_K(1); i = HQ_I(1);
    HQ_T(1) = HQ_T(end); HQ_K(1) = HQ_K(end); HQ_I(1) = HQ_I(end);
    HQ_T(end) = []; HQ_K(end) = []; HQ_I(end) = [];
    if ~isempty(HQ_T)
        [HQ_T, HQ_K, HQ_I] = siftdown(HQ_T, HQ_K, HQ_I, 1);
    end
end

function [S, P, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L, dropped] = ...
        enqueue(S, P, i, now, pool, cfg, HQ_T, HQ_K, HQ_I, ...
        TR_T, TR_P, TR_L, dropped, urgent)
%ENQUEUE Join a pool queue (capacity-checked) and start service if free.
    q = S.(pool).queue;
    if numel(q) >= cfg.queue_cap.(pool)
        dropped = dropped + 1;
        P(i).failed = true; P(i).done = true; P(i).t_end = now;
        return;
    end
    if urgent, q = [i, q]; else, q = [q, i]; end %#ok<AGROW>
    S.(pool).queue = q;
    [TR_T, TR_P, TR_L] = trace(TR_T, TR_P, TR_L, now, pool, numel(q));
    [S, P, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L] = ...
        pump(S, P, now, pool, cfg, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L);
end

function wfield = waitfield(pool)
%WAITFIELD Patient wait accumulator per pool.
    switch pool
        case 'capture', wfield = 'wcap';
        case 'ai', wfield = 'wai';
        case 'review', wfield = 'wrev';
        case 'tele', wfield = 'wtele';
    end
end

function [S, P, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L] = ...
        pump(S, P, now, pool, cfg, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L)
%PUMP Start queued jobs on free servers; schedule completions.
    free = find(S.(pool).free <= now);
    while ~isempty(free) && ~isempty(S.(pool).queue)
        i = S.(pool).queue(1);
        S.(pool).queue(1) = [];
        [TR_T, TR_P, TR_L] = trace(TR_T, TR_P, TR_L, now, pool, numel(S.(pool).queue));
        srv = free(1); free(1) = [];
        wf = waitfield(pool);
        P(i).(wf) = P(i).(wf) + (now - P(i).qentry);
        dur = svctime(pool, cfg);
        S.(pool).free(srv) = now + dur;
        S.(pool).busy = S.(pool).busy + dur;
        [HQ_T, HQ_K, HQ_I, ~, ~, ~] = heappush(HQ_T, HQ_K, HQ_I, now + dur, 2, i);
        P(i).pool = pool;
    end
end

function d = svctime(pool, cfg)
%SVCTIME Service duration: exponential (base-MATLAB) or deterministic.
    switch pool
        case 'capture', m = cfg.service_s.capture;
        case 'ai', m = cfg.service_s.ai;
        case 'review', m = cfg.service_s.review;
        case 'tele', m = cfg.service_s.tele + cfg.service_s.network_delay;
    end
    if strcmp(cfg.service_model, 'exponential')
        d = (-log(rand)) * m;
    else
        d = m;
    end
end

function [S, P, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L, dropped] = ...
        complete(S, P, i, now, cfg, HQ_T, HQ_K, HQ_I, ...
        TR_T, TR_P, TR_L, dropped, triage_cdf, triage_names)
%COMPLETE Route a finished service through quality/AI/review/tele logic.
    pool = P(i).pool;
    switch pool
        case 'capture'
            r = rand;
            qm = cfg.quality_mix;
            if r < qm.UNGRADABLE
                [P, HQ_T, HQ_K, HQ_I] = recapture_or_fail(P, i, now, cfg, HQ_T, HQ_K, HQ_I);
            elseif r < qm.UNGRADABLE + qm.BORDERLINE
                if rand < cfg.enhance_accept
                    P(i).pool = 'ai'; P(i).qentry = now;
                    [S, P, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L, dropped] = ...
                        enqueue(S, P, i, now, 'ai', cfg, HQ_T, HQ_K, HQ_I, ...
                        TR_T, TR_P, TR_L, dropped, false);
                else
                    [P, HQ_T, HQ_K, HQ_I] = recapture_or_fail(P, i, now, cfg, HQ_T, HQ_K, HQ_I);
                end
            else
                P(i).pool = 'ai'; P(i).qentry = now;
                [S, P, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L, dropped] = ...
                    enqueue(S, P, i, now, 'ai', cfg, HQ_T, HQ_K, HQ_I, ...
                    TR_T, TR_P, TR_L, dropped, false);
            end
        case 'ai'
            if rand < cfg.ai_failure_prob
                if P(i).retries >= cfg.ai_retries + 1
                    P(i).failed = true; P(i).done = true; P(i).t_end = now;
                else
                    P(i).retries = P(i).retries + 1;
                    P(i).pool = 'ai'; P(i).qentry = now;
                    [S, P, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L, dropped] = ...
                        enqueue(S, P, i, now, 'ai', cfg, HQ_T, HQ_K, HQ_I, ...
                        TR_T, TR_P, TR_L, dropped, false);
                end
            else
                k = find(rand <= triage_cdf, 1, 'first');
                P(i).class = triage_names{k};
                if strcmp(P(i).class, 'ROUTINE')
                    P(i).done = true; P(i).t_end = now;
                else
                    urgent = strcmp(P(i).class, 'URGENT_REVIEW');
                    P(i).pool = 'review'; P(i).qentry = now;
                    [S, P, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L, dropped] = ...
                        enqueue(S, P, i, now, 'review', cfg, HQ_T, HQ_K, HQ_I, ...
                        TR_T, TR_P, TR_L, dropped, urgent);
                end
            end
        case 'review'
            p = cfg.tele_escalate.(P(i).class);
            if rand < p
                P(i).escalated = true;
                P(i).pool = 'tele'; P(i).qentry = now;
                [S, P, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L, dropped] = ...
                    enqueue(S, P, i, now, 'tele', cfg, HQ_T, HQ_K, HQ_I, ...
                    TR_T, TR_P, TR_L, dropped, strcmp(P(i).class, 'URGENT_REVIEW'));
            else
                P(i).done = true; P(i).t_end = now;
            end
        case 'tele'
            P(i).done = true; P(i).t_end = now;
    end
    [S, P, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L] = ...
        pump(S, P, now, pool, cfg, HQ_T, HQ_K, HQ_I, TR_T, TR_P, TR_L);
end

function [P, HQ_T, HQ_K, HQ_I] = recapture_or_fail(P, i, now, cfg, HQ_T, HQ_K, HQ_I)
%RECAPTURE_OR_FAIL One recapture allowed; second quality failure ends the job.
    if P(i).retries >= 1
        P(i).failed = true; P(i).done = true; P(i).t_end = now;
    else
        [HQ_T, HQ_K, HQ_I, ~, ~, ~] = heappush(HQ_T, HQ_K, HQ_I, ...
            now + cfg.service_s.recapture_delay, 3, i);
    end
end

function [S, TR_T, TR_P, TR_L] = trace(S, TR_T, TR_P, TR_L, now, pool, len)
%TRACE Exact running max/mean counters + stride-10 samples (pool 1..4).
%   Counters (not the samples) drive metrics; samples are plot-only and
%   stride-capped so memory stays bounded at 150k volumes.
    S.(pool).qmax = max(S.(pool).qmax, len);
    S.(pool).qsum = S.(pool).qsum + len;
    S.(pool).qn = S.(pool).qn + 1;
    if mod(S.(pool).qn, 10) == 0
        S.(pool).qt(end + 1) = now; %#ok<AGROW>
        S.(pool).ql(end + 1) = len; %#ok<AGROW>
        code = find(strcmp({'capture', 'ai', 'review', 'tele'}, pool));
        TR_T(end + 1) = now; %#ok<AGROW>
        TR_P(end + 1) = code; %#ok<AGROW>
        TR_L(end + 1) = len; %#ok<AGROW>
    end
end
