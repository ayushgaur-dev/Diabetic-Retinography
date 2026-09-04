classdef TestSimulation < matlab.unittest.TestCase
%TESTSIMULATION Simulation contracts (run where MATLAB exists).
    methods (Test)
        function arrivalRateMath(testCase)
            cfg = config_simulation();
            cfg.annual_patients = 100000;
            lam = arrival_rate_per_s(cfg);
            testCase.verifyEqual(lam, 100000 / (300 * 8 * 3600), 'AbsTol', 1e-12);
        end
        function scenarioPresets(testCase)
            testCase.verifyEqual(create_scenario('A').annual_patients, 10000);
            testCase.verifyEqual(create_scenario('B').annual_patients, 50000);
            testCase.verifyEqual(create_scenario('C').annual_patients, 100000);
            testCase.verifyEqual(create_scenario('D').annual_patients, 150000);
        end
        function invalidConfigRejected(testCase)
            cfg = config_simulation();
            cfg.annual_patients = -5;
            [ok, errors] = validate_simulation_config(cfg);
            testCase.verifyFalse(ok);
            testCase.verifyNotEmpty(errors);
            bad = config_simulation();
            bad.quality_mix.GOOD = 0.5;   % sums to 1.5
            [ok2, ~] = validate_simulation_config(bad);
            testCase.verifyFalse(ok2);
        end
        function determinismContract(testCase)
            % Same seed + tiny deterministic scenario -> identical reports.
            s1 = create_scenario('custom', 60, 7);
            s1.arrival_model = 'deterministic';
            s2 = create_scenario('custom', 60, 7);
            s2.arrival_model = 'deterministic';
            r1 = simulate_workflow(s1);
            r2 = simulate_workflow(s2);
            testCase.verifyEqual(r1.n_completed, r2.n_completed);
            testCase.verifyEqual(r1.throughput_per_year, r2.throughput_per_year);
            testCase.verifyEqual(r1.bottleneck, r2.bottleneck);
        end
        function capacityMath(testCase)
            % 1 server, 10 s service, horizon 1000 s, 50 deterministic arrivals.
            cfg = config_simulation();
            cfg.annual_patients = 50;
            cfg.operating_days_per_year = 1;
            cfg.operating_hours_per_day = 1000 / 3600;
            cfg.arrival_model = 'deterministic';
            cfg.service_model = 'deterministic';
            cfg.service_s.capture = 10;
            cfg.servers.capture = 1;
            cfg.service_s.ai = 0.01;
            cfg.quality_mix = struct('GOOD', 1, 'BORDERLINE', 0, 'UNGRADABLE', 0);
            cfg.triage_mix = struct('ROUTINE', 1, 'REFER', 0, ...
                'URGENT_REVIEW', 0, 'TECHNICAL_REVIEW', 0);
            rep = simulate_workflow(cfg);
            testCase.verifyEqual(rep.n_completed, 50);
            % busy = 50*10 = 500 s over 1000 s on 1 server -> 0.5
            testCase.verifyEqual(rep.utilization.capture, 0.5, 'AbsTol', 1e-9);
        end
        function queueMetricsOnTrace(testCase)
            % Hand-made counters: review queue hit 3 (mean 2), ai empty.
            P = struct('done', {true, true}, 'failed', {false, false}, ...
                't_end', {10, 20}, 'arrival', {0, 1}, 'class', {'ROUTINE', 'REFER'}, ...
                'retries', {0, 0}, 'escalated', {false, false}, ...
                'wcap', {0, 0}, 'wai', {0, 0}, 'wrev', {0, 5}, 'wtele', {0, 0});
            % NOTE: struct() with cells builds struct arrays in MATLAB.
            S = struct('capture', struct('busy', 1, 'free', 0, 'qmax', 0, ...
                'qsum', 0, 'qn', 0), ...
                'ai', struct('busy', 0, 'free', 0, 'qmax', 0, 'qsum', 0, 'qn', 0), ...
                'review', struct('busy', 4, 'free', 0, 'qmax', 3, ...
                'qsum', 6, 'qn', 3), ...
                'tele', struct('busy', 0, 'free', 0, 'qmax', 0, 'qsum', 0, 'qn', 0));
            cfg = config_simulation();
            cfg.operating_days_per_year = 1;
            cfg.operating_hours_per_day = 1;
            M = collect_metrics(P, S, 0, 2, 3600, cfg);
            testCase.verifyEqual(M.queue_max.review, 3);
            testCase.verifyEqual(M.queue_mean.review, 2);
            testCase.verifyEqual(M.bottleneck, 'HUMAN REVIEW');
        end
        function bottleneckDetection(testCase)
            P = struct('done', true, 'failed', false, 't_end', 5, ...
                'arrival', 0, 'class', 'ROUTINE', 'retries', 0, ...
                'escalated', false, 'wcap', 0, 'wai', 0, 'wrev', 0, 'wtele', 0);
            S = struct('capture', struct('busy', 10, 'free', 0, 'qmax', 0, ...
                'qsum', 0, 'qn', 0), ...
                'ai', struct('busy', 3500, 'free', 0, 'qmax', 0, 'qsum', 0, 'qn', 0), ...
                'review', struct('busy', 5, 'free', 0, 'qmax', 0, 'qsum', 0, 'qn', 0), ...
                'tele', struct('busy', 1, 'free', 0, 'qmax', 0, 'qsum', 0, 'qn', 0));
            cfg = config_simulation();
            M = collect_metrics(P, S, 0, 1, 3600, cfg);
            testCase.verifyEqual(M.bottleneck, 'AI SCREENING');
        end
    end
end
