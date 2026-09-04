function rep = run_scenario(scenario, outDir)
%RUN_SCENARIO Validate -> simulate -> metrics -> figures -> saved report.
%   rep = run_scenario(create_scenario('C'), 'matlab/outputs')
%   Returns the report struct; also writes scenario_<name>.mat.
    if nargin < 2, outDir = fullfile(fileparts(mfilename('fullpath')), ...
            '..', '..', 'outputs'); end
    [ok, errors] = validate_simulation_config(scenario);
    assert(ok, 'Invalid scenario: %s', strjoin(errors, '; '));
    rep = simulate_workflow(scenario);
    rep.scenario_name = scenario.scenario_name;
    rep.seed = scenario.seed;
    rep.config_snapshot = rmfield(scenario, 'scenario_name');
    if ~isfolder(outDir), mkdir(outDir); end
    save(fullfile(outDir, sprintf('scenario_%s.mat', scenario.scenario_name)), 'rep');
    try
        plot_results(rep, outDir);
    catch e
        warning('plot_results failed: %s', e.message);
    end
    fprintf('Scenario %s: %d completed, bottleneck %s (%.0f%%).\n', ...
        scenario.scenario_name, rep.n_completed, rep.bottleneck, ...
        100 * rep.bottleneck_utilization);
end
