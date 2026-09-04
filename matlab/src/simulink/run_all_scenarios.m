function reps = run_all_scenarios(outDir)
%RUN_ALL_SCENARIOS Execute scenarios A-D (10k/50k/100k/150k per year).
%   Report language rule: each scenario reports what the SIMULATION did
%   under its assumptions — never "supports X patients/year" as a claim.
    if nargin < 1, outDir = fullfile(fileparts(mfilename('fullpath')), ...
            '..', '..', 'outputs'); end
    reps = struct();
    for name = {'A', 'B', 'C', 'D'}
        reps.(name{1}) = run_scenario(create_scenario(name{1}), outDir);
    end
    plot_scalability(reps, outDir);
end
