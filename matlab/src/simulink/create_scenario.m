function sc = create_scenario(name, annual_patients, seed)
%CREATE_SCENARIO Named scenario from presets (A-D) or custom volume.
%   sc = create_scenario('C', [], [])            -> 100k preset
%   sc = create_scenario('custom', 25000, 7)     -> custom volume
    cfg = config_simulation();
    presets = struct('A', 10000, 'B', 50000, 'C', 100000, 'D', 150000);
    if isfield(presets, name)
        cfg.annual_patients = presets.(name);
    else
        assert(annual_patients > 0, 'custom scenario needs annual_patients');
        cfg.annual_patients = annual_patients;
    end
    if nargin >= 3 && ~isempty(seed), cfg.seed = seed; end
    cfg.scenario_name = name;
    [ok, errors] = validate_simulation_config(cfg);
    assert(ok, 'Invalid scenario: %s', strjoin(errors, '; '));
    sc = cfg;
end
