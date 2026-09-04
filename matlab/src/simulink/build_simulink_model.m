function sys = build_simulink_model(modelName)
%BUILD_SIMULINK_MODEL Programmatically construct the telemedicine workflow
%   model (10 documented subsystems). Run where Simulink exists:
%     >> build_simulink_model('telemedicine_workflow')
%   SimEvents is NOT assumed: the structure uses standard blocks with
%   MATLAB Function blocks calling the engine's pure routing functions;
%   a SimEvents variant path is sketched in comments. The executable
%   simulation of record is simulate_workflow.m (base MATLAB).
    if nargin < 1, modelName = 'telemedicine_workflow'; end
    assert(logical(exist('new_system', 'file')), ...
        'Simulink is required to build the .slx model.');
    if bdIsLoaded(modelName), close_system(modelName, 0); end
    new_system(modelName);
    open_system(modelName);
    subs = {'01 Patient Arrival', '02 Image Capture', '03 Quality Gate', ...
        '04 Enhancement', '05 AI Screening', '06 Triage', ...
        '07 Local Review', '08 Tele-Ophthalmology', '09 Completion', ...
        '10 Metrics'};
    x = 100;
    prev = '';
    for k = 1:numel(subs)
        blk = [modelName '/' subs{k}];
        add_block('built-in/Subsystem', blk, 'Position', [x 100 x+140 200]);
        add_block('built-in/Inport', [blk '/in'], 'Position', [x+10 120 x+30 140]);
        add_block('built-in/Outport', [blk '/out'], 'Position', [x+110 120 x+130 140]);
        % Each subsystem documents its engine counterpart; e.g. 03 Quality
        % Gate calls assess_quality-equivalent routing via MATLAB Function
        % blocks added by hand after generation (see docs/SIMULINK_WORKFLOW.md).
        if ~isempty(prev)
            add_line(modelName, [prev '/1'], [subs{k} '/1']);
        end
        prev = subs{k};
        x = x + 200;
    end
    % SimEvents alternative (requires SimEvents license, NOT assumed):
    % replace Subsystem internals 01/02/05/07/08 with Entity Generator,
    % Entity Server, Entity Terminator, and Simulink Function callers.
    save_system(modelName);
    sys = modelName;
    fprintf('Built %s.slx (%d subsystems). Add MATLAB Function routing next.\n', ...
        modelName, numel(subs));
end
