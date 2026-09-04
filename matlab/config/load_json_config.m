function cfg = load_json_config(name)
%LOAD_JSON_CONFIG Read a repo JSON config by filename (no extension needed).
%   Thresholds are shared with Python: the SAME files under configs/.
%   Example: cfg = load_json_config('quality_thresholds')
    here = fileparts(mfilename('fullpath'));   % matlab/config
    p = fullfile(here, '..', '..', 'configs', [name '.json']);
    if ~isfile(p)
        % Fallback: search upward for a configs/ directory.
        d = here;
        for k = 1:6
            cand = fullfile(d, 'configs', [name '.json']);
            if isfile(cand), p = cand; break; end
            d = fileparts(d);
        end
    end
    assert(isfile(p), 'Config not found: %s', name);
    cfg = jsondecode(fileread(p));
end
