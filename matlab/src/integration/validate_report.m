function [ok, missing] = validate_report(rep)
%VALIDATE_REPORT Check required Phase 9 fields exist (no fabrication).
%   Optional evidence (fovea/vessels/lesions/figure) degrades to warnings.
    required = {'report_version', 'status', 'sections'};
    missing = {};
    for k = 1:numel(required)
        if ~isfield(rep, required{k}), missing{end+1} = required{k}; end %#ok<AGROW>
    end
    ok = isempty(missing);
    if ~ok, return; end
    s = rep.sections;
    for k = {'summary', 'triage', 'limitations'}
        if ~isfield(s, k{1})
            missing{end+1} = sprintf('sections.%s', k{1}); %#ok<AGROW>
        end
    end
    ok = isempty(missing);
end
