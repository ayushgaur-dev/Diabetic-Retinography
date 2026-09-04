% example_screening_summary.m — Phase 9 JSON -> console summary (Phase 11).
% Uses the committed non-PHI example report (no dataset needed).
here = fileparts(mfilename('fullpath'));
p = fullfile(here, '..', '..', 'reports', 'reporting', 'examples', ...
    'example_refer', 'report.json');
assert(isfile(p), 'example report.json missing');
rep = load_screening_json(p);
[ok, missing] = validate_report(rep);
assert(ok, strjoin(missing, ', '));
screening_summary(rep);
