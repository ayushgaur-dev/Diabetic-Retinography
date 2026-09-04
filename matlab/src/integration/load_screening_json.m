function rep = load_screening_json(path)
%LOAD_SCREENING_JSON Parse a Phase 9 report JSON (authoritative import).
%   No values are recomputed; the Python result stays authoritative.
    assert(isfile(path), 'Report JSON not found: %s', path);
    rep = jsondecode(fileread(path));
end
