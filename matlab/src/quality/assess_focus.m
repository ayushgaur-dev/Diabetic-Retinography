function [val, score, status] = assess_focus(gray, mask, cfg)
%ASSESS_FOCUS Laplacian variance inside mask (demonstration port of Phase 2).
%   Thresholds come from the SHARED configs/quality_thresholds.json.
%   Parity notes: fspecial alpha 0 matches the cv2 4-neighbour kernel;
%   MATLAB var() normalizes by N-1 vs numpy N (relative diff ~1/N, documented).
    lap = imfilter(double(gray), fspecial('laplacian', 0), 'replicate');
    v = lap(mask);
    val = var(v(:));
    lo = cfg.focus.log_lo; hi = cfg.focus.log_hi;
    score = min(max((log10(val + 1) - lo) / (hi - lo), 0), 1);
    if val >= cfg.focus.good_var, status = 'GOOD';
    elseif val >= cfg.focus.borderline_var, status = 'BORDERLINE';
    else, status = 'BAD'; end
end
