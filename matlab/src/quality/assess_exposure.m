function [clipped, status, details] = assess_exposure(gray, mask, cfg)
%ASSESS_EXPOSURE Near-black/white fractions inside mask (port of Phase 2).
    g = double(gray);
    inside = g(mask);
    c = cfg.exposure;
    black = mean(inside(:) <= c.black_level);
    white = mean(inside(:) >= c.white_level);
    clipped = black + white;
    if black >= c.black_bad || white >= c.white_bad, status = 'BAD';
    elseif black >= c.black_borderline || white >= c.white_borderline, status = 'BORDERLINE';
    else, status = 'GOOD'; end
    details = struct('black_fraction', black, 'white_fraction', white);
end
