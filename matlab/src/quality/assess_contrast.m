function [stdv, status] = assess_contrast(gray, mask, cfg)
%ASSESS_CONTRAST Std + p95-p5 guard inside mask (port of Phase 2).
    g = double(gray);
    inside = g(mask);
    stdv = std(inside(:));
    spread = prctile(inside(:), 95) - prctile(inside(:), 5);
    c = cfg.contrast;
    if stdv >= c.std_good && spread >= c.spread_borderline
        if stdv > c.std_excessive, status = 'BORDERLINE'; else, status = 'GOOD'; end
    elseif stdv >= c.std_borderline, status = 'BORDERLINE';
    else, status = 'BAD'; end
end
