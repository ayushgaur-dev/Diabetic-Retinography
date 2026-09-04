function [med, nonuni, score, status] = assess_illumination(gray, mask, cfg)
%ASSESS_ILLUMINATION Median + 4x4 block-mean non-uniformity (port of Phase 2).
    g = double(gray);
    inside = g(mask);
    med = median(inside(:));
    c = cfg.illumination;
    [h, w] = size(g);
    bm = zeros(4, 4);
    for i = 1:4
        for j = 1:4
            cell = g(floor((i-1)*h/4)+1:floor(i*h/4), floor((j-1)*w/4)+1:floor(j*w/4));
            cm = mask(floor((i-1)*h/4)+1:floor(i*h/4), floor((j-1)*w/4)+1:floor(j*w/4));
            vals = cell(cm);
            bm(i, j) = mean(vals(:));
        end
    end
    nonuni = std(bm(:));
    if med >= c.median_good_lo && med <= c.median_good_hi, ms = 1;
    elseif med >= c.median_borderline_lo && med <= c.median_borderline_hi, ms = 0.5;
    else, ms = 0; end
    if nonuni < c.nonuniformity_good, us = 1;
    elseif nonuni < c.nonuniformity_borderline, us = 0.5;
    else, us = 0; end
    score = 0.7 * ms + 0.3 * us;
    mstat = 'GOOD';
    if med < c.median_borderline_lo || med > c.median_borderline_hi, mstat = 'BAD';
    elseif med < c.median_good_lo || med > c.median_good_hi, mstat = 'BORDERLINE'; end
    ustat = 'GOOD';
    if nonuni >= c.nonuniformity_borderline, ustat = 'BAD';
    elseif nonuni >= c.nonuniformity_good, ustat = 'BORDERLINE'; end
    rank = @(s) strcmp(s, 'GOOD') * 0 + strcmp(s, 'BORDERLINE') * 1 + strcmp(s, 'BAD') * 2;
    status = mstat;
    if rank(ustat) > rank(mstat), status = ustat; end
end
