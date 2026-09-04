function out = normalize_rgb(rgb, mask)
%NORMALIZE_RGB Conservative per-channel gain toward common mean inside mask.
%   Mirrors Phase 3 gray-world (max_gain 1.25). Background outside mask kept.
    f = single(rgb);
    if nargin < 2 || isempty(mask), mask = true(size(f, 1), size(f, 2)); end
    mask = logical(mask);
    mu = zeros(1, 3);
    for c = 1:3
        ch = f(:, :, c);
        mu(c) = mean(ch(mask));
    end
    target = mean(mu);
    gains = min(max(target ./ max(mu, 1), 1 / 1.25), 1.25);
    out = rgb;
    for c = 1:3
        ch = min(max(f(:, :, c) * gains(c), 0), 255);
        tmp = f(:, :, c); tmp(mask) = ch(mask); out(:, :, c) = uint8(tmp);
    end
end
