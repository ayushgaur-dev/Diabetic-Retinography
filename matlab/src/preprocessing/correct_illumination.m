function out = correct_illumination(rgb, mask, sigma)
%CORRECT_ILLUMINATION Divide by slow background (Gaussian estimate) per
%   channel with clipped gains. Approximates Phase 3 median-blur gain
%   (median unavailable at this scale; Gaussian documented as substitute).
%   Gains clipped to [0.5, 2.0] like the Python config. Mask-protected bg.
    if nargin < 3, sigma = 30; end
    if nargin < 2 || isempty(mask), mask = true(size(rgb, 1), size(rgb, 2)); end
    mask = logical(mask);
    f = single(rgb);
    out = rgb;
    for c = 1:3
        ch = f(:, :, c);
        bg = max(imgaussfilt(ch, sigma), 1);
        m = mean(ch(mask));
        gain = min(max(m ./ bg, 0.5), 2.0);
        corr = min(max(ch .* gain, 0), 255);
        tmp = ch; tmp(mask) = corr(mask); out(:, :, c) = uint8(tmp);
    end
end
