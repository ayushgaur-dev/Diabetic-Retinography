function out = denoise_conservative(rgb, mask)
%DENOISE_CONSERVATIVE Edge-preserving bilateral filter (mild, mirrors
%   Phase 3 d=5/sigma 25/5). Mask-protected background. Small lesions must
%   survive: no aggressive smoothing here.
    if nargin < 2 || isempty(mask), mask = true(size(rgb, 1), size(rgb, 2)); end
    mask = logical(mask);
    sm = imbilatfilt(rgb, 25, 5);   % DegreeOfSmoothing, SpatialSigma
    out = rgb; out(repmat(mask, [1 1 3])) = sm(repmat(mask, [1 1 3]));
end
