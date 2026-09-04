function out = enhance_contrast_clahe(rgb, mask, clipLimit)
%ENHANCE_CONTRAST_CLAHE CLAHE on the L channel of Lab (never independent
%   RGB), mirroring Phase 3 (clip 2.0, 8x8 tiles). Mask-protected bg.
    if nargin < 3, clipLimit = 2.0; end
    if nargin < 2 || isempty(mask), mask = true(size(rgb, 1), size(rgb, 2)); end
    mask = logical(mask);
    lab = rgb2lab(rgb);
    L = lab(:, :, 1) / 100;                 % Lab L in [0,100] -> [0,1]
    Leq = adapthisteq(L, 'ClipLimit', clipLimit, 'NumTiles', [8 8]);
    lab(:, :, 1) = Leq * 100;
    enh = lab2rgb(lab, 'OutputType', 'uint8');
    out = rgb; out(repmat(mask, [1 1 3])) = enh(repmat(mask, [1 1 3]));
end
