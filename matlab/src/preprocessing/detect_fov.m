function mask = detect_fov(rgb, redFactor, redFloor)
%DETECT_FOV Retinal field mask (uint8 0/255). Mirrors Phase 2 logic:
%   red-channel threshold relative to mean red + morphology + largest
%   component. Demonstration-grade; NOT a clinical segmentation.
%   Defaults match configs/quality_thresholds.json (red_factor 0.35).
    if nargin < 2, redFactor = 0.35; end
    if nargin < 3, redFloor = 10.0; end
    red = single(rgb(:, :, 1));
    t = max(redFloor, mean(red(:)) * redFactor);
    m = red > t;
    m = imclose(m, strel('disk', 4));
    m = imopen(m, strel('disk', 3));
    cc = bwconncomp(m);
    mask = zeros(size(m), 'uint8');
    if cc.NumObjects == 0, return; end
    areas = cellfun(@numel, cc.PixelIdxList);
    [~, big] = max(areas);
    mask(cc.PixelIdxList{big}) = 255;
end
