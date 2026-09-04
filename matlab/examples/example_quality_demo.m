% example_quality_demo.m — synthetic quality walkthrough (Phase 11).
% No PHI: deterministic synthetic fundus-like image.
rng(26038);
img = uint8(zeros(224, 224, 3));
[yy, xx] = ndgrid(1:224, 1:224);
disc = (xx - 112).^2 + (yy - 112).^2 <= 100^2;
img(:, :, 1) = uint8(disc) * 110 + uint8(~disc) * 3;
img(:, :, 2) = uint8(disc) * 70;
img(:, :, 3) = uint8(disc) * 40;
img = img + uint8(randi([0 10], 224, 224, 3));
res = assess_quality(img);
fprintf('Quality: %s\n', res.status);
fprintf(' focus=%s illumination=%s contrast=%s exposure=%s fov=%s\n', ...
    res.focus, res.illumination, res.contrast, res.exposure, res.field_of_view);
mask = res.mask;
pre = denoise_conservative(enhance_contrast_clahe( ...
    correct_illumination(img, mask), mask), mask);
resp = vessel_response_demo(double(rgb2gray(pre)) / 255);
plot_retina_overview(img, pre, resp, [], [], {}, 'outputs/quality_demo.png');
fprintf('Figure: outputs/quality_demo.png\n');
