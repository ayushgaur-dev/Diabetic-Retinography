classdef TestPreprocessing < matlab.unittest.TestCase
%TESTPREPROCESSING Preprocessing contract: dtype/shape preserved, finite.
    methods (Test)
        function opsPreserveGeometry(testCase)
            rng(5);
            rgb = uint8(randi([0 255], 120, 160, 3));
            mask = uint8(ones(120, 160)) * 255;
            for f = {@(x) normalize_rgb(x, mask), @(x) correct_illumination(x, mask), ...
                    @(x) enhance_contrast_clahe(x, mask), @(x) denoise_conservative(x, mask)}
                out = f{1}(rgb);
                testCase.verifySize(out, size(rgb));
                testCase.verifyClass(out, 'uint8');
                testCase.verifyTrue(all(isfinite(single(out(:)))));
            end
        end
        function backgroundPreserved(testCase)
            rgb = uint8(100 * ones(60, 60, 3));
            mask = zeros(60, 60, 'uint8');
            mask(20:40, 20:40) = 255;
            out = enhance_contrast_clahe(rgb, mask);
            testCase.verifyEqual(out(1, 1, :), rgb(1, 1, :));
        end
        function vesselResponseBounded(testCase)
            g = uint8(128 * ones(80, 90));
            g(30:50, 30:50) = 40;
            r = vessel_response_demo(g);
            testCase.verifyGreaterThanOrEqual(min(r(:)), 0);
            testCase.verifyLessThanOrEqual(max(r(:)), 1);
            testCase.verifyTrue(any(r(:) > 0));
        end
    end
end
