classdef TestQuality < matlab.unittest.TestCase
%TESTQUALITY Quality states + shared-threshold provenance.
    methods (Test)
        function statesAreValid(testCase)
            rng(9);
            img = uint8(zeros(224, 224, 3));
            [yy, xx] = ndgrid(1:224, 1:224);
            disc = (xx - 112).^2 + (yy - 112).^2 <= 100^2;
            img = repmat(uint8(disc) * 110, [1 1 3]);
            img = img + uint8(randi([0 12], 224, 224, 3));
            res = assess_quality(img);
            testCase.verifyMember(res.status, {'GOOD', 'BORDERLINE', 'UNGRADABLE'});
        end
        function blankIsUngradable(testCase)
            res = assess_quality(uint8(zeros(224, 224, 3)));
            testCase.verifyEqual(res.status, 'UNGRADABLE');
            testCase.verifyNotEmpty(res.reasons);
        end
        function thresholdsAreShared(testCase)
            cfg = load_json_config('quality_thresholds');
            testCase.verifyEqual(cfg.focus.good_var, 300.0, 'AbsTol', 1e-9);
            testCase.verifyEqual(cfg.aggregation.weights.field_of_view, 1.5);
        end
        function ungradableRouting(testCase)
            % Gate contract: UNGRADABLE must carry recapture-style reasons.
            res = assess_quality(uint8(zeros(224, 224, 3)));
            testCase.verifyTrue(any(contains(res.reasons, 'field_of_view')) ...
                || any(contains(res.reasons, 'focus')) ...
                || any(contains(res.reasons, 'contrast')));
        end
    end
end
