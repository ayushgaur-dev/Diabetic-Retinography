classdef TestLoading < matlab.unittest.TestCase
%TESTLOADING Image loading + FOV contract tests.
    methods (Test)
        function syntheticLoads(testCase)
            rng(7);
            img = uint8(zeros(200, 240, 3));
            [yy, xx] = ndgrid(1:200, 1:240);
            img(repmat((xx - 120).^2 + (yy - 100).^2 <= 90^2, [1 1 3])) = 120;
            tmp = [tempname '.png'];
            imwrite(img, tmp);
            rgb = load_fundus(tmp);
            testCase.verifySize(rgb, [200 240 3]);
            testCase.verifyClass(rgb, 'uint8');
            delete(tmp);
        end
        function missingFileErrors(testCase)
            testCase.verifyError(@() load_fundus('no_such_file_xyz.png'), 'MATLAB:assertion:failed');
        end
        function resizeKeepsAspect(testCase)
            img = uint8(zeros(100, 200, 3));
            tmp = [tempname '.png'];
            imwrite(img, tmp);
            rgb = load_fundus(tmp, 100);
            testCase.verifySize(rgb, [50 100 3]);
            delete(tmp);
        end
        function fovIsPlausible(testCase)
            rng(11);
            img = uint8(zeros(224, 224, 3));
            [yy, xx] = ndgrid(1:224, 1:224);
            disc = (xx - 112).^2 + (yy - 112).^2 <= 100^2;
            img(repmat(disc, [1 1 3])) = 0;
            img(:, :, 1) = uint8(disc) * 110 + uint8(~disc) * 3;
            mask = detect_fov(img);
            frac = mean(mask(:) > 0);
            testCase.verifyGreaterThan(frac, 0.4);
            testCase.verifyLessThan(frac, 0.9);
        end
    end
end
