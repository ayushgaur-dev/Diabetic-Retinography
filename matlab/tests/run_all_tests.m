function run_all_tests()
%RUN_ALL_TESTS Execute the full matlab.unittest suite (Phase 11).
%   From MATLAB, with cwd at matlab/:  >> run_all_tests
    here = fileparts(mfilename('fullpath'));
    suite = testsuite(fullfile(here, 'tests'));
    runner = testrunner('textoutput');
    results = runner.run(suite);
    assert(all([results.Passed]), 'MATLAB test failures present.');
    fprintf('MATLAB suite: %d passed, %d failed.\n', ...
        sum([results.Passed]), sum([results.Failed]));
end
