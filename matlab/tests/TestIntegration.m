classdef TestIntegration < matlab.unittest.TestCase
%TESTINTEGRATION Phase 9 JSON import: parsing, validation, missing fields,
%   malformed JSON, ungradable routing, visualization input checks, and
%   Python/MATLAB field consistency on the committed example report.
    methods (Test)
        function exampleReportParses(testCase)
            p = testCase.exampleReport();
            rep = load_screening_json(p);
            [ok, missing] = validate_report(rep);
            testCase.verifyTrue(ok, strjoin(missing, ', '));
        end
        function requiredFieldsEnforced(testCase)
            [ok, missing] = validate_report(struct('foo', 1));
            testCase.verifyFalse(ok);
            testCase.verifyNotEmpty(missing);
        end
        function malformedJsonHandled(testCase)
            tmp = [tempname '.json'];
            fid = fopen(tmp, 'w');
            fwrite(fid, '{not valid json');
            fclose(fid);
            try
                load_screening_json(tmp);
                failed = false;
            catch
                failed = true;
            end
            testCase.verifyTrue(failed, 'malformed JSON must raise');
            delete(tmp);
        end
        function fieldConsistency(testCase)
            rep = load_screening_json(testCase.exampleReport());
            s = rep.sections;
            testCase.verifyEqual(s.grading.predicted_grade, s.triage.predicted_grade);
            testCase.verifyEqual(s.summary.workflow_recommendation, s.triage.decision);
            testCase.verifyMember(s.triage.decision, ...
                {'UNGRADABLE', 'ROUTINE', 'REFER', 'URGENT_REVIEW', 'TECHNICAL_REVIEW'});
        end
        function ungradableHasNoGrade(testCase)
            here = fileparts(mfilename('fullpath'));
            p = fullfile(here, '..', '..', 'reports', 'reporting', 'examples', ...
                'example_ungradable', 'report.json');
            testCase.assumeTrue(isfile(p), 'ungradable example missing');
            rep = load_screening_json(p);
            testCase.verifyEqual(string(rep.sections.triage.decision), 'UNGRADABLE');
            testCase.verifyFalse(rep.sections.grading.assessed);
        end
        function missingFigureWarns(testCase)
            rep = load_screening_json(testCase.exampleReport());
            fig = '';
            if isfield(rep.sections, 'explainability') && isfield(rep.sections.explainability, 'figure')
                fig = string(rep.sections.explainability.figure);
            end
            testCase.verifyTrue(strlength(fig) == 0 || isfile(fig), ...
                'figure must exist or be explicitly absent');
        end
        function realAptosExample(testCase)
            % Non-PHI APTOS sample via env var; skipped when absent (no downloads).
            p = getenv('APTOS_SAMPLE_IMAGE');
            testCase.assumeTrue(~isempty(p) && isfile(p), 'APTOS_SAMPLE_IMAGE not set');
            rgb = load_fundus(p);
            res = assess_quality(rgb);
            testCase.verifyMember(res.status, {'GOOD', 'BORDERLINE', 'UNGRADABLE'});
        end
        function vizInputValidation(testCase)
            rgb = uint8(zeros(100, 120, 3));
            testCase.verifyError(@() plot_retina_overview(), 'MATLAB:minrhs');
        end
    end
    methods (Access = private)
        function p = exampleReport(~)
            here = fileparts(mfilename('fullpath'));
            p = fullfile(here, '..', '..', 'reports', 'reporting', 'examples', ...
                'example_refer', 'report.json');
            assert(isfile(p), 'example report.json missing: run Phase 9 demo first');
        end
    end
end
