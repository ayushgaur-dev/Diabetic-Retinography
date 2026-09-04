function screening_summary(rep)
%SCREENING_SUMMARY Console engineering summary from an imported Phase 9
%   report struct. All values are READ from rep (imported Python result).
    s = rep.sections;
    sm = s.summary; tr = s.triage;
    fprintf('--------------------------------------------------\n');
    fprintf('EXPLAINABLE DR SCREENING (imported Python result)\n');
    fprintf('--------------------------------------------------\n\n');
    fprintf('IMAGE QUALITY\n%s\n\n', string(sm.image_quality));
    fprintf('DR GRADE\n%s\n\n', string(sm.model_prediction));
    ref = s.referable;
    fprintf('REFERABLE\n%s (score %.4f vs %.2f)\n\n', ...
        string(ref.classification), ref.score, ref.threshold);
    fprintf('ANATOMY\n');
    for k = 1:numel(s.anatomy.structures)
        st = s.anatomy.structures(k);
        fprintf('%-10s %s\n', st.structure, string(st.status));
    end
    fprintf('\nLESION EVIDENCE\n');
    for k = 1:numel(s.lesions.lesions)
        l = s.lesions.lesions(k);
        fprintf('%-14s %s (%d candidates)\n', l.lesion_type, ...
            string(l.status), l.candidate_count);
    end
    fprintf('\nEXPLAINABILITY\n%s\n\n', string(s.explainability.consistency));
    fprintf('TRIAGE\n%s [%s]\n', string(tr.decision), string(tr.priority));
    fprintf('--------------------------------------------------\n');
end
