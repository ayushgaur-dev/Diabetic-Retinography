function S = run_engineering_workflow(imagePath, reportJson)
%RUN_ENGINEERING_WORKFLOW INPUT->QUALITY->PREPROCESSING->SCREENING RESULT
%   ->ANATOMY/EVIDENCE->EXPLAINABILITY->TRIAGE->REPORT using MATLAB image
%   processing plus an imported authoritative Phase 9 JSON.
%   S = run_engineering_workflow('fundus.png', 'report.json')
    S = screening_struct();
    S.image_path = imagePath;
    S.image = load_fundus(imagePath);
    S.quality = assess_quality(S.image);
    mask = S.quality.mask;
    pre = correct_illumination(S.image, mask);
    pre = enhance_contrast_clahe(pre, mask);
    pre = denoise_conservative(pre, mask);
    S.preprocessed = normalize_rgb(pre, mask);
    rep = load_screening_json(reportJson);
    [ok, missing] = validate_report(rep);
    assert(ok, 'Invalid screening report (missing: %s).', strjoin(missing, ', '));
    S.screening = rep.sections;
    S.anatomy = rep.sections.anatomy;
    S.evidence = rep.sections.lesions;
    S.explainability = rep.sections.explainability;
    S.triage = rep.sections.triage;
    S.report_path = reportJson;
    if strcmp(S.quality.status, 'UNGRADABLE')
        S.warnings{end+1} = 'MATLAB quality gate: UNGRADABLE — imported grade shown for reference only, not trusted.';
    end
    screening_summary(rep);
end
