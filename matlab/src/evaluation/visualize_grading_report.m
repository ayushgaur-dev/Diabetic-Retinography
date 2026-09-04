function visualize_grading_report(outDir)
%VISUALIZE_GRADING_REPORT Figures from reports/grading artifacts (import-only).
%   Reads metrics.json + confusion_matrix.csv. Labels VALIDATION vs TEST from
%   the artifact itself; never recomputes metrics with new definitions.
    here = fileparts(mfilename('fullpath'));
    rep = fullfile(here, '..', '..', '..', 'reports', 'grading');
    m = jsondecode(fileread(fullfile(rep, 'metrics.json')));
    cm = readmatrix(fullfile(rep, 'confusion_matrix.csv'), 'Range', 'B2:F6');
    fig = figure('Visible', 'off');
    heatmap({'G0', 'G1', 'G2', 'G3', 'G4'}, {'G0', 'G1', 'G2', 'G3', 'G4'}, cm, ...
        'Title', 'Confusion matrix (held-out test)', ...
        'XLabel', 'Predicted', 'YLabel', 'True');
    saveas(fig, fullfile(outDir, 'matlab_confusion_matrix.png'));
    fig2 = figure('Visible', 'off');
    grades = {'G0', 'G1', 'G2', 'G3', 'G4'};
    rec = [m.five_class.per_class.G0.recall, m.five_class.per_class.G1.recall, ...
        m.five_class.per_class.G2.recall, m.five_class.per_class.G3.recall, ...
        m.five_class.per_class.G4.recall];
    bar(categorical(grades), rec);
    ylabel('Recall'); title('Per-class recall (held-out test)');
    saveas(fig2, fullfile(outDir, 'matlab_per_class_recall.png'));
    fprintf('QWK=%.4f sens=%.4f spec=%.4f AUROC=%.4f (imported values)\n', ...
        m.five_class.qwk, m.referable.sensitivity, ...
        m.referable.specificity, m.roc_pr.auroc);
end
