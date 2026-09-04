function visualize_calibration_report(outDir)
%VISUALIZE_CALIBRATION_REPORT Reliability diagrams from
%   reports/calibration/metrics.json (import-only; same bins raw vs cal).
    here = fileparts(mfilename('fullpath'));
    rep = fullfile(here, '..', '..', '..', 'reports', 'calibration');
    m = jsondecode(fileread(fullfile(rep, 'metrics.json')));
    for k = {'raw', 'calibrated'}
        bins = m.ece_bins_detail.(k{1});
        xs = cellfun(@(b) mean(b.mean_confidence), num2cell(bins));
        ys = cellfun(@(b) mean(b.accuracy), num2cell(bins));
        ns = cellfun(@(b) b.count, num2cell(bins));
        fig = figure('Visible', 'off');
        subplot(2, 1, 1);
        plot([0 1], [0 1], '--', 'Color', [0.5 0.5 0.5]); hold on;
        plot(xs, ys, 'o-'); hold off;
        xlabel('Mean confidence'); ylabel('Empirical accuracy');
        title(sprintf('Reliability (%s, test)', k{1}));
        xlim([0 1]); ylim([0 1]);
        subplot(2, 1, 2);
        bar(xs, ns);
        xlabel('Confidence bin'); ylabel('Count');
        saveas(fig, fullfile(outDir, sprintf('matlab_reliability_%s.png', k{1})));
    end
    fprintf('T=%.4f ECE %.4f->%.4f (imported values)\n', ...
        m.temperature, m.ece.('10').raw, m.ece.('10').calibrated);
end
