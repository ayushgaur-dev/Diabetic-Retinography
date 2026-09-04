function plot_scalability(reps, outDir)
%PLOT_SCALABILITY Cross-scenario (A-D) comparison figures. SIMULATION labels.
    names = {'A', 'B', 'C', 'D'};
    vols = [10000 50000 100000 150000];
    thr = zeros(1, 4); lat = zeros(1, 4); rev = zeros(1, 4);
    for k = 1:4
        r = reps.(names{k});
        thr(k) = r.throughput_per_year;
        lat(k) = r.mean_latency_s;
        rev(k) = r.utilization.review;
    end
    fig = figure('Visible', 'off');
    plot(vols, thr, 'o-'); hold on; plot(vols, vols, '--'); hold off;
    xlabel('Annual arrivals (ENGINEERING ASSUMPTION)');
    ylabel('Simulated completions/year');
    title('Throughput vs volume (SIMULATION)');
    legend('simulated', 'identity', 'Location', 'northwest');
    saveas(fig, fullfile(outDir, 'scalability_throughput.png'));
    fig = figure('Visible', 'off');
    yyaxis left; plot(vols, lat / 60, 'o-'); ylabel('Mean latency (min)');
    yyaxis right; plot(vols, rev, 's-'); ylabel('Reviewer utilization');
    xlabel('Annual arrivals (ENGINEERING ASSUMPTION)');
    title('Latency + reviewer load (SIMULATION)');
    saveas(fig, fullfile(outDir, 'scalability_latency.png'));
end
