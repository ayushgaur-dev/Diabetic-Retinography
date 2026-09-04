function plot_results(rep, outDir)
%PLOT_RESULTS Per-scenario figures (all labelled SIMULATION).
    pools = {'capture', 'ai', 'review', 'tele'};
    names = {'Capture', 'AI', 'Review', 'Tele'};
    fig = figure('Visible', 'off');
    bar([rep.utilization.capture, rep.utilization.ai, ...
        rep.utilization.review, rep.utilization.tele]);
    set(gca, 'XTickLabel', names);
    ylabel('Utilization'); title('Server utilization (SIMULATION)');
    saveas(fig, fullfile(outDir, sprintf('util_%s.png', rep.scenario_name)));
    fig = figure('Visible', 'off');
    bar([rep.queue_max.capture, rep.queue_max.ai, ...
        rep.queue_max.review, rep.queue_max.tele]);
    set(gca, 'XTickLabel', names);
    ylabel('Max queue length'); title('Peak queues (SIMULATION)');
    saveas(fig, fullfile(outDir, sprintf('queues_%s.png', rep.scenario_name)));
    fig = figure('Visible', 'off');
    bar([rep.mean_queue_wait_s.capture, rep.mean_queue_wait_s.ai, ...
        rep.mean_queue_wait_s.review, rep.mean_queue_wait_s.tele]);
    set(gca, 'XTickLabel', names);
    ylabel('Mean wait (s)'); title('Queue waiting time (SIMULATION)');
    saveas(fig, fullfile(outDir, sprintf('waits_%s.png', rep.scenario_name)));
    fig = figure('Visible', 'off');
    hold on;
    for k = {'ai', 'review', 'tele', 'capture'}
        s = rep.queue_series.(k{1});
        if ~isempty(s.t)
            plot(s.t / 3600, s.len, 'DisplayName', k{1});
        end
    end
    hold off;
    xlabel('Time (h)'); ylabel('Queue length');
    title('Queue length vs time (SIMULATION)');
    legend('show', 'Location', 'northwest');
    saveas(fig, fullfile(outDir, sprintf('queues_time_%s.png', rep.scenario_name)));
end
