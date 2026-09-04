function result = assess_quality(rgb)
%ASSESS_QUALITY Full quality verdict (demonstration port of Phase 2).
%   Thresholds are READ from configs/quality_thresholds.json (shared with
%   Python — never forked here). Returns struct with overall status
%   GOOD/BORDERLINE/UNGRADABLE, component statuses, reasons.
%   This is a DEMONSTRATION implementation; the Python pipeline is production.
    cfg = load_json_config('quality_thresholds');
    mask = detect_fov(rgb, cfg.field_of_view.red_factor, cfg.field_of_view.red_floor);
    m = logical(mask);
    if ~any(m(:))
        result = struct('status', 'UNGRADABLE', 'focus', 'BAD', 'focus_value', 0, ...
            'illumination', 'BAD', 'illumination_median', 0, ...
            'contrast', 'BAD', 'contrast_std', 0, ...
            'exposure', 'BAD', 'clipped_fraction', 0, ...
            'field_of_view', 'BAD', 'mask_fraction', 0, ...
            'reasons', {{'field_of_view: no retinal region detected'}}, 'mask', mask);
        return;
    end
    gray = rgb2gray(rgb);
    [fval, fscore, fstat] = assess_focus(gray, m, cfg);
    [imed, inonuni, iscore, istat] = assess_illumination(gray, m, cfg);
    [cstd, cstat] = assess_contrast(gray, m, cfg);
    [clip, estat, edet] = assess_exposure(gray, m, cfg);
    frac = mean(m(:));
    if frac >= cfg.field_of_view.mask_fraction_good, vstat = 'GOOD';
    elseif frac >= cfg.field_of_view.mask_fraction_borderline, vstat = 'BORDERLINE';
    else, vstat = 'BAD'; end
    stats = {fstat, istat, cstat, estat, vstat};
    names = {'focus', 'illumination', 'contrast', 'exposure', 'field_of_view'};
    if any(strcmp(stats, 'BAD'))
        overall = 'UNGRADABLE';
    elseif any(strcmp(stats, 'BORDERLINE'))
        overall = 'BORDERLINE';
    else
        overall = 'GOOD';
    end
    reasons = {};
    for k = 1:numel(stats)
        if ~strcmp(stats{k}, 'GOOD')
            reasons{end + 1} = sprintf('%s: %s', names{k}, stats{k}); %#ok<AGROW>
        end
    end
    result = struct('status', overall, 'focus', fstat, 'focus_value', fval, ...
        'illumination', istat, 'illumination_median', imed, ...
        'contrast', cstat, 'contrast_std', cstd, ...
        'exposure', estat, 'clipped_fraction', clip, ...
        'field_of_view', vstat, 'mask_fraction', frac, ...
        'reasons', {reasons}, 'mask', mask);
end
