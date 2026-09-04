function plot_retina_overview(rgb, enhanced, vesselResp, disc, fovea, lesions, outPath)
%PLOT_RETINA_OVERVIEW Publication/demo figure: original, enhanced, vessel
%   response, anatomy overlay, lesion overlay, notes. Candidate/evidence/
%   response wording only — never confirmed pathology.
%   disc: struct with center ([x y]) and radius (optional, may be []).
%   fovea: [x y] or []. lesions: cell array of binary masks (optional).
    fig = figure('Visible', 'off', 'Position', [100 100 1200 700]);
    subplot(2, 3, 1); imshow(rgb); title('1. Original fundus');
    subplot(2, 3, 2); imshow(enhanced); title('2. Enhanced (CLAHE/illum/denoise)');
    subplot(2, 3, 3); imagesc(vesselResp); axis image off; colormap hot; colorbar;
    title('3. Vessel response (top-hat)');
    subplot(2, 3, 4); imshow(rgb); hold on;
    if ~isempty(disc) && isfield(disc, 'center') && ~isempty(disc.center)
        rectangle('Position', [disc.center - disc.radius, 2 * disc.radius, 2 * disc.radius], ...
            'Curvature', [1 1], 'EdgeColor', 'g', 'LineWidth', 1.5);
        plot(disc.center(1), disc.center(2), 'g+', 'MarkerSize', 10);
    end
    if ~isempty(fovea)
        plot(fovea(1), fovea(2), 'mx', 'MarkerSize', 12, 'LineWidth', 2);
    end
    hold off; title('4. Anatomy: disc localization + fovea');
    subplot(2, 3, 5); imshow(rgb); hold on;
    if ~isempty(lesions)
        for k = 1:numel(lesions)
            B = bwboundaries(lesions{k} > 0, 'noholes');
            for b = 1:min(numel(B), 40)
                plot(B{b}(:, 2), B{b}(:, 1), 'y', 'LineWidth', 0.8);
            end
        end
    end
    hold off; title('5. Lesion evidence (candidates)');
    subplot(2, 3, 6); axis off;
    text(0.05, 0.9, {'6. Notes', '', ...
        'Overlays show candidate evidence,', 'not confirmed pathology.', '', ...
        'Model/explainability values are', 'imported, never recomputed here.'}, ...
        'FontSize', 10);
    if nargin >= 7 && ~isempty(outPath)
        saveas(fig, outPath);
    end
end
