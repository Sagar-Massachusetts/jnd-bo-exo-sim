% Md Mainuddin Sagar
% fig3_reward_surfaces.m
% 2x2 panel of the four synthetic reward surfaces, sliced over
% (peak_timing, peak_torque) with the other two params at their optimum.
% Pure MATLAB 
%
% Run Process: openning in MATLAB 2026 in my macbook, Current Folder = repo root, press Run.

clear; clc; close all;

%% STYLE BLOCK
OUT_DIR  = fullfile('figures', 'output');
OUT_NAME = 'fig3_reward_surfaces_matlab';
FIG_TITLE = 'Four synthetic reward surfaces (peak\_timing vs peak\_torque)';
N_GRID = 120;
%

if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

% Bounds
lo = [0.10 0.30 0.05 0.10];
hi = [0.40 0.60 0.20 1.00];
span = hi - lo;
to_unit = @(p) (min(max(p, lo), hi) - lo) ./ span;
from_unit = @(u) lo + min(max(u, 0), 1) .* span;

% Surface centres in unit space (match src/reward_surfaces.py)
centers = struct( ...
    'unimodal',         [0.45 0.50 0.40 0.55], ...
    'multimodal',       {{[0.20 0.25 0.20 0.70], [0.80 0.75 0.80 0.30]}}, ...
    'asymmetric',       [0.45 0.40 0.40 0.55], ...
    'timing_sensitive', [0.45 0.50 0.40 0.55]);

names = {'unimodal','multimodal','asymmetric','timing_sensitive'};

% Sweep grid
pt = linspace(lo(2), hi(2), N_GRID);
pq = linspace(lo(4), hi(4), N_GRID);
[PT, PQ] = meshgrid(pt, pq);

fig = figure('Color', 'w', 'Position', [100 100 1100 950]);

for k = 1:numel(names)
    name = names{k};
    c = centers.(name);
    switch name
        case 'unimodal'
            opt = from_unit(c);
        case 'multimodal'
            opt = from_unit(c{1});
        otherwise
            opt = from_unit(c);
    end

    R = zeros(N_GRID, N_GRID);
    for i = 1:N_GRID
        for j = 1:N_GRID
            p = opt;
            p(2) = PT(i, j);
            p(4) = PQ(i, j);
            R(i, j) = eval_surface(name, p, c, lo, hi, to_unit);
        end
    end

    subplot(2, 2, k);
    imagesc(pq, pt, R); axis xy; colormap(gca, 'parula');
    clim([0 1]); colorbar;
    hold on;
    plot(opt(4), opt(2), 'p', 'MarkerSize', 20, 'MarkerFaceColor', 'w', ...
         'MarkerEdgeColor', 'k', 'LineWidth', 1.2);
    xlabel('Peak torque (N\cdotm/kg)');
    ylabel('Peak timing (fraction)');
    title(strrep(name, '_', '\_'));
end

sgtitle(FIG_TITLE, 'FontWeight', 'bold');

exportgraphics(fig, fullfile(OUT_DIR, [OUT_NAME '.pdf']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(OUT_DIR, [OUT_NAME '.svg']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(OUT_DIR, [OUT_NAME '.png']), 'Resolution', 300);
fprintf('Saved %s.{pdf,svg,png}\n', OUT_NAME);

%% Local helpers 
function val = eval_surface(name, p, c, lo, hi, to_unit)
    u = to_unit(p);
    span = hi - lo;
    switch name
        case 'unimodal'
            val = gauss_bump(u, c, 0.18);
        case 'multimodal'
            val = 0.5 * gauss_bump(u, c{1}, 0.15) + ...
                  0.5 * gauss_bump(u, c{2}, 0.15);
        case 'asymmetric'
            d = u - c;
            sig = [0.08 0.08 0.18 0.18];
            sig(d <= 0) = 0.25;    % slower fall on negative side for rise, peak
            sig(3) = 0.18; sig(4) = 0.18;
            val = exp(-0.5 * sum((d ./ sig).^2));
        case 'timing_sensitive'
            d = u - c;
            sig = [0.10 0.10 0.10 0.35];
            val = exp(-0.5 * sum((d ./ sig).^2));
    end
    val = min(max(val, 0), 1);
end

function val = gauss_bump(u, center, sigma)
    d2 = sum((u - center).^2);
    val = exp(-d2 / (2 * sigma^2));
end
