% Md Mainuddin Sagar
% fig4_human_oracle.m
% Left: Bradley-Terry preference probability.
% Right: JND-aware tie zone on a (peak_timing, peak_torque) slice.
% Pure MATLAB
%
% Running process: openning in MATLAB 2026 in my macbook, Current Folder = repo root, press Run.

clear; clc; close all;

%% STYLE BLOCK
OUT_DIR  = fullfile('figures', 'output');
OUT_NAME = 'fig4_human_oracle_matlab';
BETAS    = [1.0 5.0 20.0];
N_GRID   = 120;
%

if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

lo = [0.10 0.30 0.05 0.10];
hi = [0.40 0.60 0.20 1.00];
span = hi - lo;
PARAM_JND = [0.030 0.030 0.030 0.100];

% Optimum of unimodal surface
opt = lo + [0.45 0.50 0.40 0.55] .* span;

fig = figure('Color', 'w', 'Position', [100 100 1300 500]);

%% Left: Bradley-Terry
subplot(1, 2, 1); hold on; box on;
deltas = linspace(-1, 1, 401);
for k = 1:numel(BETAS)
    beta = BETAS(k);
    p = 1 ./ (1 + exp(-beta * deltas));
    plot(deltas, p, 'LineWidth', 1.8, 'DisplayName', sprintf('\\beta = %g', beta));
end
xlabel('Reward difference  f(A) - f(B)');
ylabel('P(A preferred over B)');
title('Bradley-Terry preference probability');
grid on;
legend('Location', 'southeast');

%% Right: JND tie zone
subplot(1, 2, 2); hold on; box on;
pt = linspace(lo(2), hi(2), N_GRID);
pq = linspace(lo(4), hi(4), N_GRID);
[PT, PQ] = meshgrid(pt, pq);

tie = zeros(N_GRID, N_GRID);
for i = 1:N_GRID
    for j = 1:N_GRID
        p = opt;
        p(2) = PT(i, j);
        p(4) = PQ(i, j);
        dist_norm = abs(p - opt) ./ span;
        if all(dist_norm < PARAM_JND)
            tie(i, j) = 1;
        end
    end
end

imagesc(pq, pt, tie); axis xy; colormap(gca, flipud(gray));
clim([0 1]);
plot(opt(4), opt(2), 'p', 'MarkerSize', 22, 'MarkerFaceColor', 'k', ...
     'MarkerEdgeColor', 'w', 'LineWidth', 1.5, 'DisplayName', 'optimum');
xlabel('Peak torque (N\cdotm/kg)');
ylabel('Peak timing (fraction)');
title('JND-aware tie zone (dark = perceptually indistinguishable)');
legend('Location', 'northeast');

sgtitle('Simulated human oracle: preference + JND-aware comparison', ...
        'FontWeight', 'bold');

exportgraphics(fig, fullfile(OUT_DIR, [OUT_NAME '.pdf']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(OUT_DIR, [OUT_NAME '.svg']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(OUT_DIR, [OUT_NAME '.png']), 'Resolution', 300);
fprintf('Saved %s.{pdf,svg,png}\n', OUT_NAME);
