% fig5_gp_surrogate.m
% GP posterior mean and std on the (peak_timing, peak_torque) slice, with
% N = 8 and N = 30 training points. Pure MATLAB — uses fitrgp.
%
% RUN: open in MATLAB 2026, Current Folder = repo root, press Run.

clear; clc; close all;

%% ============================ STYLE BLOCK ================================
OUT_DIR  = fullfile('figures', 'output');
OUT_NAME = 'fig5_gp_surrogate_matlab';
N_GRID   = 60;
N8       = 8;
N30      = 30;
SEED     = 7;
% ========================================================================

if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end
rng(SEED);

lo = [0.10 0.30 0.05 0.10];
hi = [0.40 0.60 0.20 1.00];
span = hi - lo;

% Unimodal surface optimum + reward fn
opt_u = [0.45 0.50 0.40 0.55];
opt = lo + opt_u .* span;
reward = @(p) gauss_bump((min(max(p, lo), hi) - lo) ./ span, opt_u, 0.18);

% 2D slice grid
pt = linspace(lo(2), hi(2), N_GRID);
pq = linspace(lo(4), hi(4), N_GRID);
[PT, PQ] = meshgrid(pt, pq);
R_true = zeros(N_GRID, N_GRID);

P_grid = zeros(N_GRID, N_GRID, 4);
for i = 1:N_GRID
    for j = 1:N_GRID
        p = opt;
        p(2) = PT(i, j);
        p(4) = PQ(i, j);
        P_grid(i, j, :) = p;
        R_true(i, j) = reward(p);
    end
end

% Sample training points on the slice
Pt8  = zeros(N8,  4); y8  = zeros(N8,  1);
Pt30 = zeros(N30, 4); y30 = zeros(N30, 1);
for k = 1:N8
    p = opt;
    p(2) = unifrnd(lo(2), hi(2));
    p(4) = unifrnd(lo(4), hi(4));
    Pt8(k, :) = p;  y8(k) = reward(p);
end
for k = 1:N30
    p = opt;
    p(2) = unifrnd(lo(2), hi(2));
    p(4) = unifrnd(lo(4), hi(4));
    Pt30(k, :) = p;  y30(k) = reward(p);
end

% Fit GPs (unit-normalized inputs)
gp8  = fitrgp((Pt8  - lo) ./ span,  y8,  'KernelFunction', 'ardsquaredexponential', 'Standardize', true);
gp30 = fitrgp((Pt30 - lo) ./ span,  y30, 'KernelFunction', 'ardsquaredexponential', 'Standardize', true);

% Predict on slice
Xq = reshape((P_grid - lo) ./ span, [], 4);
[mu8,  s8]  = predict(gp8,  Xq);
[mu30, s30] = predict(gp30, Xq);
mu8  = reshape(mu8,  N_GRID, N_GRID);
mu30 = reshape(mu30, N_GRID, N_GRID);
s30  = reshape(s30,  N_GRID, N_GRID);

% Plot
fig = figure('Color', 'w', 'Position', [100 100 600 1400]);

subplot(4,1,1); imagesc(pq, pt, R_true); axis xy; colormap(parula);
clim([0 1]); colorbar;
xlabel('Peak torque (N\cdotm/kg)'); ylabel('Peak timing');
title('True reward');

subplot(4,1,2); imagesc(pq, pt, mu8); axis xy; colormap(parula);
clim([0 1]); colorbar; hold on;
plot(Pt8(:,4), Pt8(:,2), 'wo', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k');
xlabel('Peak torque (N\cdotm/kg)'); ylabel('Peak timing');
title(sprintf('GP posterior mean (N = %d)', N8));

subplot(4,1,3); imagesc(pq, pt, mu30); axis xy; colormap(parula);
clim([0 1]); colorbar; hold on;
plot(Pt30(:,4), Pt30(:,2), 'wo', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k');
xlabel('Peak torque (N\cdotm/kg)'); ylabel('Peak timing');
title(sprintf('GP posterior mean (N = %d)', N30));

subplot(4,1,4); imagesc(pq, pt, s30); axis xy; colormap(magma);
colorbar;
xlabel('Peak torque (N\cdotm/kg)'); ylabel('Peak timing');
title(sprintf('GP posterior std (N = %d)', N30));

sgtitle('GP surrogate mean and uncertainty on (peak\_timing, peak\_torque)', ...
        'FontWeight', 'bold');

exportgraphics(fig, fullfile(OUT_DIR, [OUT_NAME '.pdf']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(OUT_DIR, [OUT_NAME '.svg']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(OUT_DIR, [OUT_NAME '.png']), 'Resolution', 300);
fprintf('Saved %s.{pdf,svg,png}\n', OUT_NAME);

%% ---- Helper -----------------------------------------------------------
function v = gauss_bump(u, center, sigma)
    d2 = sum((u - center).^2);
    v = exp(-d2 / (2 * sigma^2));
end
