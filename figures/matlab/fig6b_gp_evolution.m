% fig6b_gp_evolution.m
% Two panels: (left) GP mean at the true optimum over iterations,
%             (right) GP std at the true optimum over iterations.
% Reads results/logs/gp_evolution_unimodal.json (exported from Python).
%
% RUN: open in MATLAB 2026, Current Folder = repo root, press Run.

clear; clc; close all;

%% ============================ STYLE BLOCK ================================
OUT_DIR  = fullfile('figures', 'output');
OUT_NAME = 'fig6b_gp_evolution_matlab';
JSON     = fullfile('results', 'logs', 'gp_evolution_unimodal.json');
COLORS   = [0.12 0.47 0.71; 0.84 0.15 0.16];
LABELS   = {'EI', 'JND-EI'};
METHODS  = {'ei', 'jnd_ei'};
% ========================================================================

if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end
if ~exist(JSON, 'file')
    error('Missing %s. Run the Python export command first.', JSON);
end
D = jsondecode(fileread(JSON));
budget = D.budget;
t = 1:budget;

fig = figure('Color', 'w', 'Position', [100 100 1200 450]);

% Left: mean
subplot(1,2,1); hold on; box on;
for k = 1:numel(METHODS)
    m = METHODS{k};
    vals = D.gp_mean_at_optimum.(m);
    vals = vals(:)';
    plot(t(1:numel(vals)), vals, 'LineWidth', 1.8, ...
        'Color', COLORS(k, :), 'DisplayName', LABELS{k});
end
yline(1.0, '--', 'Color', [0 0 0 0.4], 'LineWidth', 0.9);
xlabel('Iteration'); ylabel('GP posterior mean at true optimum');
title('GP mean at the optimum');
grid on; legend('Location', 'southeast');

% Right: std
subplot(1,2,2); hold on; box on;
for k = 1:numel(METHODS)
    m = METHODS{k};
    vals = D.gp_std_at_optimum.(m);
    vals = vals(:)';
    plot(t(1:numel(vals)), vals, 'LineWidth', 1.8, ...
        'Color', COLORS(k, :), 'DisplayName', LABELS{k});
end
xlabel('Iteration'); ylabel('GP posterior std at true optimum');
title('GP uncertainty at the optimum');
grid on; legend('Location', 'northeast');

sgtitle('GP surrogate evolution — EI vs JND-EI', 'FontWeight', 'bold');

exportgraphics(fig, fullfile(OUT_DIR, [OUT_NAME '.pdf']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(OUT_DIR, [OUT_NAME '.svg']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(OUT_DIR, [OUT_NAME '.png']), 'Resolution', 300);
fprintf('Saved %s.{pdf,svg,png}\n', OUT_NAME);
