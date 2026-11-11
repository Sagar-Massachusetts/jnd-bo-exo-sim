% Md Mainuddin Sagar
% fig7b_regret_bars.m
% Simple regret bar chart: all 5 methods x 4 surfaces, mean +/- std over seeds.
% Reads results/tables/sweep_summary.csv (20-seed).
%
% Running in my macbook: openning in MATLAB 2026, Current Folder = repo root, press Run.

clear; clc; close all;

%% STYLE BLOCK
OUT_DIR       = fullfile('figures', 'output');
OUT_NAME      = 'fig7b_summary_bars_matlab';
CSV           = fullfile('results', 'tables', 'sweep_summary_publication.csv');
METHODS       = {'random','grid','ei','ts','jnd_ei'};
METHOD_LABELS = containers.Map(METHODS, {'Random','Grid','EI','Thompson','JND-EI'});
SURFACES      = {'unimodal','multimodal','asymmetric','timing_sensitive'};
SURF_LABELS   = containers.Map(SURFACES, {'Unimodal','Multimodal','Asymmetric','Timing-sens.'});
METHOD_COLORS = containers.Map(METHODS, ...
    {[0.50 0.50 0.50],[0.55 0.34 0.29],[0.12 0.47 0.71],[0.17 0.63 0.17],[0.84 0.15 0.16]});
% 

if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end
if ~exist(CSV, 'file')
    error('Missing %s. Run demo_step7.py or demo_step9.py first.', CSV);
end
T = readtable(CSV);

nM = numel(METHODS); nS = numel(SURFACES); width = 0.16;

fig = figure('Color', 'w', 'Position', [100 100 1100 500]);
hold on; box on;
for mi = 1:nM
    m = METHODS{mi};
    means = zeros(1, nS); stds = zeros(1, nS);
    for si = 1:nS
        mask = strcmp(T.method, m) & strcmp(T.surface, SURFACES{si});
        means(si) = T.regret_mean(mask);
        stds(si)  = T.regret_std(mask);
    end
    x = (1:nS) + (mi - (nM+1)/2) * width;
    bar(x, means, width, 'FaceColor', METHOD_COLORS(m), ...
        'EdgeColor', 'k', 'DisplayName', METHOD_LABELS(m));
    errorbar(x, means, stds, 'k.', 'LineWidth', 0.8, 'HandleVisibility', 'off');
end
set(gca, 'XTick', 1:nS, ...
    'XTickLabel', cellfun(@(s) SURF_LABELS(s), SURFACES, 'UniformOutput', false), ...
    'XTickLabelRotation', 15, 'FontSize', 11);
ylabel('Simple regret (lower is better)', 'FontSize', 11);
%title('Simple regret across methods and surfaces (mean \pm std over 20 seeds)', ...
     % 'FontSize', 11);
grid off;
legend('Location', 'northeast', 'NumColumns', 2, 'FontSize', 11);

exportgraphics(fig, fullfile(OUT_DIR, [OUT_NAME '.pdf']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(OUT_DIR, [OUT_NAME '.svg']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(OUT_DIR, [OUT_NAME '.png']), 'Resolution', 600);
fprintf('Saved %s.{pdf,svg,png}\n', OUT_NAME);
