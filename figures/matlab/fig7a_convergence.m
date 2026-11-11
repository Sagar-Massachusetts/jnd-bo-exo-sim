% Md Mainuddin Sagar
% fig7a_convergence.m
% Convergence-band figure (mean ± 95% CI across seeds).
% Reads results/tables/sweep_summary.csv for final values; if you need the
% full curves, run demo_step7.py first to export histories, or use the
% Python script figures/fig7a_convergence.py.
%
% Runs standalone in MATLAB 2026 in my macbook.

clear; clc; close all;

%%  STYLE BLOCK 
%FIG_TITLE      = 'Best-so-far reward across acquisition strategies';
OUT_DIR        = fullfile('figures', 'output');
OUT_NAME       = 'fig7a_convergence_bands_matlab';
SURFACES       = {'unimodal','multimodal','asymmetric','timing_sensitive'};
METHODS        = {'random','ei','ts','jnd_ei'};
METHOD_LABELS  = containers.Map(METHODS, {'Random','EI','Thompson','JND-EI'});
METHOD_COLORS  = containers.Map(METHODS, ...
    {[0.50 0.50 0.50],[0.12 0.47 0.71],[0.17 0.63 0.17],[0.84 0.15 0.16]});
% 

if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
T = readtable(fullfile('results','tables','sweep_summary_publication.csv'));

fig = figure('Color','w','Position',[100 100 1000 750]);
for si = 1:numel(SURFACES)
    subplot(2,2,si); hold on; box on;
    for mi = 1:numel(METHODS)
        m = METHODS{mi};
        mask = strcmp(T.method, m) & strcmp(T.surface, SURFACES{si});
        if ~any(mask), continue; end
        mean_val = T.regret_mean(mask);
        std_val  = T.regret_std(mask);
        errorbar(mi, mean_val, std_val, 'o', ...
            'Color', METHOD_COLORS(m), 'MarkerFaceColor', METHOD_COLORS(m), ...
            'MarkerSize', 8, 'LineWidth', 1.5, 'CapSize', 6);
    end
    set(gca,'XTick',1:numel(METHODS), ...
        'XTickLabel', cellfun(@(m) METHOD_LABELS(m), METHODS, 'UniformOutput',false), ...
        'FontSize', 11);
    ylabel('Simple regret (lower is better)');
    title(SURFACES{si});
    ylim([0 1]);
    grid off;
end
%sgtitle(FIG_TITLE,'FontWeight','bold','FontSize',12);
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '.pdf']), 'ContentType','vector');
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '.eps']), 'ContentType','vector');
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '.svg']), 'ContentType','vector');
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '.png']), 'Resolution',1200);
fprintf('Saved %s.{pdf,svg,png}\n', OUT_NAME);
