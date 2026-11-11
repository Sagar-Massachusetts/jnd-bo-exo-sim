% Md Mainuddin Sagar
% fig9_tost.m
% TOST equivalence figure. Runs standalone in MATLAB 2026 in my macbook.
% Reads results/logs/tost_summary.json.

clear; clc; close all;

%% STYLE BLOCK 
%FIG_TITLE      = 'TOST equivalence of EI and JND-EI optima (JND-informed bounds)';
OUT_DIR        = fullfile('figures', 'output');
OUT_NAME       = 'fig9_tost_equivalence_matlab';
SURFACES_ORDER = {'unimodal','multimodal','asymmetric','timing_sensitive'};
PARAM_JND      = [0.030, 0.030, 0.030, 0.100];
PARAM_LABELS   = {'Rise time','Peak timing','Fall time','Peak torque'};
% 

if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
tost = jsondecode(fileread(fullfile('results','logs','tost_summary_publication.json')));

fig = figure('Color','w','Position',[100 100 1000 750]);

for si = 1:numel(SURFACES_ORDER)
    s = SURFACES_ORDER{si};
    subplot(2,2,si); hold on; box on;
    if ~isfield(tost, s)
        text(0.5,0.5,sprintf('%s: no TOST data', s),'Units','normalized', ...
            'HorizontalAlignment','center');
        continue;
    end
    r = tost.(s);
    for i = 1:4
        p = r.params(i);
        m  = p.mean_diff;
        % --- CI fields: new schema first, legacy fallback 
        if isfield(p,'ci95_lo')
            lo = p.ci95_lo;  hi = p.ci95_hi;
        else
            lo = p.ci_lo;    hi = p.ci_hi;
        end
        % --- equivalence flag: new schema first, legacy fallback 
        if isfield(p,'equivalent_95')
            eq = p.equivalent_95;
        else
            eq = p.equivalent;
        end

        if eq, c = [0.2 0.7 0.2]; else, c = [0.85 0.2 0.2]; end
        errorbar(m, i, m-lo, hi-m, 'o', 'Color', c, 'MarkerFaceColor', c, ...
            'MarkerSize', 8, 'LineWidth', 2, 'CapSize', 6);
        plot([-PARAM_JND(i) -PARAM_JND(i)], [i-0.3 i+0.3], 'k--', 'LineWidth', 1);
        plot([ PARAM_JND(i)  PARAM_JND(i)], [i-0.3 i+0.3], 'k--', 'LineWidth', 1);
    end
    xline(0, 'Color',[0.5 0.5 0.5 0.3], 'LineWidth', 1);
    set(gca,'YTick',1:4,'YTickLabel',PARAM_LABELS,'FontSize',11);
    xlabel('Mean difference (EI - JND-EI)');

    % --- all-equivalent flag: new schema first, legacy fallback ---
    if isfield(r,'all_equivalent_95')
        all_eq = r.all_equivalent_95;
    else
        all_eq = r.all_equivalent;
    end
    if all_eq, eqstr = 'TRUE'; else, eqstr = 'FALSE'; end
    title(sprintf('%s (all eq = %s)', s, eqstr));
    grid off;
end

%sgtitle(FIG_TITLE,'FontWeight','bold','FontSize',12);
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '.pdf']), 'ContentType','vector');
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '.eps']), 'ContentType','vector');
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '.svg']), 'ContentType','vector');
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '.png']), 'Resolution',1200);
fprintf('Saved %s.{pdf,svg,png}\n', OUT_NAME);
