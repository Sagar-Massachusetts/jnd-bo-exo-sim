% Md Mainuddin Sagar
clear; clc; close all;

%% STYLE BLOCK
%FIG_TITLE      = 'JND-informed Bayesian optimization for ankle-exoskeleton tuning';
OUT_DIR        = fullfile('figures', 'output');
OUT_NAME       = 'fig10_master_summary_matlab';
METHODS        = {'random','grid','ei','ts','jnd_ei'};
METHOD_LABELS  = containers.Map(METHODS, ...
                   {'Random','Grid','EI','Thompson','JND-EI'});
SURFACES       = {'unimodal','multimodal','asymmetric','timing_sensitive'};
SURFACE_LABELS = containers.Map(SURFACES, ...
                   {'Unimodal','Multimodal','Asymmetric','Time Sensitive'});
METHOD_COLORS  = containers.Map(METHODS, ...
    {[0.50 0.50 0.50],[0.55 0.34 0.29],[0.12 0.47 0.71],[0.17 0.63 0.17],[0.84 0.15 0.16]});
% 

if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

% Load data
T    = readtable(fullfile('results','tables','sweep_summary_publication.csv'));
tost = jsondecode(fileread(fullfile('results','logs','tost_summary_publication.json')));

nM = numel(METHODS); nS = numel(SURFACES); width = 0.16;

%% (a) Regret bars
fig = figure('Color','w','Position',[100 100 700 500],'Visible','on');
hold on; box on;
for mi = 1:nM
    m = METHODS{mi};
    means = zeros(1,nS); stds = zeros(1,nS);
    for si = 1:nS
        mask = strcmp(T.method, m) & strcmp(T.surface, SURFACES{si});
        means(si) = T.regret_mean(mask);
        stds(si)  = T.regret_std(mask);
    end
    x = (1:nS) + (mi - (nM+1)/2) * width;
    bar(x, means, width, 'FaceColor', METHOD_COLORS(m), 'EdgeColor','k');
    errorbar(x, means, stds, 'k.', 'LineWidth', 0.8);
end
% axes: tick labels + rotation 
set(gca,'XTick',1:nS, ...
    'XTickLabel', cellfun(@(s) SURFACE_LABELS(s), SURFACES, 'UniformOutput',false), ...
    'XTickLabelRotation', 15);

%  global axes styling (set BEFORE ylabel so label inherits font) 
ax = gca;
ax.FontSize          = 11;         % tick labels (x and y values)
ax.FontName          = 'Helvetica';
ax.LineWidth         = 1.0;
ax.YLabel.FontSize   = 12;
ax.YLabel.FontName   = 'Helvetica';
% ax.Title.FontSize  = 12;         % uncomment if you re-enable title

%  axis label (created after font is set) 
ylabel('Simple regret (lower is better)');
%title('(a) Regret across surfaces');

% legend 
legend(cellfun(@(m) METHOD_LABELS(m), METHODS, 'UniformOutput',false), ...
    'Location','best','FontSize',10,'NumColumns',2);

grid off;    % restore original; use grid off only if intentional and applied to all panels

% export
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '_a.pdf']), 'ContentType','vector');
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '_a.eps']), 'ContentType','vector');
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '_a.svg']), 'ContentType','vector');
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '_a.png']), 'Resolution',1200);

%% (b) TOST
fig = figure('Color','w','Position',[100 100 700 500],'Visible','on');
hold on; box on;

% 1) list only surface entries, skip 'meta' if present
snames = fieldnames(tost);
snames = snames(~strcmp(snames,'meta'));
nS_b   = numel(snames);

%  2) read equivalence flag from either schema
vals = zeros(1, nS_b);
ps   = nan(1, nS_b);
deg  = false(1, nS_b);
for i = 1:nS_b
    s = tost.(snames{i});
    if isfield(s,'all_equivalent_95')        % publication-safe schema
        vals(i) = double(s.all_equivalent_95);
    elseif isfield(s,'all_equivalent')       % legacy schema
        vals(i) = double(s.all_equivalent);
    else
        vals(i) = NaN;
    end
    if isfield(s,'worst_p'),         ps(i)  = s.worst_p;         end
    if isfield(s,'degenerate_pair'), deg(i) = s.degenerate_pair; end
end

%  3) colors: green = equivalent, red = not, grey = missing 
barColors = zeros(nS_b,3);
for i = 1:nS_b
    if isnan(vals(i)),     barColors(i,:) = [0.70 0.70 0.70];
    elseif vals(i),        barColors(i,:) = [0.20 0.70 0.20];
    else,                  barColors(i,:) = [0.85 0.20 0.20];
    end
end

% 4) bars + p-value labels 
b = bar(1:nS_b, vals, 'FaceColor','flat');
b.CData = barColors;
for i = 1:nS_b
    if ~isnan(ps(i))
        text(i, 0.5, sprintf('p=%.3f', ps(i)), ...
            'HorizontalAlignment','center', ...
            'Color','w','FontWeight','bold','FontSize',8);
    end
end

% 5) flag degenerate surfaces (EI == JND-EI) so the figure is honest
for i = 1:nS_b
    if deg(i)
        text(i, 1.05, 'degenerate', ...
            'HorizontalAlignment','center','FontSize',7, ...
            'Color',[0.4 0.4 0.4],'FontAngle','italic');
    end
end

% 6) axes
%  axes: tick labels + rotation 
set(gca,'XTick',1:nS_b,'XTickLabel',snames,'XTickLabelRotation',15);
ylim([0 1.2]);

% global axes styling (setting BEFORE ylabel so label inherits font)
ax = gca;
ax.FontSize          = 11;         % tick labels (x and y values)
ax.FontName          = 'Helvetica';
ax.LineWidth         = 1.0;
ax.YLabel.FontSize   = 12;
ax.YLabel.FontName   = 'Helvetica';

ylabel('1 = equivalent within JND');
%title('(b) TOST: EI vs JND-EI');

% 5b) status legend 
hGreen = patch(NaN,NaN,[0.20 0.70 0.20]);
hRed   = patch(NaN,NaN,[0.85 0.20 0.20]);
hGrey  = patch(NaN,NaN,[0.70 0.70 0.70]);
legend([hGreen hRed hGrey], ...
    {'equivalent (95%)','not equivalent','n/a'}, ...
    'Location','northeast','FontSize',10,'Box','off');

grid off;    % match panels (a)/(c); use off only if applied to all three

% exportting
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '_b.pdf']), 'ContentType','vector');
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '_b.eps']), 'ContentType','vector');
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '_b.svg']), 'ContentType','vector');
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '_b.png']), 'Resolution',1200);

%% (c) Trials to 80%
fig = figure('Color','w','Position',[100 100 700 500],'Visible','on');
hold on; box on;
for mi = 1:nM
    m = METHODS{mi};
    means = zeros(1,nS);
    for si = 1:nS
        mask = strcmp(T.method, m) & strcmp(T.surface, SURFACES{si});
        means(si) = T.trials_to_80pct_mean(mask);
    end
    x = (1:nS) + (mi - (nM+1)/2) * width;
    bar(x, means, width, 'FaceColor', METHOD_COLORS(m), 'EdgeColor','k');
end
% axes: tick labels + rotation
set(gca,'XTick',1:nS, ...
    'XTickLabel', cellfun(@(s) SURFACE_LABELS(s), SURFACES, 'UniformOutput',false), ...
    'XTickLabelRotation', 15);

% global axes styling (set BEFORE ylabel so label inherits font)
ax = gca;
ax.FontSize          = 11;         % tick labels (x and y values)
ax.FontName          = 'Helvetica';
ax.LineWidth         = 1.0;
ax.YLabel.FontSize   = 12;
ax.YLabel.FontName   = 'Helvetica';
% ax.Title.FontSize  = 12;         % uncomment if you re-enable title

% axis label (created after font is set)
ylabel('Trials to 80% of optimum');
%title('(c) Sample efficiency');

% legend (match panel (a): 10-pt, 2 columns) 
legend(cellfun(@(m) METHOD_LABELS(m), METHODS, 'UniformOutput',false), ...
    'Location','best','FontSize',10,'NumColumns',2);

grid off;    % match panels (a)/(b); use off only if applied to all three

%  exporting
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '_c.pdf']), 'ContentType','vector');
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '_c.eps']), 'ContentType','vector');
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '_c.svg']), 'ContentType','vector');
exportgraphics(fig, fullfile(OUT_DIR,[OUT_NAME '_c.png']), 'Resolution',1200);

fprintf('Saved individual panels %s_{a,b,c,d,e,f}.{pdf,svg,png}\n', OUT_NAME);
