% Md Mainuddin Sagar
% fig1_2_walker_profile.m
% Generating fig1 (4-parameter ankle torque profile) and fig2 (gait phase).
% Pure MATLAB
%
% Run process: opening in MATLAB 2026 in my macbook, Current Folder = repo root, press Run.

clear; clc; close all;

%% STYLE BLOCK
OUT_DIR   = fullfile('figures', 'output');
FIG1_NAME = 'fig1_torque_profile_matlab';
FIG2_NAME = 'fig2_walker_phase_matlab';
TITLES    = {'4-parameter ankle torque profile', ...
             'Walker gait phase over 3 strides (period = 1.0 s)'};
LINE_WIDTH = 1.8;
COLORS     = lines(4);
% 

if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

%% Torque profile (fig1) 
% Bounds from src/torque_profile.py
lo = [0.10 0.30 0.05 0.10];
hi = [0.40 0.60 0.20 1.00];

torque = @(phase, p, m) torque_fn(mod(phase, 1), p, m);

param_sets = {
    'low magnitude',  [0.20 0.45 0.10 0.25];
    'mid magnitude',  [0.20 0.45 0.10 0.50];
    'high magnitude', [0.20 0.45 0.10 0.85];
    'later peak',     [0.20 0.55 0.10 0.50];
};

phases = linspace(0, 1, 501);
mass = 70;

fig1 = figure('Color', 'w', 'Position', [100 100 800 500]);
hold on; box on;
for i = 1:size(param_sets, 1)
    label = param_sets{i, 1};
    p = param_sets{i, 2};
    tau = arrayfun(@(x) torque(x, p, mass), phases);
    plot(phases, tau, 'LineWidth', LINE_WIDTH, 'Color', COLORS(i, :), ...
         'DisplayName', label);
end
xlabel('Gait phase (fraction of stride)');
ylabel(sprintf('Ankle torque (N\\cdotm), body mass %d kg', mass));
title(TITLES{1});
xlim([0 1]);
grid on;
legend('Location', 'northeast', 'FontSize', 8);

exportgraphics(fig1, fullfile(OUT_DIR, [FIG1_NAME '.pdf']), 'ContentType', 'vector');
exportgraphics(fig1, fullfile(OUT_DIR, [FIG1_NAME '.svg']), 'ContentType', 'vector');
exportgraphics(fig1, fullfile(OUT_DIR, [FIG1_NAME '.png']), 'Resolution', 300);
fprintf('Saved %s.{pdf,svg,png}\n', FIG1_NAME);

%% Gait phase (fig2) 
dt = 0.01;
t = 0:dt:3;
ph = mod(t / 1.0, 1);   % stride = 1.0 s

fig2 = figure('Color', 'w', 'Position', [100 100 800 350]);
hold on; box on;
plot(t, ph, 'LineWidth', LINE_WIDTH, 'Color', COLORS(1, :));
for k = 0:3
    xline(k, '--', 'Color', [0.5 0.5 0.5 0.4]);
end
xlabel('Time (s)');
ylabel('Gait phase');
title(TITLES{2});
xlim([0 3]); ylim([0 1.02]);
grid on;

exportgraphics(fig2, fullfile(OUT_DIR, [FIG2_NAME '.pdf']), 'ContentType', 'vector');
exportgraphics(fig2, fullfile(OUT_DIR, [FIG2_NAME '.svg']), 'ContentType', 'vector');
exportgraphics(fig2, fullfile(OUT_DIR, [FIG2_NAME '.png']), 'Resolution', 300);
fprintf('Saved %s.{pdf,svg,png}\n', FIG2_NAME);

%% Local helper: torque profile 
function tau = torque_fn(phase, p, mass)
    rise   = p(1); peak_t = p(2); fall = p(3); peak_tq = p(4);
    rise   = min(max(rise,   0.10), 0.40);
    peak_t = min(max(peak_t, 0.30), 0.60);
    fall   = min(max(fall,   0.05), 0.20);
    peak_tq= min(max(peak_tq,0.10), 1.00);

    tau_peak = peak_tq * mass;
    t_start  = peak_t - rise;
    t_end    = peak_t + fall;

    if tau_peak <= 0, tau = 0; return; end

    if phase >= t_start && phase < peak_t
        eff_start = max(t_start, 0);
        denom = max(peak_t - eff_start, 1e-9);
        tau = tau_peak * (phase - eff_start) / denom;
    elseif phase >= peak_t && phase < t_end
        denom = max(t_end - peak_t, 1e-9);
        tau = tau_peak * (t_end - phase) / denom;
    else
        tau = 0;
    end
end
