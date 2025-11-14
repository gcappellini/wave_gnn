% MATLAB Reference Solution for 1D Heat Equation with Source
% PDE: u_t = u_xx + f(x)
% IC:  u(x, 0) = a * sin(pi * x)
% BC:  u(0, t) = u(1, t) = 0
% Domain: x ∈ [0, 1], t ∈ [0, 0.1]

clear; close all; clc;

%% Parameters (matching Python test case)
a_test = 1.5;           % IC amplitude
source_amplitude = 5.0; % Source amplitude
source_center = 0.50;   % Gaussian center
source_width = 0.3;     % Gaussian width

% Spatial domain
L = 1.0;
nx = 100;               % Number of spatial points
x = linspace(0, L, nx)';
dx = x(2) - x(1);

% Time domain
T_final = 0.1;
nt = 50;                % Number of time snapshots
t = linspace(0, T_final, nt);
dt = t(2) - t(1);

%% Initial Condition
u0 = a_test * sin(pi * x);

%% Source Function (Gaussian, space-only)
f_x = source_amplitude * exp(-((x - source_center) / source_width).^2);

%% PDE Solver using pdepe
% Define the PDE in the form: c(x,t,u,ux) * ut = x^(-m) * d/dx[x^m * f(x,t,u,ux)] + s(x,t,u,ux)
% For heat equation: u_t = u_xx + f(x)
% m = 0 (Cartesian coordinates)

m = 0;

% PDE function
function [c, f, s] = pdeFun(x, t, u, dudx, source_amplitude, source_center, source_width)
    c = 1;
    f = dudx;  % flux term (for u_xx)
    % Source term (Gaussian)
    source = source_amplitude * exp(-((x - source_center) / source_width)^2);
    s = source;
end

% Initial condition function
function u0 = icFun(x, a_test)
    u0 = a_test * sin(pi * x);
end

% Boundary condition function (Dirichlet: u(0,t) = u(1,t) = 0)
function [pl, ql, pr, qr] = bcFun(xl, ul, xr, ur, t)
    pl = ul;   % u(0,t) = 0
    ql = 0;
    pr = ur;   % u(1,t) = 0
    qr = 0;
end

%% Solve PDE using pdepe
fprintf('Solving 1D Heat Equation with pdepe...\n');
fprintf('Parameters:\n');
fprintf('  IC amplitude a = %.2f\n', a_test);
fprintf('  Source amplitude = %.2f\n', source_amplitude);
fprintf('  Spatial points: %d\n', nx);
fprintf('  Time points: %d\n', nt);
fprintf('  Domain: x ∈ [0, %.1f], t ∈ [0, %.2f]\n', L, T_final);
fprintf('Solving...\n');

% Wrap functions with parameters
pde = @(x, t, u, dudx) pdeFun(x, t, u, dudx, source_amplitude, source_center, source_width);
ic = @(x) icFun(x, a_test);
bc = @(xl, ul, xr, ur, t) bcFun(xl, ul, xr, ur, t);

% Solve
sol = pdepe(m, pde, ic, bc, x, t);

% Extract solution (sol is nt x nx)
u_solution = sol;  % u_solution(time_index, space_index)

fprintf('✓ Solution computed!\n\n');

%% Save solution to text file (3 columns: x, t, u)
fprintf('Saving solution to text file...\n');

% Create meshgrid
[X, T] = meshgrid(x, t);

% Flatten arrays
x_flat = X(:);
t_flat = T(:);
u_flat = u_solution(:);

% Create output matrix [x, t, u]
output_data = [x_flat, t_flat, u_flat];

% Save to file
output_file = 'heat_equation_matlab_solution.txt';
dlmwrite(output_file, output_data, 'delimiter', '\t', 'precision', 10);

fprintf('✓ Solution saved to: %s\n', output_file);
fprintf('  Format: 3 columns [x, t, u(x,t)]\n');
fprintf('  Total data points: %d\n', length(x_flat));
fprintf('\n');

%% Visualize Solution
figure('Position', [100, 100, 1200, 400]);

% Plot 1: Solution snapshots
subplot(1, 3, 1);
time_indices = [1, round(nt/4), round(nt/2), round(3*nt/4), nt];
hold on;
for idx = time_indices
    plot(x, u_solution(idx, :), 'LineWidth', 2, 'DisplayName', sprintf('t=%.3f', t(idx)));
end
hold off;
xlabel('x', 'FontSize', 12);
ylabel('u(x,t)', 'FontSize', 12);
title('Solution Evolution', 'FontSize', 12);
legend('Location', 'best');
grid on;

% Plot 2: Spatiotemporal heatmap
subplot(1, 3, 2);
contourf(T, X, u_solution, 20, 'LineStyle', 'none');
colorbar;
xlabel('t', 'FontSize', 12);
ylabel('x', 'FontSize', 12);
title('Solution u(x,t)', 'FontSize', 12);
colormap('jet');

% Plot 3: IC and Source
subplot(1, 3, 3);
yyaxis left
plot(x, u0, 'b--', 'LineWidth', 2, 'DisplayName', 'IC: u(x,0)');
ylabel('IC', 'FontSize', 12);
yyaxis right
plot(x, f_x, 'r-', 'LineWidth', 2, 'DisplayName', 'Source f(x)');
ylabel('Source', 'FontSize', 12);
xlabel('x', 'FontSize', 12);
title('Initial Condition & Source', 'FontSize', 12);
legend('Location', 'best');
grid on;

sgtitle('MATLAB Reference Solution: 1D Heat Equation', 'FontSize', 14, 'FontWeight', 'bold');

% Save figure
saveas(gcf, 'heat_equation_matlab_solution.png');
fprintf('✓ Figure saved to: heat_equation_matlab_solution.png\n');

%% Summary Statistics
fprintf('\n=== Solution Statistics ===\n');
fprintf('Initial energy: %.6e\n', trapz(x, u0.^2));
fprintf('Final energy: %.6e\n', trapz(x, u_solution(end, :).^2));
fprintf('Max temperature: %.6f\n', max(u_solution(:)));
fprintf('Min temperature: %.6f\n', min(u_solution(:)));
fprintf('\n=== DONE ===\n');
