clear all
close all
clc

% Simulation setup
t_f = 1;
tlist = linspace(0, t_f, 100);   % Nt time steps
Nt = numel(tlist);

% Dataset parameters
N_samples = 1;                    % single validation sample
Nx = 100; Ny = 100;               % uniform query grid
K = 4; L = 4;                     % kept for compatibility (unused)

% Grids for storage and visualization
x_grid = linspace(0, 1, Nx);
y_grid = linspace(0, 1, Ny);
[Xq, Yq] = meshgrid(x_grid, y_grid);

% PDE coefficients
global source_width source_amp source_center_x source_center_y
wave_speed = 1;
d = 20;
a = 0;
m = 0.1;
source_center_x = 0.35;
source_center_y = 0.65;
source_width = 0.3;
source_amp = 10.0;                 

% Build geometry and mesh once
model = createpde(1);
R1 = [3, 4, 0, 1, 1, 0, 0, 0, 1, 1]';
g = decsg(R1);
geometryFromEdges(model, g);

specifyCoefficients(model, m = m, d = 0, c = wave_speed^2, a = a, f = @force);
applyBoundaryCondition(model, "dirichlet", "Edge", [1, 2, 3, 4], "u", 0);

generateMesh(model);
timeStruct = struct('time', 0);
results = assembleFEMatrices(model, timeStruct);
specifyCoefficients(model, m = m, d = d * results.M, c = wave_speed^2, a = a, f = @force);

mesh = model.Mesh;

% Preallocate storage for displacement and velocity over space-time for all samples
U_data = zeros(Nx, Ny, Nt, N_samples);
V_data = zeros(Nx, Ny, Nt, N_samples);
F_data = zeros(Nx, Ny, N_samples);  % Force field (constant in time)

cpu_time_start = cputime;
for sample_id = 1:N_samples
    % Compute and store force field for this sample
    for xi = 1:Nx
        for yi = 1:Ny
            x_val = x_grid(xi);
            y_val = y_grid(yi);
            F_data(xi, yi, sample_id) = source_amp * ...
                exp(-((x_val - source_center_x)^2 + (y_val - source_center_y)^2) / source_width^2);
        end
    end

    % Deterministic, simple IC (slightly out of training amplitude range)
    [u0_fun, ut0_fun] = generate_simple_ic();
    setInitialConditions(model, u0_fun, ut0_fun);

    % Solve PDE for this IC
    result = solvepde(model, tlist);

    % Sample solution on uniform grid for storage
    for ti = 1:Nt
        u_grid = interpolateSolution(result, Xq(:), Yq(:), ti);
        U_data(:, :, ti, sample_id) = reshape(u_grid, Ny, Nx)';
    end
    
    % Compute velocity via temporal finite differences on displacement
    for ti = 1:Nt
        if ti == 1
            % Forward difference for first time step
            u_curr = reshape(interpolateSolution(result, Xq(:), Yq(:), ti), Ny, Nx)';
            u_next = reshape(interpolateSolution(result, Xq(:), Yq(:), ti+1), Ny, Nx)';
            v_grid = (u_next - u_curr) / (tlist(ti+1) - tlist(ti));
        elseif ti == Nt
            % Backward difference for last time step
            u_curr = reshape(interpolateSolution(result, Xq(:), Yq(:), ti), Ny, Nx)';
            u_prev = reshape(interpolateSolution(result, Xq(:), Yq(:), ti-1), Ny, Nx)';
            v_grid = (u_curr - u_prev) / (tlist(ti) - tlist(ti-1));
        else
            % Central difference for interior time steps
            u_next = reshape(interpolateSolution(result, Xq(:), Yq(:), ti+1), Ny, Nx)';
            u_prev = reshape(interpolateSolution(result, Xq(:), Yq(:), ti-1), Ny, Nx)';
            v_grid = (u_next - u_prev) / (tlist(ti+1) - tlist(ti-1));
        end
        V_data(:, :, ti, sample_id) = v_grid;
    end

    fprintf('Finished sample %d/%d\n', sample_id, N_samples);
end
cpu_time_end = cputime - cpu_time_start

% Save dataset to script directory
script_dir = fileparts(mfilename('fullpath'));
% save(fullfile(script_dir, 'data', 'constant_force_test.mat'), 'U_data', 'V_data', 'F_data', 'x_grid', 'y_grid', 'tlist', '-v7.3');

% Compute zlim ranges from full data
u_min = min(U_data(:));
u_max = max(U_data(:));
v_min = min(V_data(:));
v_max = max(V_data(:));

% Add small buffer if min==max to avoid degenerate limits
if u_min == u_max
    u_min = u_min - 0.1;
    u_max = u_max + 0.1;
end
if v_min == v_max
    v_min = v_min - 0.1;
    v_max = v_max + 0.1;
end

% Animate the single sample over time (displacement and velocity)
figure;
tiledlayout(1, 2);

nexttile;
hU = surf(Xq, Yq, U_data(:, :, 1, 1)');
shading interp;
xlabel('x'); ylabel('y'); zlabel('u');
title(sprintf('Displacement at t=%.3f', tlist(1)));
zlim([u_min u_max]);

nexttile;
hV = surf(Xq, Yq, V_data(:, :, 1, 1)');
shading interp;
xlabel('x'); ylabel('y'); zlabel('v');
title(sprintf('Velocity at t=%.3f', tlist(1)));
zlim([v_min v_max]);

% GIF output settings
gif_path = fullfile(script_dir, 'data/constant_force_animation.gif');
frame_delay = 0.05; % seconds

for ti = 1:Nt
    set(hU, 'ZData', U_data(:, :, ti, 1)');
    set(hV, 'ZData', V_data(:, :, ti, 1)');
    title(hU.Parent, sprintf('Displacement at t=%.3f', tlist(ti)));
    title(hV.Parent, sprintf('Velocity at t=%.3f', tlist(ti)));
    drawnow;

    frame = getframe(gcf);
    [im, cmap] = rgb2ind(frame2im(frame), 256);
    if ti == 1
        imwrite(im, cmap, gif_path, 'gif', 'LoopCount', inf, 'DelayTime', frame_delay);
    else
        imwrite(im, cmap, gif_path, 'gif', 'WriteMode', 'append', 'DelayTime', frame_delay);
    end
end

% ------ Helper functions ------
function fcoeff = force(location, state)
    global source_width source_amp source_center_x source_center_y %#ok<NUSED>
    fcoeff = source_amp * exp(-(((location.x - source_center_x).^2) + ((location.y - source_center_y).^2)) / source_width^2);
end

function [u0_fun, ut0_fun] = generate_simple_ic()
    % Zero initial conditions - dynamics driven purely by constant force
    u0_fun = @(location) 0 * location.x;
    ut0_fun = @(location) 0 * location.x;
end

function [u0_fun, ut0_fun] = generate_random_ic(K, L)
    if nargin < 1
        K = 4; L = 4;
    end

    u_coeffs = make_coeffs(K, L);
    v_coeffs = make_coeffs(K, L);

    u0_fun = @(location) fourier_field(location.x, location.y, u_coeffs);
    ut0_fun = @(location) fourier_field(location.x, location.y, v_coeffs);
end

function coeffs = make_coeffs(K, L)
    raw = -1 + 2 * rand(K, L);
    [k_idx, l_idx] = ndgrid(1:K, 1:L);
    decay = 1 ./ (k_idx.^2 + l_idx.^2);
    coeffs = raw .* decay;
end

function val = fourier_field(x, y, coeffs)
    [K, L] = size(coeffs);
    val = zeros(size(x));
    for k = 1:K
        for l = 1:L
            val = val + coeffs(k, l) .* sin(k * pi * x) .* sin(l * pi * y);
        end
    end
end