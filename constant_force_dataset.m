clear all
close all
clc

% Simulation setup
t_f = 1;
tlist = linspace(0, t_f, 100);   % Nt time steps
Nt = numel(tlist);

% Dataset parameters
N_samples = 128;                  % randomized force location
Nx = 100; Ny = 100;               % uniform query grid

% Grids for storage and visualization
x_grid = linspace(0, 1, Nx);
y_grid = linspace(0, 1, Ny);
[Xq, Yq] = meshgrid(x_grid, y_grid);

% PDE coefficients
global source_width source_amp source_center_x source_center_y source_sign
wave_speed = 1;
d = 1;
a = 0;
m = 1;
source_width = 0.3;
source_amp = 5.0;
% Defaults so coefficient evaluation works before per-sample overrides
source_center_x = 0.5;
source_center_y = 0.5;
source_sign = 1;

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
    % Random force location (range [0.2, 0.8] in both x and y)
    source_center_x = 0.2 + 0.6 * rand();
    source_center_y = 0.2 + 0.6 * rand();
    source_sign = 2 * randi([0,1]) - 1;  % Random sign: +1 or -1
    
    % Compute and store force field for this sample
    for xi = 1:Nx
        for yi = 1:Ny
            x_val = x_grid(xi);
            y_val = y_grid(yi);
            F_data(xi, yi, sample_id) = source_sign * source_amp * ...
                exp(-((x_val - source_center_x)^2 + (y_val - source_center_y)^2) / source_width^2);
        end
    end

    % Zero ICs 
    [u0_fun, ut0_fun] = generate_zero_ic();
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
save(fullfile(script_dir, 'data', 'constant_force.mat'), 'U_data', 'V_data', 'F_data', 'x_grid', 'y_grid', 'tlist', '-v7.3');

% Visualize three random samples at mid time (displacement and force field)
mid_idx = round(Nt / 2);
pick_ids = randperm(N_samples, 3);
figure('Position', [100, 100, 1400, 800]);
tiledlayout(2, 3);

% Row 1: Displacement at mid time
for k = 1:3
    nexttile;
    surf(Xq, Yq, U_data(:, :, mid_idx, pick_ids(k))');
    shading interp;
    xlabel('x'); ylabel('y'); zlabel('u');
    title(sprintf('Sample %d: Displacement at t=%.3f', pick_ids(k), tlist(mid_idx)));
end

% Row 2: Force field
for k = 1:3
    nexttile;
    surf(Xq, Yq, F_data(:, :, pick_ids(k))');
    shading interp;
    xlabel('x'); ylabel('y'); zlabel('f');
    title(sprintf('Sample %d: Force Field', pick_ids(k)));
end

% ------ Helper functions ------
function fcoeff = force(location, state)
    global source_sign source_width source_amp source_center_x source_center_y %#ok<NUSED>
    fcoeff = source_sign * source_amp * exp(-(((location.x - source_center_x).^2) + ((location.y - source_center_y).^2)) / source_width^2);
end

function [u0_fun, ut0_fun] = generate_zero_ic()
    u0_fun = @(location) 0 * location.x;
    ut0_fun = @(location) 0 * location.x;
end
