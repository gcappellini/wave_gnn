clear all
close all
clc

script_dir = fileparts(mfilename('fullpath'));
cfg = readyaml(fullfile(script_dir, '..', 'configs', 'config.yaml'));
p = cfg.datasets.free_evolution;
Lx = p.domain_length;

% Simulation setup
t_f = p.t_final;
tlist = linspace(0, t_f, p.nt_time_steps);
Nt = numel(tlist);

% Dataset parameters
N_samples = p.n_samples;
Nx = p.nx; Ny = p.ny;
K = p.fourier_modes_k; L = p.fourier_modes_l;

% Grids for storage and visualization
x_grid = linspace(0, Lx, Nx);
y_grid = linspace(0, Lx, Ny);
[Xq, Yq] = meshgrid(x_grid, y_grid);

% PDE coefficients
global source_width source_amp source_center_x source_center_y
wave_speed = p.wave_speed;
d = p.damping;
a = p.reaction_coefficient;
m = p.mass;
source_center_x = p.source_center_x;
source_center_y = p.source_center_y;
source_width = p.source_width;
source_amp = p.source_amplitude;

% Build geometry and mesh once
model = createpde(1);
R1 = [3, 4, 0, Lx, Lx, 0, 0, 0, Lx, Lx];
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

cpu_time_start = cputime;
for sample_id = 1:N_samples
    % Random ICs via Fourier synthesis with decaying coefficients
    [u0_fun, ut0_fun] = generate_random_ic( ...
        K, L, p.fourier_coefficient_min, p.fourier_coefficient_max, p.fourier_decay_power);
    setInitialConditions(model, u0_fun, ut0_fun);

    % Solve PDE for this IC
    result = solvepde(model, tlist);

    % Sample solution on uniform grid for storage
    for ti = 1:Nt
        u_grid = interpolateSolution(result, Xq(:), Yq(:), ti);
        U_data(:, :, ti, sample_id) = reshape(u_grid, Ny, Nx);
    end
    
    % Compute velocity via temporal finite differences on displacement
    for ti = 1:Nt
        if ti == 1
            % Forward difference for first time step
            u_curr = reshape(interpolateSolution(result, Xq(:), Yq(:), ti), Ny, Nx);
            u_next = reshape(interpolateSolution(result, Xq(:), Yq(:), ti+1), Ny, Nx);
            v_grid = (u_next - u_curr) / (tlist(ti+1) - tlist(ti));
        elseif ti == Nt
            % Backward difference for last time step
            u_curr = reshape(interpolateSolution(result, Xq(:), Yq(:), ti), Ny, Nx);
            u_prev = reshape(interpolateSolution(result, Xq(:), Yq(:), ti-1), Ny, Nx);
            v_grid = (u_curr - u_prev) / (tlist(ti) - tlist(ti-1));
        else
            % Central difference for interior time steps
            u_next = reshape(interpolateSolution(result, Xq(:), Yq(:), ti+1), Ny, Nx);
            u_prev = reshape(interpolateSolution(result, Xq(:), Yq(:), ti-1), Ny, Nx);
            v_grid = (u_next - u_prev) / (tlist(ti+1) - tlist(ti-1));
        end
        V_data(:, :, ti, sample_id) = v_grid;
    end

    fprintf('Finished sample %d/%d\n', sample_id, N_samples);
end
cpu_time_end = cputime - cpu_time_start

% Save dataset to script directory
save(fullfile(script_dir, 'free_evolution.mat'), 'U_data', 'V_data', 'x_grid', 'y_grid', 'tlist', '-v7.3');

% Visualize three random samples at mid time
mid_idx = round(Nt / 2);
pick_ids = randperm(N_samples, 3);
figure;
tiledlayout(1, 3);
for k = 1:3
    nexttile;
    surf(Xq, Yq, U_data(:, :, mid_idx, pick_ids(k)));
    shading interp;
    xlabel('x'); ylabel('y'); zlabel('u');
    title(sprintf('Sample %d at t=%.3f', pick_ids(k), tlist(mid_idx)));
end

% ------ Helper functions ------
function fcoeff = force(location, state)
    global source_width source_amp source_center_x source_center_y %#ok<NUSED>
    fcoeff = 0 * source_amp * exp(-(((location.x - source_center_x).^2) + ((location.y - source_center_y).^2)) / source_width^2);
end

function [u0_fun, ut0_fun] = generate_random_ic(K, L, coefficient_min, coefficient_max, decay_power)
    u_coeffs = make_coeffs(K, L, coefficient_min, coefficient_max, decay_power);
    v_coeffs = make_coeffs(K, L, coefficient_min, coefficient_max, decay_power);

    u0_fun = @(location) fourier_field(location.x, location.y, u_coeffs);
    ut0_fun = @(location) fourier_field(location.x, location.y, v_coeffs);
end

function coeffs = make_coeffs(K, L, coefficient_min, coefficient_max, decay_power)
    raw = coefficient_min + (coefficient_max - coefficient_min) * rand(K, L);
    [k_idx, l_idx] = ndgrid(1:K, 1:L);
    decay = 1 ./ (k_idx.^2 + l_idx.^2).^decay_power;
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