% =========================================================================
% GENERATORE DATASET: MEMBRANA 90x90 PER TRAINING DEEPONET
% =========================================================================
clear; clc;

script_dir = fileparts(mfilename('fullpath'));
cfg = readyaml(fullfile(script_dir, '..', 'configs', 'config.yaml'));
p = cfg.datasets.contact;

%% 1. CONFIGURAZIONE SIMULAZIONE
Lx = p.domain_length;
N = p.nx;
if p.ny ~= N
    error('Contact dataset requires a square grid (nx must equal ny).');
end
dx = Lx / (N - 1);
inv_dx2 = 1.0 / (dx^2);
c = p.wave_speed;
c2 = c^2;
kappa = p.damping;

toolSize = p.tool_size;
halfTool = toolSize / 2;
inputVelocityFilter = p.input_velocity_filter;

duration = p.duration;
dt_fixed = p.dt_fixed;
subSteps = p.sub_steps;
dt = dt_fixed / subSteps;
num_frames = round(duration / dt_fixed);
if abs(dt - p.dt) > 1e-12 || num_frames ~= p.num_frames
    error('Contact time settings are inconsistent in configs/config.yaml.');
end
center_min = p.contact_center_range.min;
center_max = p.contact_center_range.max;
if center_min < p.contact_margin || center_max > Lx - p.contact_margin
    error('Contact center range must respect contact_margin and domain_length.');
end

num_simulations = p.n_samples;

dataset_dir = fullfile(script_dir, p.dataset_dir);
if ~exist(dataset_dir, 'dir')
    mkdir(dataset_dir);
end
output_path = fullfile(dataset_dir, p.output_file);
if isfile(output_path)
    error('Dataset already exists and will not be overwritten: %s', output_path);
end

% The sample axis keeps the complete dataset in one v7.3 file without RAM-sized arrays.
dataset = matfile(output_path, 'Writable', true);
dataset.U_hist(N, N, num_frames, num_simulations) = cast(0, p.saved_precision);
dataset.V_hist(N, N, num_frames, num_simulations) = cast(0, p.saved_precision);
dataset.Force(num_frames, 1, num_simulations) = cast(0, p.saved_precision);
dataset.Contact(num_frames, 1, num_simulations) = false;
dataset.CenterX(1, num_simulations) = 0;
dataset.CenterZ(1, num_simulations) = 0;
dataset.ToolY_hist(num_frames, 1, num_simulations) = cast(0, p.saved_precision);

% Pre-calcolo coordinate griglia
x_vec = linspace(0, Lx, N);
[X, Y_grid] = meshgrid(x_vec, x_vec);
area_hr = dx * dx;
time_vector = (0:num_frames-1) * dt_fixed;
dataset.Time(:, 1) = time_vector(:);

disp(['Inizio generazione di ', num2str(num_simulations), ' simulazioni...']);
disp(['File dataset: ', output_path]);
tic;

%% 2. CICLO DI GENERAZIONE (Un solo file .mat)
for sim = 1:num_simulations
    
    % --- A. INIZIALIZZAZIONE DEFORMAZIONE CON SERIE DI FOURIER ---
    U = zeros(N, N);
    V = zeros(N, N);
    
    for n = p.fourier_modes.min:p.fourier_modes.max
        for m = p.fourier_modes.min:p.fourier_modes.max
            A_coeff = (rand() - 0.5) * p.fourier_amplitude_factor / (n * m);
            U = U + A_coeff * sin(n * pi * X / Lx) .* sin(m * pi * Y_grid / Lx);
        end
    end
    
    % --- B. POSIZIONE RANDOMICA DEL CONTATTO ---
    % Manteniamo il centro ad almeno mezza toolSize dai bordi per non uscire
    target_x = center_min + rand() * (center_max - center_min);
    target_z = center_min + rand() * (center_max - center_min);
    
    % Maschera logica del tool
    inPatch = (abs(X - target_x) <= halfTool) & (abs(Y_grid - target_z) <= halfTool);
    
    % --- C. GENERAZIONE TRAIETTORIA TOOL (MANO SIMULATA) ---
    patch_avg_init = mean(U(inPatch));
    y_start = patch_avg_init;
    
    t_down = random_in_range(p.trajectory_timing.t_down);
    t_hold = t_down + random_in_range(p.trajectory_timing.delta_t_hold);
    t_up = t_hold + random_in_range(p.trajectory_timing.delta_t_up);
    
    y_depth = y_start + random_in_range(p.trajectory_depth_offset);
    y_air = y_start + p.trajectory_air_offset;
    
    t_keys = [0, t_down, t_hold, t_up, duration];
    y_keys = [y_start, y_depth, y_depth, y_air, y_air];
    
    smooth_y = pchip(t_keys, y_keys, time_vector);
    
    raw_v = zeros(num_frames, 1);
    raw_v(2:end) = diff(smooth_y) / dt_fixed;
    raw_v(1) = raw_v(2);
    
    % --- D. SETUP VARIABILI DI STATO LOCALI ---
    isContactActive = false;
    lastReactionForceHR = 0.0;
    filteredToolVelocityY = 0.0;
    
    % Buffer per questa singola simulazione
    sim_U_hist = zeros(N, N, num_frames, p.saved_precision);
    sim_V_hist = zeros(N, N, num_frames, p.saved_precision);
    sim_force = zeros(num_frames, 1, p.saved_precision);
    sim_contact = false(num_frames, 1);
    
    % --- E. CICLO FISICO ---
    for frame = 1:num_frames
        
        toolBottomEdge = smooth_y(frame);
        rawToolVelocityY = raw_v(frame);
        
        filteredToolVelocityY = (inputVelocityFilter * rawToolVelocityY) + ((1.0 - inputVelocityFilter) * filteredToolVelocityY);
        
        if ~isContactActive
            patchAvgTrueU = mean(U(inPatch));
            if toolBottomEdge <= patchAvgTrueU
                isContactActive = true;
            end
        else
                if toolBottomEdge > p.contact_thresholds.release_elevation || ...
                    (lastReactionForceHR <= p.contact_thresholds.min_reaction_force && ...
                    filteredToolVelocityY > p.contact_thresholds.min_release_velocity)
                isContactActive = false;
            end
        end
        
        forceTrueAccumulator = 0.0;
        
        for step = 1:subSteps
            lap = zeros(N, N);
            lap(2:end-1, 2:end-1) = (U(1:end-2, 2:end-1) + U(3:end, 2:end-1) + ...
                                     U(2:end-1, 1:end-2) + U(2:end-1, 3:end) - ...
                                     4.0 * U(2:end-1, 2:end-1)) * inv_dx2;
            
            if isContactActive
                if step == 1 
                    nodalForce = (c2 * lap(inPatch) - kappa * filteredToolVelocityY);
                    forceTrueAccumulator = sum(nodalForce) * area_hr;
                end
                
                accel = c2 * lap - kappa * V;
                V = V + accel * dt;
                U = U + V * dt;
                
                U(inPatch) = toolBottomEdge;
                V(inPatch) = filteredToolVelocityY;
            else
                accel = c2 * lap - kappa * V;
                V = V + accel * dt;
                U = U + V * dt;
            end
        end
        
        lastReactionForceHR = forceTrueAccumulator;
        
        % Salvataggio del frame
        sim_U_hist(:,:,frame) = cast(U, p.saved_precision);
        sim_V_hist(:,:,frame) = cast(V, p.saved_precision);
        if isContactActive
            sim_force(frame) = cast(forceTrueAccumulator, p.saved_precision);
        else
            sim_force(frame) = 0.0;
        end
        sim_contact(frame) = isContactActive;
    end
    
    % --- F. SALVATAGGIO NEL FILE UNICO ---
    dataset.U_hist(:, :, :, sim) = sim_U_hist;
    dataset.V_hist(:, :, :, sim) = sim_V_hist;
    dataset.Force(:, 1, sim) = sim_force;
    dataset.Contact(:, 1, sim) = sim_contact;
    dataset.CenterX(1, sim) = target_x;
    dataset.CenterZ(1, sim) = target_z;
    dataset.ToolY_hist(:, 1, sim) = cast(smooth_y(:), p.saved_precision);
            
    % Stampa a schermo ogni 100 simulazioni per feedback
    if mod(sim, 100) == 0
        fprintf('Completate %d / %d simulazioni.\n', sim, num_simulations);
    end
end
toc;
disp('Generazione Dataset completata con successo!');
disp(['Dataset salvato in: ', output_path]);

function value = random_in_range(range)
    value = range.min + rand() * (range.max - range.min);
end