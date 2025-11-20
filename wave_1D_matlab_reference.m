clear
close all
clc

%% Global variables
global a b T mu k t_f source_amp source_width
T = 1;
mu = 1;
k = 1;
t_f = 10;
source_amp = 1;    % Gaussian source amplitude
a = 0.5;
b = 2.0;

% Gaussian source parameters
source_width = 0.1;


x = linspace(0,1,100);
t = linspace(0,t_f,50);


%% CREATE PDE for a vibrating string

cpu_time_start = cputime;

function f = stringforce(x, t)
    global source_width source_amp
    % Source center changes randomly every second in range [0.1, 0.9]
    % Use floor(t) to get which second we're in, then use it as random seed
    % rng(floor(t), 'twister');  % Set random seed based on current second
    % source_center = 0.1 + 0.8 * rand();  % Random center in [0.1, 0.9]
    source_center = 0.1 + 0.8 * mod(floor(t),10)/9;  % Deterministic center changing every second
    % source_center = 0.17;
    f = source_amp * exp(-((x - source_center)/source_width)^2);
end

function [c,f,s] = stringpde(x, t, u, dudx)
    global k T mu
    c = [1; 1];
    F = 15*stringforce(x, t);
    s = [u(2); F - k*u(2)];
    f = [0; (T/mu)*dudx(1)];
end

function u_0 = stringic(x)
    global a b
    u_0 = [a*sin(pi*x);b*sin(pi*x)];
end

function [pl,ql,pr,qr] = stringbc(xl,ul,xr,ur,t)
    pl = [ul(1); ul(2)];
    ql = [0; 0];
    pr = [ur(1); ur(2)];
    qr = [0; 0];
end

m = 0;
sol = pdepe(m,@stringpde,@stringic,@stringbc,x,t);

cpu_time_end = cputime-cpu_time_start

%% Save solution to CSV files
% Combine x, t, force, displacement, and velocity into one file
output_data = [];
for i = 1:length(t)
    for j = 1:length(x)
        output_data = [output_data; x(j), t(i), stringforce(x(j), t(i)), sol(i, j, 1), sol(i, j, 2)];
    end
end
writematrix(output_data, '/Users/guglielmocappellini/Desktop/research/code/pinns-wave/wave-gnn/1_gcn_string/data/gt_wave1D_with_source_rollout.csv');

%% PLOT SOLUTION

figure
pcolor(x,t,sol(:, :, 1))
title('Exact solution')
xlabel('x')
ylabel('t')
hold on
shading interp
colorbar
hold off


