clear
close all
clc

%% Global variables
global a b T mu k t_f source_amp_global source_center_global source_width_global
T = 1;
mu = 1;
k = 1;
t_f = 2;
source_amp = 7.5;    % Gaussian source amplitude
T_max = 2.0;         % Final time
a = 0.5;
b = 2.0;

% Gaussian source parameters
source_center = 0.17;
source_width = 0.3;

source_amp_global = source_amp;
source_center_global = source_center;
source_width_global = source_width;

x = linspace(0,1,100);
t = linspace(0,t_f,50);


%% CREATE PDE for a vibrating string

cpu_time_start = cputime;

function f = stringforce(x)
    global source_amp_global source_center_global source_width_global
    f = source_amp_global * exp(-((x - source_center_global)/source_width_global)^2);
% f = 0;
end

function [c,f,s] = stringpde(x, t, u, dudx)
    global k T mu
    c = [1; 1];
    F = stringforce(x);
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
% Combine x, t, and solution into one file with three columns
output_data = [];
for i = 1:length(t)
    for j = 1:length(x)
        output_data = [output_data; x(j), t(i), stringforce(x(j)), sol(i, j, 1)];
    end
end
writematrix(output_data, '/Users/guglielmocappellini/Desktop/research/code/pinns-wave/wave-gnn/1_gcn_string/gt_wave1D_withsource.csv');

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


