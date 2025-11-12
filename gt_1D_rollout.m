clear
close all
clc

%% Global variables
global T mu k t_f f_min x_f_1 x_f_2
T = 1;
mu = 1;
k = 1;
t_f = 10;
f_min = -3;
x_f_1 = 0.2;
x_f_2 = 0.8;

x = linspace(0,1,100);
t = linspace(0,t_f,t_f*100);


%% CREATE PDE for a vibrating string

cpu_time_start = cputime;

function f = stringforce(x, t)
persistent v_branch_data v_branch_x
if isempty(v_branch_data)
    data = readmatrix('/Users/guglielmocappellini/Desktop/research/code/pinns-wave/wave-gnn/1_gcn_string/v_branch.csv');
    v_branch_x = linspace(0, 1, length(data));
    v_branch_data = data;
end
space = interp1(v_branch_x, v_branch_data, x, 'spline', 0);
f = exp(-((t - 2).^2) / (2 * 0.5^2)) .* space;
end

function [c,f,s] = stringpde(x, t, u, dudx)
    global k T mu
    c = [1; 1];
    F = 10*stringforce(x, t);
    s = [u(2); F - k*u(2)];
    f = [0; (T/mu)*dudx(1)];
end

function u_0 = stringic(x)
    u_0 = [0;0];
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
        output_data = [output_data; x(j), t(i), stringforce(x(j), t(i)), sol(i, j, 1)];
    end
end
writematrix(output_data, '/Users/guglielmocappellini/Desktop/research/code/pinns-wave/wave-gnn/1_gcn_string/gt_wave1D_rollout.csv');

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


