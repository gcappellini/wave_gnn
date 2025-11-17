clear all
close all
clc

t_f = 2;

tlist = linspace(0,t_f,100);

%% Simulating PINNs with numerical methods

cpu_time_start = cputime;

wave_speed = 1;
d = 1;
a = 0;
m = 1;
k = 1;
u0_coeff = 1.0;
v0_coeff = 0.0;

source_width=0.3;
source_amp = 7.5;
source_center_x = 0.17;
source_center_y = 0.17;

numberOfPDE = 1;
model = createpde(numberOfPDE);
R1 = [3,4,0,1,1,0,0,0,1,1]';
g = decsg(R1);
geometryFromEdges(model,g);
pdegplot(model,"EdgeLabels","on"); 
ylim([-0.1 1.1]);
axis equal
title("Geometry With Edge Labels Displayed")
xlabel("x")
ylabel("y")


specifyCoefficients(model,m=m,d=0,c=wave_speed^2 ,a=a,f=@force);


function fcoeff=force(location, state)
    global source_width source_amp source_center_x source_center_y
    % t1 = 0.3*10;
    % t2 = 0.6*10;
    % %f1 = -3*exp(-400.*((location.x-0.7).^2).*((location.y-0.7).^2)).*exp(-(state.time - t1)^2/(2*0.5^2));
    % %f2 = -3*exp(-400.*((location.x-0.3).^2).*((location.y-0.3).^2)).*exp(-(state.time - t2)^2/(2*0.5^2));
    % f1 = -3*exp(-400.*(((location.x-0.7).^2)+((location.y-0.7).^2))).*exp(-(state.time - t1)^2/(2*0.5^2));
    % f2 = -3*exp(-400.*(((location.x-0.2).^2)+((location.y-0.5).^2))).*exp(-(state.time - t2)^2/(2*0.5^2));
    % fcoeff = f1+f2;
    fcoeff = zeros(1, length(location.x));  % Zero source
    % fcoeff = source_amp * exp(-((location.x - source_center_x)/source_width).^2 - ((location.y - source_center_y)/source_width).^2);
end

applyBoundaryCondition(model,"dirichlet","Edge",[1,2,3,4],"u",0);
%applyBoundaryCondition(model,"neumann","Edge",[2,4],"g",0);
%applyBoundaryCondition(model,"dirichlet","Edge",[1,3],"u",0);
%applyBoundaryCondition(model,"neumann","Edge",[1,2,3,4],"g",0);

generateMesh(model);
timeStruct = struct('time', 0);
results = assembleFEMatrices(model, timeStruct);
specifyCoefficients(model,m=m,d=d*results.M,c=wave_speed^2,a=a,f=@force);
figure
pdemesh(model);
ylim([-0.1 1.1]);
axis equal
xlabel x
ylabel y


mesh = model.Mesh;
nodes = mesh.Nodes;
elements = mesh.Elements;

% Export nodes (coordinates)
writematrix(nodes', '/Users/guglielmocappellini/Desktop/research/code/pinns-wave/wave-gnn/1_gcn_string/logs_multibranch_wave2D/nodes.csv');

% Export elements (triangle connectivity, 0-indexed for Python)
writematrix(elements'-1, '/Users/guglielmocappellini/Desktop/research/code/pinns-wave/wave-gnn/1_gcn_string/logs_multibranch_wave2D/elements.csv');

% Export boundary nodes (from boundary edges 1,2,3,4)
boundary_nodes = [];
for edge_id = 1:4
    edge_nodes = findNodes(mesh, 'region', 'Edge', edge_id);
    boundary_nodes = [boundary_nodes; edge_nodes];
end
boundary_nodes = unique(boundary_nodes) - 1; % 0-indexed for Python
writematrix(boundary_nodes, '/Users/guglielmocappellini/Desktop/research/code/pinns-wave/wave-gnn/1_gcn_string/logs_multibranch_wave2D/boundary_nodes.csv');

u0 = @(location) u0_coeff*sin(pi*location.x).*sin(pi*location.y);% v0_coeff*sin(pi*location.x).*sin(pi*location.y)];% atan(cos(pi/2*location.x));
ut0 = @(location) v0_coeff*sin(pi*location.x).*sin(pi*location.y);% 3*sin(pi*location.x).*exp(sin(pi/2*location.y));

setInitialConditions(model,u0, ut0);


model.SolverOptions.ReportStatistics ='on';
result = solvepde(model,tlist);

cpu_time_end = cputime-cpu_time_start

u = result.NodalSolution;
umax = max(max(u));
umin = min(min(u));

% % Compute velocity by numerical differentiation of displacement
% dt = tlist(2) - tlist(1);  % Time step
% v = zeros(size(u));
% v(:, 1) = (u(:, 2) - u(:, 1)) / dt;  % Forward difference for first time step
% for i = 2:length(tlist)-1
%     v(:, i) = (u(:, i+1) - u(:, i-1)) / (2*dt);  % Central difference
% end
% v(:, end) = (u(:, end) - u(:, end-1)) / dt;  % Backward difference for last time step

%% Save solution to CSV files
% Combine x, y, t, force, displacement, and velocity into one file
% Format: [x, y, t, f, u, v]
output_data = [];

% Get mesh coordinates
x_nodes = mesh.Nodes(1, :)';  % x-coordinates of all nodes
y_nodes = mesh.Nodes(2, :)';  % y-coordinates of all nodes
n_nodes = length(x_nodes);

% Loop over time steps
for i = 1:length(tlist)
    t_current = tlist(i);
    
    % Loop over spatial nodes
    for j = 1:n_nodes
        x_j = x_nodes(j);
        y_j = y_nodes(j);
        
        % Evaluate force at this location and time
        loc.x = x_j;
        loc.y = y_j;
        state.time = t_current;
        f_val = force(loc, state);
        
        % Get displacement and velocity at this node and time
        u_val = result.NodalSolution(j, i);
        v_val = 0;
        
        % Append to output: [x, y, t, f, u, v]
        output_data = [output_data; x_j, y_j, t_current, f_val, u_val, v_val];
    end
end

writematrix(output_data, '/Users/guglielmocappellini/Desktop/research/code/pinns-wave/wave-gnn/1_gcn_string/data/gt_wave2D_nosource.csv');


%% PLOT MATLAB SOLUTION

u = result.NodalSolution;  % size = [1537, 100]

% Compute scalar bounds
umin = min(u(:));
umax = max(u(:));

% Animate
figure;
for i = 1:length(tlist)
    pdeplot(model, "XYData", u(:, i), "ZData", u(:, i), ...
                    "ZStyle", "continuous", "Mesh", "off");
    axis([0 1 0 1 umin umax]);
    xlabel("x"); ylabel("y"); zlabel("u");
    title(sprintf("Numerical - t = %.2f", tlist(i)));
    drawnow;
    M(i) = getframe(gcf);
end

% To play the animation
movie(M);
