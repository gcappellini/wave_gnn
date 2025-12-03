clear all
close all
clc

t_f = 1;

tlist = linspace(0,t_f,100);

%% Simulating PINNs with numerical methods
global source_width source_amp source_center_x source_center_y

cpu_time_start = cputime;

wave_speed = 1;
d = 1;
a = 0;
m = 1;

% 2D coefficient arrays for IC (n=2): independent x and y modes
% u(x,y) = 0.5*sin(π*x)*sin(π*y) + 0.3*sin(π*x)*sin(2π*y) + 0.2*sin(2π*x)*sin(π*y) + 0.1*sin(2π*x)*sin(2π*y)
% u0_coeffs = [0.5, 0.3; 0.2, 0.1];
u0_coeffs = [0.0, 0.0; 0.0, 0.0];

% v(x,y) similar structure
% v0_coeffs = [0.2, 0.1; 0.05, 0.025];
v0_coeffs = [0.0, 0.0; 0.0, 0.0];

source_center_x = 0.35;
source_center_y = 0.65;
source_width=0.3;
source_amp = 15.0;

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
    % f1 = -3*exp(-400.*(((location.x-0.7).^2)+((location.y-0.7).^2))).*exp(-(state.time - t1)^2/(2*0.5^2));
    % f2 = -3*exp(-400.*(((location.x-0.2).^2)+((location.y-0.5).^2))).*exp(-(state.time - t2)^2/(2*0.5^2));
    % fcoeff = f1+f2;


    fcoeff = source_amp*exp(-(((location.x-source_center_x).^2)+((location.y-source_center_y).^2))/source_width^2);
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

u0 = @(location) (u0_coeffs(1,1)*sin(pi*location.x).*sin(pi*location.y) + ...
                   u0_coeffs(1,2)*sin(pi*location.x).*sin(2*pi*location.y) + ...
                   u0_coeffs(2,1)*sin(2*pi*location.x).*sin(pi*location.y) + ...
                   u0_coeffs(2,2)*sin(2*pi*location.x).*sin(2*pi*location.y));

ut0 = @(location) (v0_coeffs(1,1)*sin(pi*location.x).*sin(pi*location.y) + ...
                   v0_coeffs(1,2)*sin(pi*location.x).*sin(2*pi*location.y) + ...
                   v0_coeffs(2,1)*sin(2*pi*location.x).*sin(pi*location.y) + ...
                   v0_coeffs(2,2)*sin(2*pi*location.x).*sin(2*pi*location.y));


setInitialConditions(model,u0, ut0);


model.SolverOptions.ReportStatistics ='on';
result = solvepde(model,tlist);

cpu_time_end = cputime-cpu_time_start

u = result.NodalSolution;
umax = max(max(u));
umin = min(min(u));

%% Extract displacement u
u_all = result.NodalSolution;   % nNodes x nTime
[nNodes, nTime] = size(u_all);
dt = tlist(2) - tlist(1);

%% Compute velocity v = du/dt using 4th-order central differences for accuracy
v_all = zeros(size(u_all));

% Forward difference for first two points
v_all(:,1) = (-25*u_all(:,1) + 48*u_all(:,2) - 36*u_all(:,3) + 16*u_all(:,4) - 3*u_all(:,5))/(12*dt);
v_all(:,2) = (-3*u_all(:,1) - 10*u_all(:,2) + 18*u_all(:,3) - 6*u_all(:,4) + u_all(:,5))/(12*dt);

% Central differences for interior points (4th-order)
for k = 3:nTime-2
    v_all(:,k) = (u_all(:,k-2) - 8*u_all(:,k-1) + 8*u_all(:,k+1) - u_all(:,k+2))/(12*dt);
end

% Backward difference for last two points
v_all(:,nTime-1) = (3*u_all(:,nTime) + 10*u_all(:,nTime-1) - 18*u_all(:,nTime-2) + 6*u_all(:,nTime-3) - u_all(:,nTime-4))/(12*dt);
v_all(:,nTime) = (25*u_all(:,nTime) - 48*u_all(:,nTime-1) + 36*u_all(:,nTime-2) - 16*u_all(:,nTime-3) + 3*u_all(:,nTime-4))/(12*dt);

%% Animate displacement u and velocity v side by side
umin = min(u_all(:)); umax = max(u_all(:));
vmin = min(v_all(:)); vmax = max(v_all(:));

figure;
for ti = 1:nTime
    subplot(1,2,1);
    pdeplot(model,'XYData',u_all(:,ti),'ZData',u_all(:,ti),'Mesh','off','ZStyle','continuous');
    axis([0 1 0 1 umin umax]);
    title(sprintf('Displacement u at t=%.3f', tlist(ti)));
    xlabel('x'); ylabel('y'); zlabel('u');
    
    subplot(1,2,2);
    pdeplot(model,'XYData',v_all(:,ti),'ZData',v_all(:,ti),'Mesh','off','ZStyle','continuous');
    axis([0 1 0 1 vmin vmax]);
    title(sprintf('Velocity v at t=%.3f', tlist(ti)));
    xlabel('x'); ylabel('y'); zlabel('v');
    
    drawnow;
    M(ti) = getframe(gcf);
end

% Play movie
movie(M);


%% Save solution to CSV files (vectorized)
% Combine x, y, t, force, displacement u, and velocity v into one file

% Mesh coordinates
x_nodes = mesh.Nodes(1,:)';  % nNodes x 1
y_nodes = mesh.Nodes(2,:)';  % nNodes x 1
n_nodes = length(x_nodes);
n_time = length(tlist);

% Create grids for node indices and time indices
[node_idx_grid, time_idx_grid] = ndgrid(1:n_nodes, 1:n_time);

% Flatten grids
x_flat = x_nodes(node_idx_grid(:));
y_flat = y_nodes(node_idx_grid(:));
t_flat = tlist(time_idx_grid(:))';

% Flatten displacement and velocity
u_flat = u_all(:);
v_flat = v_all(:);

% Compute force at all points
r2 = (x_flat - source_center_x).^2 + (y_flat - source_center_y).^2;
f_flat = source_amp * exp(-r2 / source_width^2);

% Combine into single matrix
output_data = [x_flat, y_flat, t_flat, f_flat, u_flat, v_flat];

% Write to CSV
writematrix(output_data, '/Users/guglielmocappellini/Desktop/research/code/pinns-wave/wave-gnn/1_gcn_string/data/gt_wave2D_with_source.csv');

fprintf('CSV export complete: %d rows x %d columns\n', size(output_data,1), size(output_data,2));