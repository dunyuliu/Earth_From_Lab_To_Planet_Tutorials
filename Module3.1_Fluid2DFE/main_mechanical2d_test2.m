%% This is the main script of the fluid2DFE.
%  The code is to calculate the velocity and pressure fields surrounding a
%  sinking sphere. 
%  It is derived from MILANMIN 1.0, the information of which is
%  provided below. MILANMIN 1.0 is distributed under GNU 2.0 License. 
%  It also relies on mesh generator triangle to build the finite element
%  mesh. Triangle is developed by Jonathan Richard Shewchuk and it could be 
%  accessed via https://www.cs.cmu.edu/~quake/triangle.html. 
%%
% On MILAMIN. 
%   Part of MILAMIN: MATLAB-based FEM solver for large problems, Version 1.0
%   Copyright (C) 2007, M. Dabrowski, M. Krotkiewski, D.W. Schmid
%   University of Oslo, Physics of Geological Processes
%   http://milamin.org
%   MILANMIN 1.0 is distributed under GNU 2.0. 
%   Please check the License file for terms of use.
% On Triangle.
%   To credit Triangle, please cite 
%   1) Jonathan Richard Shewchuk, Triangle: Engineering a 2D Quality Mesh Generator and Delaunay Triangulator, in ``Applied Computational Geometry: Towards Geometric Engineering'' (Ming C. Lin and Dinesh Manocha, editors), volume 1148 of Lecture Notes in Computer Science, pages 203-222, Springer-Verlag, Berlin, May 1996.
%   and 2) Jonathan Richard Shewchuk, Delaunay Refinement Algorithms for Triangular Mesh Generation, Computational Geometry: Theory and Applications 22(1-3):21-74, May 2002. 

% CLEARING AND INITIALIZATION
%==========================================================================

%CLEAR ENVIRONMENT, BUT NOT BREAKPOINTS
%clc;
%clear variables;

%
% boundary conditions
%
bcmode=6; % 1: pure shear 2: no slip 3: free slip 
          % 4: periodic in x 5: periodic and top prescribed motion
          % 6: driven top, else free slip

%==========================================================================
% Physical parameters
%==========================================================================
parameters.Eta           = [1 ;   100];                %Viscosity
parameters.Rho         =   [  0;   0];                %Density
parameters.Gy           = [ 0 -1];                %Gravity in y direction
parameters.eta_lower_mantle = 1;
%==========================================================================
% MESH GENERATION:
%==========================================================================
fprintf(1, 'PREPROCESSING:      '); tic

second_order=1; % we NEED second order (six node, geometrically) triangles
nip=6;                                  % number of integration points


mesh_par.no_pts_incl = 30;% points on inclusion
mesh_par.radius =     0.1; %
mesh_par.type   =       1; %
mesh_par.ellipticity = 0.0;
mesh_par.qangle = 25;
%mesh_par.area_glob = 0.0001;
mesh_par.area_glob = 0.0001;

%
% if periodic BCs in x are desired, need to make sure we have the same
% number of nodes on both left and right side
%
%
if(bcmode == 4 || bcmode == 5) % make sure sides of box are controlled spacing
   mesh_par.fix_box = 1;
else
   mesh_par.fix_box = 0;
end

conv_to_int = 0;                        % convert int arrays to actual ints


[GCOORD, ELEM2NODE, Point_id, Phases] = ...
        generate_mesh(mesh_par,second_order,conv_to_int);

nnod    = size(GCOORD,2);
nel     = size(ELEM2NODE,2);

if(second_order==1)
   %add 7th node
   ELEM2NODE(7,:)  = nnod+1:nnod+nel;
   GCOORD          = [GCOORD, [...
       mean(reshape(GCOORD(1, ELEM2NODE(1:3,:)), 3, nel));...
       mean(reshape(GCOORD(2, ELEM2NODE(1:3,:)), 3, nel))]];

   nnod    = size(GCOORD,2);
end



fprintf(1, [num2str(toc,'%8.6f'),'\n']);

%
% mesh boundaries
%
xmin        = min(GCOORD(1,:));
xmax       = max(GCOORD(1,:));
ymin        = min(GCOORD(2,:));
ymax        = max(GCOORD(2,:));
       
periodic_xbc = 0; % default is free slip, or set by bcmode
switch bcmode
   case 1
       %==========================================================================
       % BOUNDARY CONDITION: PURE SHEAR
       %==========================================================================

      % old way of specifying boundary conditions
      %
      %         Bc_ind  = find(Point_id==1);
      %         Bc_val  = [GCOORD(1,Bc_ind)  -GCOORD(2,Bc_ind)];
      %         Bc_ind  = [2*(Bc_ind-1)+1       2*(Bc_ind-1)+2];
      %
      %
      % new way
      %

       % Define BC's, by specifying nodes, dof & values
       leftright = [ find(GCOORD(1,:)==xmin) find(GCOORD(1,:)==xmax) ];
       topbottom = [ find(GCOORD(2,:)==ymin) find(GCOORD(2,:)==ymax) ];
       % set nodes for which BCs apply
       BC_nodes    = [leftright,leftright,topbottom,topbottom];
       % set which DOF is constrained 1=x 2=y
       BC_dof      = [ones(size(leftright))*1    ones(size(leftright))*2 ...
                      ones(size(topbottom))*1    ones(size(topbottom))*2];
       % values to be prescribed
       BC_val      = [GCOORD(2,leftright) zeros(size(leftright)) ...
           GCOORD(2,topbottom)  zeros(size(topbottom))     ];

   case 2
       % no slip
       node_number  = find(Point_id==1);
       % Define BC's, by specifying nodes, dof & values
       BC_nodes    = [node_number,                 node_number];
       BC_dof      = [ones(size(node_number))*1    ones(size(node_number))*2];
       BC_val      = [ zeros(size(node_number))    zeros(size(node_number)) ];


   case 3
       % free slip
       % left and right constrained in x
       % top and bottom constrained in y
       
       leftright = [ find(GCOORD(1,:)==xmin) find(GCOORD(1,:)==xmax) ];
       topbottom = [ find(GCOORD(2,:)==ymin) find(GCOORD(2,:)==ymax) ];

       % Define BC's, by specifying nodes, dof & values
       BC_nodes    = [leftright,                 topbottom];
       BC_dof      = [ones(size(leftright))*1    ones(size(topbottom))*2];
       BC_val      = [ zeros(size(leftright))    zeros(size(topbottom)) ];


   case 4
       % free slip top, periodic in x
       % top and bottom constrained in y
       %
       % pick centered nodes on sides
       %
       
   
       
       % Define free slip top and bottom BC's
       topbottom = [ find(GCOORD(2,:)==ymin) find(GCOORD(2,:)==ymax ) ];
       BC_nodes    = [topbottom];
       BC_dof      = [ones(size(topbottom))*2];
       BC_val      = [zeros(size(topbottom)) ];

       periodic_xbc = 1;

    case 5
        %
        % part of top is pushed, periodic BCs
        %
        %
        x_left = 0.5;x_push = 0.0005;
        % bottom free slip
        bottom = [ find(GCOORD(2,:)==ymin) ];
        BC_nodes    = [bottom];
        BC_dof      = [ones(size(bottom))*2];
        BC_val      = [zeros(size(bottom)) ];

        % top pushed
        toppush = [ find(GCOORD(2,:)==ymax & GCOORD(1,:) <= x_left) ];
        BC_nodes    = [BC_nodes, toppush , toppush];
        BC_dof      = [BC_dof ones(size(toppush))*1 ones(size(toppush))*2];
        BC_val      = [BC_val x_push * ones(size(toppush)) zeros(size(toppush))];
        
        % part of top that is fixed
        topfixed = [ find(abs(GCOORD(2,:)-ymax) < 1e-5 & GCOORD(1,:) > x_left) ];
        BC_nodes    = [BC_nodes, topfixed , topfixed];
        BC_dof      = [BC_dof ones(size(topfixed))*1 ones(size(topfixed))*2];
        BC_val      = [BC_val zeros(size(topfixed)) zeros(size(topfixed))];

        periodic_xbc = 1;
    
    
    case 6
        % Driven lid 
        % free slip everywhere, except top which has prescribed vel.
        

        leftright = [ find(GCOORD(1,:)==xmin) find(GCOORD(1,:)==xmax) ];
        bottom = [ find(GCOORD(2,:)==ymin) ];
        
        % Define BC's, by specifying nodes, dof & values
        BC_nodes    = [leftright,                 bottom];
        BC_dof      = [ones(size(leftright))*1    ones(size(bottom))*2];
        BC_val      = [ zeros(size(leftright))    zeros(size(bottom)) ];
        
       
        % top pushed
        
        toppush = [ find(GCOORD(2,:)==ymax) ];
        x_push = 0.0005 * sin(GCOORD(1,toppush)*pi);
        
        BC_nodes    = [BC_nodes, toppush , toppush];
        BC_dof      = [BC_dof ones(size(toppush))*1 ones(size(toppush))*2];
        BC_val      = [BC_val x_push  zeros(size(toppush))];

        
       

end

if(periodic_xbc == 1)
    %
    % periodic BCs?
    %
    %------------------------------------------------------------------
    % Periodic BC
    %
    % center of box, without top and bottom
    ylevels = sort(GCOORD(2,(find(GCOORD(1,:)==xmin))));
    %ylevels = ylevels(find(ylevels > ymin & ylevels < ymax));
    left=[];right =[];
    for y=ylevels
        left   = [ left   find(GCOORD(1,:)==xmin & GCOORD(2,:) ==y) ];
        right  = [ right  find(GCOORD(1,:)==xmax & GCOORD(2,:) ==y) ];
    end
    if(length(left) ~= length(right))
        error('need same node number on left and right side for periodic BC')
    end
    if(GCOORD(2,left)~=GCOORD(2,right))
        error('coord mismatch, need to have same node height for periodic BC')
    end
    % Define nodes, which are periodic (this assumes that the mesh is indeed perfectly periodic,
    % meaning that the z-coordinates of the points on the right side of the box matches those of the
    % points on the left side of the box!):
    PERIOD(1,:) = [left,                left];                % keep nodes on left boundary
    PERIOD(2,:) = [right,               right];               % eliminate nodes on right boundary
    PERIOD(3,:) = [ones(size(left))*1,  ones(size(right))*2]; % the dof's that are periodic
else
    PERIOD = [];
end

plot_mesh=1;
switch plot_mesh
   case 1
       figure(2);clf(2);
       trisurf(ELEM2NODE(1:3,:)', GCOORD(1,:), GCOORD(2,:), ...
           zeros(size(GCOORD(1,:))),'EdgeColor','k','FaceColor','w');
       hold on;
       % edge nodes
       bnodes=find(Point_id==1);
       plot(GCOORD(1,bnodes),GCOORD(2,bnodes),'ro');
       % hole nodes
       bnodes=find(Point_id==100);
       plot(GCOORD(1,bnodes),GCOORD(2,bnodes),'g^');
       

       % special elements
       cstring='rgbcmyb';
       for i = 2:max(Phases)
           sele=find(Phases == i);
           trisurf(ELEM2NODE(1:3,sele)', GCOORD(1,:), GCOORD(2,:), ...
               zeros(size(GCOORD(1,:))),'FaceColor',cstring(i-1));
       end
       view(2);  axis image
end

fprintf(1, [num2str(toc,'%8.6f')]);
fprintf(1, ['\n Number of nodes:   ', num2str(nnod)]);
fprintf(1, ['\n Number of elems:   ', num2str(nel),'\n']);


%==========================================================================
% SOLVER
%==========================================================================

[V PRESSURE] = mechanical2d_std(ELEM2NODE, Phases, GCOORD, parameters, ...,
       BC_nodes, BC_dof, BC_val, nip,PERIOD);
% properly compute pressure (not really needed, if used, also adjust
% mechanical_std.m
%PRESSURE = evaluate_pressure_at_nodes(PRESSURE, ELEM2NODE, GCOORD);
%
   
%==========================================================================
% POSTPROCESSING
%==========================================================================
fprintf(1, 'POSTPROCESSING:     '); tic



hh = figure(1);
clf(1)

% choose background
pmode=1;
if(pmode==1) % pressure - mean
    mean_pressure = element_avg(PRESSURE,GCOORD,ELEM2NODE);
    trisurf(reshape(1:3*nel,3, nel)', GCOORD(1,ELEM2NODE(1:3,:)), ...,
       GCOORD(2,ELEM2NODE(1:3,:)), ...
       zeros(size(GCOORD(1,ELEM2NODE(1:3,:)))), PRESSURE-mean_pressure);
   title('Pressure');
   %caxis([-1e2 1e2])
   view(2);
   shading interp;
else % absolute  velocity
   [xi,yi] = meshgrid(0:.0025:1,0:.0025:1);
   zi = griddata(GCOORD(1,:)',GCOORD(2,:)', ...
        sqrt(V(1:2:nnod*2-1).^2+V(2:2:nnod*2).^2),...
       xi,yi,'linear');
   pcolor(xi,yi,abs(zi));
   title('|V|');

   shading interp
end
%
% plot velocity vectors on top
%
colorbar; axis image; %axis off
hold on
velmode=2;
if(velmode==1)				% all velocities
  quiver(GCOORD(1,:)',GCOORD(2,:)', ...
      V(1:2:nnod*2-1),V(2:2:nnod*2),'k');
else
  [xi,yi] = meshgrid(0:.05:1,0:.05:1);
  
  preferred=1;
  if(preferred == 1)
    % get triangular interpolation structure
    Vx = TriScatteredInterp(GCOORD(1,:)',GCOORD(2,:)', V(1:2:nnod*2-1));
    Vy = TriScatteredInterp(GCOORD(1,:)',GCOORD(2,:)', V(2:2:nnod*2));
    vx = Vx(xi,yi);
    vy = Vy(xi,yi);
  else
    vx = griddata(GCOORD(1,:)',GCOORD(2,:)', V(1:2:nnod*2-1), ...
	xi,yi,'linear');
    vy = griddata(GCOORD(1,:)',GCOORD(2,:)', V(2:2:nnod*2), ...
	xi,yi,'linear'); 
  end
  
  quiver(xi,yi,vx,vy);
end

fprintf(1, [num2str(toc,'%8.6f'),'\n']);
fprintf(1, ['\n']);
