function [Pressure_nd] = evaluate_pressure_at_nodes(PRESSURE, ELEM2NODE, GCOORD)
%
% evaluate the pressure properly
%
nel         = size(ELEM2NODE,2); % number elements
ndim        = 2; % number of dimensions
nnodel      = size(ELEM2NODE,1); % number of nodes per element

%
nip = 3; % at three corner locations
IP_X = zeros(3,ndim);
IP_X(1,1) = 0;IP_X(1,2) = 0;
IP_X(2,1) = 1;IP_X(2,2) = 0;
IP_X(3,1) = 0;IP_X(3,2) = 1;
% get shape functions and shape function derivatives, evaluated at
% the element-local coordinates for the Gauss points and all of the nodes of the element
[   N, dNdu]    = shp_deriv_triangle(IP_X, nnodel);

np          = 3; % nodes that enter into the pressure formulation
% (linear, discontinuous)
P           = ones(np);
Pb          = ones(np,1);
Pressure_nd = zeros(nel,nip);

PressureShapeFunction = 'Local';
for iel = 1:nel
 %==============================================================
 % ii) FETCH DATA OF ELEMENT
 %==============================================================
 ECOORD_X  = GCOORD(:,ELEM2NODE(:,iel)); % the coords remain to be obtained from the elem2node mapping
 P(2:3,:) = ECOORD_X(:,1:3);
 for ip=1:nip % loop over integration points
   Ni       =       N{ip}; % shape function for each of the lement nodes at integration point
   switch PressureShapeFunction
       case 'Local'
           % Linear, discontinuous, pressure shape function
           %  This is DIFFERENT than the approach taken in the
           %  published MILAMIN paper.
           Pi = [1; IP_X(ip,1); IP_X(ip,2)];           % linear in local coordinates
       case 'Global'
           ECOORD_x = ECOORD_X(1,:)';
           ECOORD_y = ECOORD_X(2,:)';
           GIP_x    = Ni'*ECOORD_x;
           GIP_y    = Ni'*ECOORD_y;
           Pi       = [1; GIP_x; GIP_y];                % linear in global coordinates
       case 'Original'
           Pb(2:3)     = ECOORD_X*Ni; % center of element
           Pi          =   P\Pb; % pressure shape function
   end
   Pressure_nd(iel,ip) = Pi'*PRESSURE(:,iel);
 end
end
Pressure_nd = Pressure_nd'; % for consistency with PRESSURE
