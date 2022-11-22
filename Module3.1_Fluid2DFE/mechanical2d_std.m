function [Vel Pressure] = mechanical2d_std(ELEM2NODE, Phases, GCOORD, ...
   parameters, BC_nodes, BC_dof, BC_val, nip,PERIOD)
%
% simplified version of incompressible Stokes fluid solver
% including periodic boundary conditions
%


%MECHANICAL2D Two dimensional finite element mechanical problem solver of MILAMIN


%   Part of MILAMIN: MATLAB-based FEM solver for large problems, Version 1.0
%   Copyright (C) 2007, M. Dabrowski, M. Krotkiewski, D.W. Schmid
%   University of Oslo, Physics of Geological Processes
%   http://milamin.org
%   See License file for terms of use.

%==========================================================================
% MODEL INFO
%==========================================================================
nnod        = size(GCOORD,2); % number of nodes
nel         = size(ELEM2NODE,2); % number elements

%==========================================================================
% CONSTANTS
%==========================================================================
ndim        = 2; % number of dimensions
nnodel      = size(ELEM2NODE,1); % number of nodes per element
nedof       = nnodel*ndim; % number of elemental degrees of freedom
sdof        = 2*nnod; % global nodal number of degrees of freedom
%
np          = 3; % nodes that enter into the pressure formulation
% (linear, discontinuous)


%--------------------------------------------------------------------------
% additional global node numbering array
% 
%
LOC2GLOB        = [1:sdof];
if ~isempty(PERIOD)
    % periodic BC addition
    tic; fprintf(1, 'PERIODIZE DOFs:         ');
    Keep            = ndim*(PERIOD(1,:)-1) + PERIOD(3,:); % select the DOFs to keep
    Elim            = ndim*(PERIOD(2,:)-1) + PERIOD(3,:); % select the DOFs to eliminate
    Dummy           = zeros(size(LOC2GLOB));
    Dummy(Elim)     = 1;
    LOC2GLOB        = LOC2GLOB-cumsum(Dummy); % reduce the entries in loc2glob by the total 
                                              % of eliminated DOFs
    LOC2GLOB(Elim)  = LOC2GLOB(Keep);         % have all eliminated DOFs refer to the
                                              % actual DOFs
    fprintf(1, [num2str(toc),'\n']);
end
LOC2GLOB        = reshape(LOC2GLOB, [ndim, nnod]);% make the loc2glob vector a (ndim,nnod) matrix
LOC2GLOB        = int32(LOC2GLOB); % convert to integer to save space
sdof            = max(LOC2GLOB(:));   %REDUCE GLOBAL DOF COUNT

%
% Compute the global equation numbers from the boundary conditions
%
Bc_ind  = zeros(1,size(BC_nodes,2));
for i = 1:length(BC_nodes)
   bc_nod     = BC_nodes(i);
   Bc_ind(i)  = LOC2GLOB(BC_dof(i), bc_nod); % the index is now in the new DOF system
end

%
%
%--------------------------------------------------------------------------


% this is the material matrix
DEV   = [ 4/3 -2/3 0;...
        -2/3  4/3 0;...
           0    0 1];

%
%
%
PF = 1e3*max(parameters.Eta);  % fake compressibility



%==========================================================================
% i) PREPARE INTEGRATION POINTS & DERIVATIVES wrt LOCAL COORDINATES
%==========================================================================
[IP_X, IP_w]    = ip_triangle(nip);
% get shape functions and shape function derivatives, evaluated at
% the element-local coordinates for the Gauss points and all of the nodes of the element
[   N, dNdu]    = shp_deriv_triangle(IP_X, nnodel);

%==========================================================================
% DECLARE VARIABLES (ALLOCATE MEMORY)
%==========================================================================
A_all       = zeros(nedof*(nedof+1)/2,nel); % upper triangle for K_bar
Q_all       = zeros(nedof*np,nel); % integral over divergence from flow shape function times pressure shape function
invM_all    = zeros(np*np,nel); % inverse of M, the pressure shape function inner product
Rhs_all     = zeros(nedof,nel); % solution
%==========================================================================
% INDICES EXTRACTING LOWER PART
%==========================================================================
indx_l = tril(ones(nedof)); indx_l = indx_l(:); indx_l = indx_l==1;


EG        = parameters.Gy'; % element gravity


%==================================================================
% DECLARE VARIABLES (ALLOCATE MEMORY)
%==================================================================
A_elem      = zeros(nedof,nedof);
Q_elem      = zeros(nedof,np);
M_elem      = zeros(np,np);
Rhs_elem    = zeros(ndim,nnodel);

B           = zeros(nedof,ndim*(ndim+1)/2);
P           = ones(np);
Pb          = ones(np,1);


%PressureShapeFunction = {'Original','Local','Global'};
%PressureShapeFunction = PressureShapeFunction{2};


%==================================================================
% i) ELEMENT LOOP - MATRIX COMPUTATION
%==================================================================
fprintf(1, 'MATRIX COMPUTATION: '); tic;
for iel = 1:nel
 %==============================================================
 % ii) FETCH DATA OF ELEMENT
 %==============================================================
 ECOORD_X  = GCOORD(:,ELEM2NODE(:,iel)); % the coords remain to be obtained from the elem2node mapping
 EEta      = parameters.Eta(Phases(iel));% element viscosity
 %
 % higher lower mantle viscosity? 
 %
 if(mean(ECOORD_X(2,1:3)) < 0.5)
     EEta = EEta * parameters.eta_lower_mantle;
 end
 ERho      = parameters.Rho(Phases(iel)); % element density

 %==============================================================
 % iii) INTEGRATION LOOP
 %==============================================================
 A_elem(:) = 0;
 Q_elem(:) = 0;
 M_elem(:) = 0;
 Rhs_elem(:) = 0;

 P(2:3,:) = ECOORD_X(:,1:3);% node coordinates at pressure points
 for ip=1:nip % loop over integration points
   %==========================================================
   % iv) LOAD SHAPE FUNCTIONS DERIVATIVES FOR INTEGRATION POINT
   %==========================================================
   Ni       =       N{ip}; % shape function for each of the lement nodes at integration point
   dNdui       =    dNdu{ip}; % derivatives

   %switch PressureShapeFunction
   %    case 'Local'
           % Linear, discontinuous, pressure shape function
           %  This is DIFFERENT than the approach taken in the
           %  published MILAMIN paper.
    %       Pi = [1; IP_X(ip,1); IP_X(ip,2)];           % linear in local coordinates
    %   case 'Global'
          % ECOORD_x = ECOORD_X(1,:)';
          % ECOORD_y = ECOORD_X(2,:)';
          % GIP_x    = Ni'*ECOORD_x;
          % GIP_y    = Ni'*ECOORD_y;
          % %Pi       = [1; GIP_x; GIP_y];                % linear in global coordinates
          %  Pb = [1; GIP_x; GIP_y];
          %  Pi          =   P\Pb; % pressure shape function
           
          %   case 'Original'
          Pb(2:3)     = ECOORD_X*Ni; % center of element
          Pi          =   P\Pb; % pressure shape function
   %end
   
   %==========================================================mechanical2d_std
   % v) CALCULATE JACOBIAN, ITS DETERMINANT AND INVERSE
   %==========================================================
   J           = ECOORD_X*dNdui;
   %detJ        = det(J);
   %invJ        = inv(J);
   detJ = det2D(J);
   invJ = inv2D(J,detJ);
   
   %==========================================================
   % vi) DERIVATIVES wrt GLOBAL COORDINATES
   %==========================================================
   dNdX        = dNdui*invJ;

   %==========================================================
   % vii) NUMERICAL INTEGRATION OF ELEMENT MATRICES
   %==========================================================
   weight       = IP_w(ip)*detJ;
   B(1:2:end,1) = dNdX(:,1);
   B(2:2:end,2) = dNdX(:,2);
   B(1:2:end,3) = dNdX(:,2);
   B(2:2:end,3) = dNdX(:,1);
   Bvol         = dNdX';

   A_elem       = A_elem + weight*EEta*(B*DEV*B');
   Q_elem       = Q_elem - weight*Bvol(:)*Pi';
   M_elem       = M_elem + weight*(Pi*Pi');
   Rhs_elem     = Rhs_elem + weight*ERho*EG*Ni';
 end
%==============================================================
% viii) STATIC CONDENSATION
%==============================================================
invM_elem = inv(M_elem);
%
% Schur complement A+kappa Q' M^-1 Q
%
A_elem    = A_elem + PF*Q_elem*invM_elem*Q_elem';

%==============================================================
% ix) WRITE DATA INTO GLOBAL STORAGE
%==============================================================
A_all(:, iel)      = A_elem(indx_l);
Q_all(:, iel)      = Q_elem(:);
invM_all(:,iel)    = invM_elem(:);
Rhs_all(:,iel)     = Rhs_elem(:);
end
fprintf(1, [num2str(toc),'\n']);




%==========================================================================
% ix) CREATE TRIPLET FORMAT INDICES
%==========================================================================
tic; fprintf(1, 'TRIPLET INDICES:    ');
%A matrix

% % Old numbering:
% ELEM_DOF = zeros(nedof, nel);
% ELEM_DOF(1:ndim:end,:) = ndim*(ELEM2NODE-1)+1;
% ELEM_DOF(2:ndim:end,:) = ndim*(ELEM2NODE-1)+2;

%--------------------------------------------------------------------------
%
% periodic BC modification: we use a local to global numbering scheme, which
% simplifies matters if we have periodic BC's
%
ELEM_DOF                = zeros(nedof, nel);
ELEM_DOF(1:ndim:end,:)  = reshape(LOC2GLOB(1,ELEM2NODE),nnodel, nel);
ELEM_DOF(2:ndim:end,:)  = reshape(LOC2GLOB(2,ELEM2NODE),nnodel, nel);
%
%--------------------------------------------------------------------------


indx_j = repmat(1:nedof,nedof,1); indx_i = indx_j';
indx_i = tril(indx_i); indx_i = indx_i(:); indx_i = indx_i(indx_i>0);
indx_j = tril(indx_j); indx_j = indx_j(:); indx_j = indx_j(indx_j>0);

A_i = ELEM_DOF(indx_i(:),:);
A_j = ELEM_DOF(indx_j(:),:);

indx       = A_i < A_j;
tmp        = A_j(indx);
A_j(indx)  = A_i(indx);
A_i(indx)  = tmp;

%Q matrix
Q_i = repmat(1:nel*np,nedof,1);
Q_j = repmat(ELEM_DOF,np,1);

%invM matrix
indx_j = repmat(1:np,np,1); indx_i = indx_j';
invM_i = reshape(1:nel*np,np, nel);
invM_j = invM_i(indx_i,:);
invM_i = invM_i(indx_j,:);

fprintf(1, [num2str(toc),'\n']);

%==========================================================================
% x) CONVERT TRIPLET DATA TO SPARSE MATRIX
%==========================================================================
fprintf(1, 'SPARSIFICATION:     '); tic
A    = sparse(A_i(:)   ,    A_j(:),    A_all(:));
Q    = sparse(Q_i(:)   ,    Q_j(:),    Q_all(:));
invM = sparse(invM_i(:), invM_j(:), invM_all(:));
Rhs  = accumarray(ELEM_DOF(:), Rhs_all(:));
clear ELEM_DOF A_i A_j A_all Q_i Q_j Q_all invM_i invM_j invM_all Rhs_all;
fprintf(1, [num2str(toc),'\n']);


%
% uses a highly inefficient solver
%

%==========================================================================
% BOUNDARY CONDITIONS
%==========================================================================
fprintf(1, 'BDRY CONDITIONS:    '); tic;
Free        = 1:sdof;
%
% apply fixed velocity BCs
%
TMP         = A(:,Bc_ind) + transpose(A(Bc_ind,:));
Rhs         = Rhs - TMP*BC_val';
replace     = Bc_ind;

%
%
% remove parts relating to fixed BCs
Free(replace)= [];
A           = A(Free,Free);

%
% fill in upper triangular part
A = A + triu(A',1);


fprintf(1, [num2str(toc),'\n']);


%==========================================================================
% POWELL-HESTENES ITERATIONS
%==========================================================================
tic
div_max_uz  = 1e-10; div_max     = realmax;
uz_iter     =     0; uz_iter_max =       3;

Pressure    = zeros(nel*np, 1);
% total solution
Vel         = zeros(sdof  , 1);
% boundary condition
Vel(Bc_ind) = BC_val;
%
% iterate for free solution
%
while (div_max>div_max_uz  && uz_iter<uz_iter_max)
   uz_iter         = uz_iter + 1;
   % solve for the free velocities
   Vel(Free) = A\Rhs(Free);


   Div             = invM*(Q*Vel);                            % COMPUTE QUASI-DIVERGENCE
   Rhs             = Rhs - PF*(Q'*Div);                       % UPDATE RHS

   Pressure        = Pressure + PF*Div;                       % UPDATE TOTAL PRESSURE (negative sign convention)
   div_max         = max(abs(Div(:)));                        % CHECK INCOMPRESSIBILITY
   disp([' PH_ITER: ', num2str(uz_iter), ' ', num2str(div_max)]);
end
Pressure = reshape(Pressure,np, nel);
fprintf(1, 'P-H ITERATIONS:     ');
fprintf(1, [num2str(toc,'%8.6f'),'\n']);


% reshape velocity (really only required for periodic BC's)
Vel               = Vel(LOC2GLOB(:));




