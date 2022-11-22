function [ipx] = lp_triangle(second_order)

% local coordinates of nodes
%
% is second_order is 1, will return six nodes, else three

%
% ipx(nip, 2): coordinates of local points
%
% TWB 
%

%   Part of MILAMIN: MATLAB-based FEM solver for large problems, Version 1.0
%   Copyright (C) 2007, M. Dabrowski, M. Krotkiewski, D.W. Schmid
%   University of Oslo, Physics of Geological Processes
%   http://milamin.org
%   See License file for terms of use.

% see e.g.: 
% Dunavant, D. A. 1985. High-Degree efficient symmetrical Gaussian quadrature rules for the triangle. Int. J. Numer. Methods Eng. 21, 6, 1129--1148.

if(second_order == 1)
    n=6;
else
    n=3;
end
ipx=zeros(n,2);

ipx(1,1) = 1; % r
ipx(1,2) = 0; % s
ipx(2,1) = 0;
ipx(2,2) = 1; 
ipx(3,1) = 0;
ipx(3,2) = 0; 
if(n > 3)
    ipx(4,1) = .5; % r
    ipx(4,2) = .5; % s
    ipx(5,1) = 0;
    ipx(5,2) = .5;
    ipx(6,1) = 0.5;
    ipx(6,2) = 0;
    
end


end
