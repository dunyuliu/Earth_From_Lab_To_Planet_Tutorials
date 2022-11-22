function area = tri_area(x,y)
%
% compute area of a triangle in 2D given as x and y vectors
%
a = [ x' y' ones(size(x))' ];
area = det(a)/2;

