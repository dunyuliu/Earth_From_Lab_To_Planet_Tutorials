function avg = element_avg(scalar, GCOORD, ELEM2NODE)
%
% compute global average of a quantity given on the nodes of each 
% element such as pressure
%
nel     = size(ELEM2NODE,2);

avg = 0;
area = 0;
for i=1:nel
    eavg = mean(scalar(:,i));
    earea = tri_area(GCOORD(1,ELEM2NODE(1:3,i)),GCOORD(2,ELEM2NODE(1:3,i)));
    avg = avg + earea * eavg;
    area = area + earea;
end

avg = avg/area;
