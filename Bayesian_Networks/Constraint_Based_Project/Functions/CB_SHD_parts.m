function [ EE2, ME2, WD4, WD5 ] = CB_SHD_parts( dag_list, true )
%CB_SHD_PARTS Summary of this function goes here
%   Detailed explanation goes here

dag_step2 = dag_list{1, 1};
dag_step4 = dag_list{1, 2};
dag_step5 = dag_list{1, 3};

EE2 = 0; 
ME2 = 0;
WD4 = 0;
WD5 = 0;

% Compute measures for after step 2 - EE, ME, precision
s = size(dag_step4);
true = double(true);
dag_step2 = double(dag_step2);

for i=1:s(1)
    for j=1:i
        if (true(i, j) == 0 && dag_step2(i, j) == 1)
            if true(j, i) == 0
                EE2 = EE2 + 1;
            end
        end
        if (true(i,j) == 1 && dag_step2(i,j) == 0)
            if dag_step2(j, i) == 0
                ME2 = ME2 + 1;
            end
        end
        if ((true(i, j) == 0) && (dag_step2(i, j) == 0) && (true(j, i) ~= dag_step2(j, i)))
            if true(j, i) == 0
                EE2 = EE2 + 1;
            else
                ME2 = ME2 + 1;
            end
        end
    end
end


% Compute measures for after step 4 and after step 5 - WD
dag_step4 = double(dag_step4);
dag_step5 = double(dag_step5);

for i=1:s(1)
    for j=1:i
        if (true(i, j) == 0 && dag_step4(i, j) == 1)
            if true(j, i) == 1
                WD4 = WD4 + 1;
            end
        end
        
        if (true(i, j) == 0 && dag_step5(i, j) == 1)
            if true(j, i) == 1
                WD5 = WD5 + 1;
            end
        end
        
        if (true(i,j) == 1 && dag_step4(i,j) == 0)
            if dag_step4(j, i) == 1
                WD4 = WD4 + 1;
            end
        end
        
        if (true(i,j) == 1 && dag_step5(i,j) == 0)
            if dag_step5(j, i) == 1
                WD5 = WD5 + 1;
            end
        end
    end
end
               


end

