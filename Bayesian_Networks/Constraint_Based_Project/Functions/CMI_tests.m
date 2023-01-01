function [flag, sep_sets, num_of_tests] = CMI_tests(a, b, data, sep_sets, max_order, pc, remove, num_of_tests)
%MCI_TESTS Summary of this function goes here
%   Detailed explanation goes here

order = 0;
flag = 1;
curr_pc = setdiff(union(pc{1,a}, pc{1,b}), union(a, b)); % Unite previous pc of both a and b

while order <= max_order
    if remove
        fprintf('Checking removal of edge %d -> %d with CI tests of order %d\n', a, b, order);
    else
        fprintf('Checking addition of edge %d -> %d with CI tests of order %d\n', a, b, order);
    end
    if length(curr_pc) >= order
        Seps = subsets1(curr_pc, order);  % Get all possible subsets in the current order
        for si=1:length(Seps)   % Go through the subsets
            S = Seps{si};
            num_of_tests = num_of_tests + 1;
            threshold = s1cond(a, b, S, data, 0.01);  % Compute current CMI threshold
            % CMI test - CI is 0 if dependant and 1 otherwise, I is the test's value
            [CI, I] = mutual2_decision(a, b, S, data, threshold);
            if CI  % a and b are indipendant given S
                sep_sets{a, b} = S;
                sep_sets{b, a} = S;
                order = max_order + 1;  % Exit while loop
                flag = 0;   % No edge flag
                break;
            end
        end
    else
        order = max_order + 1;
    end
    order = order + 1;
end % End while

end

