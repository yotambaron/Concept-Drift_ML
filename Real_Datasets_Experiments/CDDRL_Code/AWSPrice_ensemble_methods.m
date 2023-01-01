clear all;

% Set Parameters
window_steps = 2;
stable_steps = 4;
Jump = 3700;  
Window = Jump * window_steps; % Two steps
stable = Jump * stable_steps; 
steps = 54;
steps_col = (1: steps);
steps_col = reshape(steps_col, [steps 1]);

% CDDRL Parameters
classes = 3;
alpha = 0.5;
lambda = 0.8;
tabu_flag = 1;
tabu_size = 10;
robust_flag = 0;
params_flag = 1;
UCL_steps = 3;

% Load Data
data_path = 'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data\AWSPrice';
save_path = 'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Results\AWSPrice\CDDRL\Ensemble_Methods';
df = readtable([data_path '\AWSPrice_Discretization.csv']); % Discritizied database
df = table2array(df);

n_features = size(df, 2);
initial_df = df(1:stable,:);
num_of_classes = max(df(:, size(df, 2)));

% Normal, Tabu10+params, Robust, UCL, CB  
methods_flags = [1, 1, 1, 1, 1];
methods_num = sum(methods_flags);
tabu_size = 10;

% Set CDDRL parameters
alpha = 0.5;
lambda = 0.8;
weights = ones(1, methods_num) / methods_num;
weight_method = 0;

% Results variables to save
BNlist_all = [];
C_nodes_all = [];
AUCs_methods = cell(steps, 1);

n_features = size(df, 2);
initial_df = df(1:stable, :);
dag = Causal_Explorer('MMHC', initial_df-1, max(initial_df), 'MMHC', [], 10, 'unif');
dag = full(dag);
dag = abs(dag);
bnet = mk_bnet(dag, max(df));
for n=1:n_features
    bnet.CPD{n} = tabular_CPD(bnet, n);
end

bnet = learn_params(bnet, initial_df');

for method=1:length(methods_flags)
   
    % Normal CDDRL
    if method == 1 && methods_flags(method) == 1
        tabu_flag = 0;
        params_flag = 0;
        robust_flag = 0;
        [C_nodes_normal, BNlist_normal, ~, ~] = CDDRL_dynamic(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag);
        %BNlist_normal = save_bns(BNlist_normal, save_path, 'Normal', 0);
        BNlist_normal = adjust_BNlist(BNlist_normal, window_steps, 0);
        BNlist_all = [BNlist_all; BNlist_normal];
        C_nodes_all = [C_nodes_all, C_nodes_normal];
    end
    
    % Tabu size 10 + params CDDRL
    if method == 2 && methods_flags(method) == 1
        tabu_flag = 1;
        params_flag = 1;
        robust_flag = 0;
        [C_nodes_tabu_params, BNlist_tabu_params, ~, ~] = CDDRL_dynamic(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag);
        %BNlist_tabu_params = save_bns(BNlist_tabu_params, save_path, 'Tabu_Params', 0);
        BNlist_tabu_params = adjust_BNlist(BNlist_tabu_params, window_steps, 0);
        BNlist_all = [BNlist_all; BNlist_tabu_params];
        C_nodes_all = [C_nodes_all, C_nodes_tabu_params];
    end
    
    % Robust CDDRL
    if method == 3 && methods_flags(method) == 1
        tabu_flag = 0;
        params_flag = 0;
        robust_flag = 1;
        [C_nodes_robust, BNlist_robust, ~, ~] = CDDRL_dynamic(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag);
        %BNlist_robust = save_bns(BNlist_robust, save_path, 'Robust', 0);
        BNlist_robust = adjust_BNlist(BNlist_robust, window_steps, 0);
        BNlist_all = [BNlist_all; BNlist_robust];
        C_nodes_all = [C_nodes_all, C_nodes_robust];
    end
    
    % Dynamic UCLs CDDRL
    if method == 4 && methods_flags(method) == 1
        tabu_flag = 0;
        params_flag = 0;
        robust_flag = 0;
        UCL_days = 4;
        [C_nodes_UCLs, BNlist_UCLs, ~, ~] = CDDRL_dynamic_UCLs_stable(bnet, df, Window, Jump, 'Chi-Squared', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag, UCL_steps);
        %BNlist_UCLs = save_bns(BNlist_UCLs, save_path, 'Dynamic_UCLs', 0); 
        BNlist_UCLs = adjust_BNlist(BNlist_UCLs, window_steps, 0);
        BNlist_all = [BNlist_all; BNlist_UCLs];
        C_nodes_all = [C_nodes_all, C_nodes_UCLs];
    end
    
    % Constraint-Based CDDRL
    if method == 5 && methods_flags(method) == 1
        [C_nodes_CB, BNlist_CB, test_num_CB] = CDDRL_dynamic_CB(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', stable, lambda);
        %BNlist_CB = save_bns(BNlist_CB, save_path, 'CB', 0);
        BNlist_CB = adjust_BNlist(BNlist_CB, window_steps, 0);
        BNlist_all = [BNlist_all; BNlist_CB];
        C_nodes_all = [C_nodes_all, C_nodes_CB];
    end
end

netowrks = (size(BNlist_all, 2)) - 1;
break_points = zeros(steps, 1);

for net=1:steps
    break_points(net) = Jump + Jump * (net-1);
end

    preds_array = [];
    scores_array = [];
    weights_all = [];
    
    for b=1:steps
        display('Predicting step: ')
        b
        weights_all = [weights_all; weights];
        bns = [];
        if b==1
            data = df(1:break_points(b),:);
            true = df(1:break_points(b),n) - 1;
        else
            data = df((break_points(b-1)+1):break_points(b),:);
            true = df((break_points(b-1)+1):break_points(b),n)-1;
        end
        
        for m=1:size(BNlist_all, 1)
            bns = [bns; BNlist_all{m, b}];
        end
        
        % Infere by the weighted predictions of all methods
        [preds, scores] = BN_inference_ensemble(bns, data, n, methods_num);
        probabilities = zeros(size(scores,1), num_of_classes);
        for s=1:length(weights)
            probabilities = probabilities + weights(s) * reshape(cell2mat(scores(:, s)), [num_of_classes, size(scores, 1)])';
        end
        
        % Update the previous weights with the current methods' performance
        [weights, AUCs_methods{b}] = update_ensemble_weights(weights, scores, true, weight_method, 0, classes);

        scores_array = [scores_array; bsxfun(@rdivide, probabilities, sum(probabilities, 2))];
        [~, pred] = max(probabilities, [], 2);
        preds_array = [preds_array; pred];

    end

% Combine all BNs from the different methods by a certain combine method
weight_threshold = 0.5;
bns_all = BNlist_all(:, (1 + window_steps):steps+1);

combine_method = 'every_edge'; % options = ['every_edge', 'majority_vote', 'every_edge_with_weight', 'every_edge_over_weight_threshold']
folder_name = ['Combined\' combine_method];
bns_combined = combine_bns(bns_all, combine_method, weights_all, weight_threshold);
bns_combined = save_bns(bns_combined, save_path, folder_name, 1);


combine_method = 'majority_vote'; % options = ['every_edge', 'majority_vote', 'every_edge_with_weight', 'every_edge_over_weight_threshold']
folder_name = ['Combined\' combine_method];
bns_combined = combine_bns(bns_all, combine_method, weights_all, weight_threshold);
bns_combined = save_bns(bns_combined, save_path, folder_name, 1);


combine_method = 'every_edge_with_weight'; % options = ['every_edge', 'majority_vote', 'every_edge_with_weight', 'every_edge_over_weight_threshold']
folder_name = ['Combined\' combine_method];
bns_combined = combine_bns(bns_all, combine_method, weights_all, weight_threshold);
bns_combined = save_bns(bns_combined, save_path, folder_name, 1);


combine_method = 'every_edge_over_weight_threshold'; % options = ['every_edge', 'majority_vote', 'every_edge_with_weight', 'every_edge_over_weight_threshold']
folder_name = ['Combined\' combine_method];
bns_combined = combine_bns(bns_all, combine_method, weights_all, weight_threshold);
bns_combined = save_bns(bns_combined, save_path, folder_name, 1);

% Save results
writetable(array2table(preds_array),[save_path '\Predictions.csv'], 'WriteVariableNames', 0)
writetable(array2table(scores_array),[save_path '\Scores.csv'], 'WriteVariableNames', 0)
writetable(cell2table(AUCs_methods),[save_path '\AUCs_methods.csv'])
writetable(array2table(C_nodes_all),[save_path '\Changed_Nodes.csv'])
writetable(array2table(weights_all),[save_path '\weights.csv'])
    