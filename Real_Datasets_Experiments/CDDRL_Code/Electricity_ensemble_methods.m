clear all;

% Set Parameters
jump_weeks = 1;
window_weeks = 3;
stable_weeks = 3;
Jump = 24 * 2 * 7 * jump_weeks;  % One week
Window = Jump * window_weeks; % Two weeks
stable = Jump * stable_weeks;
weeks = 83;
weeks_col = (1: weeks);
weeks_col = reshape(weeks_col, [weeks 1]);

% CDDRL Parameters
alpha = 0.5;
lambda = 0.8;
tabu_flag = 1;
tabu_size = 10;
robust_flag = 0;
params_flag = 1;
UCL_weeks = 3;
threshold = 0.5;

% Load Data
data_path = 'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data\Electricity';
save_path = 'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Results\Electricity\Ensemble_Methods';
df = readtable([data_path '\Electricity_Discretization.csv']); % Discritizied database
df = table2array(df);

% Normal, Tabu10+params, Robust, UCL, CB  
methods_flags = [1, 1, 1, 1, 1];
methods_num = sum(methods_flags);
tabu_size = 10;

% Set CDDRL parameters
alpha = 0.5;
lambda = 0.8;
weights = ones(1, methods_num) / methods_num;
weight_method = 1;
thresholds = 0.5;

% Results variables to save
BNlist_all = [];
C_nodes_all = [];
AUCs_methods = cell(weeks, 1);

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
        BNlist_normal = adjust_BNlist(BNlist_normal, window_weeks, 0);
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
        BNlist_tabu_params = adjust_BNlist(BNlist_tabu_params, window_weeks, 0);
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
        BNlist_robust = adjust_BNlist(BNlist_robust, window_weeks, 0);
        BNlist_all = [BNlist_all; BNlist_robust];
        C_nodes_all = [C_nodes_all, C_nodes_robust];
    end
    
    % Dynamic UCLs CDDRL
    if method == 4 && methods_flags(method) == 1
        tabu_flag = 0;
        params_flag = 0;
        robust_flag = 0;
        UCL_days = 4;
        [C_nodes_UCLs, BNlist_UCLs, ~, ~] = CDDRL_dynamic_UCLs_stable(bnet, df, Window, Jump, 'Chi-Squared', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag, UCL_weeks);
        %BNlist_UCLs = save_bns(BNlist_UCLs, save_path, 'Dynamic_UCLs', 0); 
        BNlist_UCLs = adjust_BNlist(BNlist_UCLs, window_weeks, 0);
        BNlist_all = [BNlist_all; BNlist_UCLs];
        C_nodes_all = [C_nodes_all, C_nodes_UCLs];
    end
    
    % Constraint-Based CDDRL
    if method == 5 && methods_flags(method) == 1
        [C_nodes_CB, BNlist_CB, test_num_CB] = CDDRL_dynamic_CB(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', stable, lambda);
        %BNlist_CB = save_bns(BNlist_CB, save_path, 'CB', 0);
        BNlist_CB = adjust_BNlist(BNlist_CB, window_weeks, 0);
        BNlist_all = [BNlist_all; BNlist_CB];
        C_nodes_all = [C_nodes_all, C_nodes_CB];
    end
end

netowrks = (size(BNlist_all, 2)) - 1;
break_points = zeros(weeks, 1);

for net=1:weeks
    break_points(net) = Jump + Jump * (net-1);
end

    aucs = zeros(weeks, 1) + 0.5;
    TPs = zeros(weeks, 1);
    FPs = zeros(weeks, 1);
    FNs = zeros(weeks, 1);
    TNs = zeros(weeks, 1);
    preds_array = [];
    scores_array = [];
    weights_all = [];
    
    for b=1:weeks
        display('Predicting week: ')
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
        probabilities = zeros(size(scores,1), 2);
        for s=1:length(weights)
            probabilities = probabilities + weights(s) * reshape(cell2mat(scores(:, s)), [2, size(scores, 1)])';
        end
        pred = zeros(size(scores,1), 1);
        pred(probabilities(:, 2) > threshold) = 1;
        
        % Update the previous weights with the current methods' performance
        [weights, AUCs_methods{b}] = update_ensemble_weights(weights, scores, true, weight_method);

        preds_array = [preds_array; pred];
        scores_array = [scores_array; probabilities];

        [~,~,~,AUC] = perfcurve(true, probabilities(:, 2), 1);
        aucs(b) = AUC;

        [TPs(b),FPs(b),FNs(b),TNs(b)] = confusion_matrix(pred, true);
        [TP_precision(b,1),TP_recall(b,1),TN_precision(b,1),TN_recall(b,1),FA_rate(b,1),f1_p(b,1),f1_n(b,1),f1_a(b,1),gm(b,1),b_accuracy(b,1),y_score(b,1)] = score_matrices(TPs(b),FPs(b),FNs(b),TNs(b));
    end

% Combine all BNs from the different methods by a certain combine method
weight_threshold = 0.5;
bns_all = BNlist_all(:, (1 + window_weeks):weeks+1);

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
All_Results = array2table([weeks_col,TPs,FPs,FNs,TNs,FA_rate,TP_precision,TP_recall,TN_precision,TN_recall,f1_p,f1_n,f1_a,gm,b_accuracy,y_score,aucs]);
All_Results.Properties.VariableNames = {'Week' 'TP' 'FP' 'FN' 'TN' 'FA' 'TP_Precision' 'TP_Recall' 'TN_Precision' 'TN_Recall' 'F1_pos' 'F1_neg' 'F1_Avg' 'Geometric_mean' 'Balanced_Accuracy' 'Y_score' 'AUC'};
writetable(All_Results,[save_path '\All_Results.csv'])
writetable(cell2table(AUCs_methods),[save_path '\AUCs_methods.csv'])
writetable(array2table(C_nodes_all),[save_path '\Changed_Nodes.csv'])
writetable(array2table(weights_all),[save_path '\weights.csv'])
    