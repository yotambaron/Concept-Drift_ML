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

% Set CDDRL parameters
threshold = 0.5;
alpha = 0.5;
lambda = 0.8;
tabu_flag = 1;
tabu_size = 10;
robust_flag = 0;
params_flag = 1;
UCL_days = 4;
bns_ensemble = 4;
weights = [];
curr_bn_weight = 0.5;
previous_bns_weight = 1 - curr_bn_weight;
ensemble_list = [];
bns_inds = cell(weeks, 1);

% Results variables to save
aucs_all = zeros(weeks, 1);
TPs_all = zeros(weeks, 1);
FPs_all = zeros(weeks, 1);
FNs_all = zeros(weeks, 1);
TNs_all = zeros(weeks, 1);
C_nodes_all = [];
bns_inds_list = [];

% Load Data
data_path = 'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data\Electricity';
save_path = 'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Results\Electricity\Ensemble_BNs';
df = readtable([data_path '\Electricity_Discretization.csv']); % Discritizied database
df = table2array(df);

n_features = size(df, 2);
initial_df = df(1:stable,:);

dag = Causal_Explorer('MMHC', initial_df-1, max(initial_df), 'MMHC', [], 10, 'unif');
dag = full(dag);
dag = abs(dag);
bnet = mk_bnet(dag, max(df));
for n=1:n_features
    bnet.CPD{n} = tabular_CPD(bnet, n);
end

bnet = learn_params(bnet, initial_df');

%[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_static(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag);
[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag);
%[C_nodes, BNlist, test_num] = CDDRL_dynamic_CB(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', stable, lambda);
%[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic_UCLs_stable(bnet, df, Window, Jump, 'Chi-Squared', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag, UCL_days);
    
 netowrks = (size(BNlist, 2));
 break_points = zeros(weeks, 1);

 for net=1:weeks
     break_points(net) = Jump + Jump * (net-1);
 end

% add BN0 for the first window days to predict correctly
for w=2:window_weeks
    BNlist = [{BNlist{1}}, BNlist];
end

gm = zeros(weeks, 1);
y_score = zeros(weeks, 1);
aucs = zeros(weeks, 1) + 0.5;
TPs = zeros(weeks, 1);
FPs = zeros(weeks, 1);
FNs = zeros(weeks, 1);
TNs = zeros(weeks, 1);

for b=1:weeks
   display('Predicting week: ')
   b
   bn = BNlist{b};
      
   if b==1
       data = df(1:break_points(b),:);
       true = df(1:break_points(b),n) - 1;
       % Normal inference only with the current BN
       [preds, scores] = BN_inference(bn, data, n);
       pred = zeros(size(scores,1), 1);
       pred(scores(:, 2) > threshold) = 1;
       % Add the current BN to the BN list and update weights
       ensemble_list = [ensemble_list; bn];
       weights = [weights; 1];
       bns_inds_list = [bns_inds_list; b];
      
   else
       data = df((break_points(b-1)+1):break_points(b),:);
       true = df((break_points(b-1)+1):break_points(b),n)-1;
           
       % Infere with all BNs in the list + current BN - current BN's
       % weight is decided by 'curr_bn_weight' and the past BNs
       % contribute the rest of the weight by their previous weights
       temp_ensemble_list = [ensemble_list; bn];
       [preds, scores] = BN_inference_ensemble(temp_ensemble_list, data, n, length(temp_ensemble_list));
       temp_weights = weights * previous_bns_weight;
       temp_weights = [temp_weights; curr_bn_weight];
       probabilities = zeros(size(scores,1), 2);
       for s=1:length(temp_weights)
           probabilities = probabilities + temp_weights(s) * reshape(cell2mat(scores(:, s)), [2, size(scores, 1)])';
       end
       pred = zeros(size(scores,1), 1);
       pred(probabilities(:, 2) > threshold) = 1;
        
       % Update current weights by the current performance
       new_weights = update_ensemble_weights(ones(1, length(temp_ensemble_list)), scores, true, 0);
       % Normalize the previous weights with a non real BN weight
       if b > 2
           weights = [weights; 1/length(weights)];
           weights = weights / sum(weights);
           weights(end) = [];
       end
       curr_weight = new_weights(end);
       new_weights(end) = [];
       % New weights of the past BNs is the average between previous
       % and normalized current weights
       weights = (weights + new_weights') / 2;
       if b <= bns_ensemble
           weights = [weights; curr_weight];
           ensemble_list = [ensemble_list; bn];
           bns_inds_list = [bns_inds_list; b];
       else
           % If current BN weight is bigger than any of the previous
           % BNs then replace them in BN list
           min_weight_pos = find(weights == min(weights));
           if weights(min_weight_pos(1)) <= curr_weight
               weights(min_weight_pos(1)) = curr_weight;
               ensemble_list(min_weight_pos(1)) = bn;
               bns_inds_list(min_weight_pos(1)) = b;
           end
       end
       % Normalize the updated BN list weights
       weights = weights / sum(weights);
       scores = probabilities;
   end
      
   bns_inds{b, 1} = bns_inds_list;

   [~,~,~,AUC] = perfcurve(true, scores(:, 2), 1);
   aucs(b) = AUC;
   
   [TPs(b),FPs(b),FNs(b),TNs(b)] = confusion_matrix(pred, true);
end

for b = 1:weeks
    [TP_precision(b,1),TP_recall(b,1),TN_precision(b,1),TN_recall(b,1),FA_rate(b,1),f1_p(b,1),f1_n(b,1),f1_a(b,1),gm(b,1),b_accuracy(b,1),y_score(b,1)] = score_matrices(TPs(b),FPs(b),FNs(b),TNs(b));
end

% Organize Results
All_Results = array2table([weeks_col,TPs,FPs,FNs,TNs,FA_rate,TP_precision,TP_recall,TN_precision,TN_recall,f1_p,f1_n,f1_a,gm,b_accuracy,y_score,aucs]);
All_Results.Properties.VariableNames = {'Week' 'TP' 'FP' 'FN' 'TN' 'FA' 'TP_Precision' 'TP_Recall' 'TN_Precision' 'TN_Recall' 'F1_pos' 'F1_neg' 'F1_Avg' 'Geometric_mean' 'Balanced_Accuracy' 'Y_score' 'AUC'};
writetable(All_Results,[save_path '\All_Results_Ensemble.csv'])
writetable(array2table(C_nodes),[save_path '\Changed_Nodes_Ensemble.csv'])
writetable(cell2table(bns_inds),[save_path '\BNs_In_Ensemble.csv'])
    

