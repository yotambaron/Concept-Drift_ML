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

% Set CDDRL parameters
classes = 3;
alpha = 0.5;
lambda = 0.8;
tabu_flag = 1;
tabu_size = 10;
robust_flag = 0;
params_flag = 1;
UCL_steps = 3;
bns_ensemble = 4;
weights = [];
curr_bn_weight = 0.5;
previous_bns_weight = 1 - curr_bn_weight;
ensemble_list = [];
bns_inds = cell(steps, 1);

% Results variables to save
aucs_all = zeros(steps, 1);
TPs_all = zeros(steps, 1);
FPs_all = zeros(steps, 1);
FNs_all = zeros(steps, 1);
TNs_all = zeros(steps, 1);
C_nodes_all = [];
bns_inds_list = [];

% Load Data
data_path = 'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data\AWSPrice';
save_path = 'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Results\AWSPrice\CDDRL\Ensemble_BNs';
df = readtable([data_path '\AWSPrice_Discretization.csv']); % Discritizied database
df = table2array(df);

n_features = size(df, 2);
initial_df = df(1:stable,:);
num_of_classes = max(df(:, size(df, 2)));

dag = Causal_Explorer('MMHC', initial_df-1, max(initial_df), 'MMHC', [], 10, 'unif');
dag = full(dag);
dag = abs(dag);
bnet = mk_bnet(dag, max(df));
for n=1:n_features
    bnet.CPD{n} = tabular_CPD(bnet, n);
end

bnet = learn_params(bnet, initial_df');

[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag);
%[C_nodes, BNlist, test_num] = CDDRL_dynamic_CB(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', stable, lambda);
%[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic_UCLs_stable(bnet, df, Window, Jump, 'Chi-Squared', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag, UCL_steps);
    
 netowrks = (size(BNlist, 2));
 break_points = zeros(steps, 1);

 for net=1:steps
     break_points(net) = Jump + Jump * (net-1);
 end

% add BN0 for the first window days to predict correctly
for w=1:window_steps
    BNlist = [{BNlist{1}}, BNlist];
end

gm = zeros(steps, 1);
y_score = zeros(steps, 1);
aucs = zeros(steps, 1) + 0.5;
TPs = zeros(steps, 1);
FPs = zeros(steps, 1);
FNs = zeros(steps, 1);
TNs = zeros(steps, 1);
preds_array = [];
scores_array = [];

for b=1:steps
   display('Predicting time step: ')
   b
   bn = BNlist{b};
      
   if b==1
       data = df(1:break_points(b),:);
       true = df(1:break_points(b),n) - 1;
       % Normal inference only with the current BN
       [preds, scores] = BN_inference(bn, data, n);
       preds_array = [preds_array; preds];
       scores_array = [scores_array; scores];
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
       probabilities = zeros(size(scores,1), num_of_classes);
       for s=1:length(temp_weights)
           probabilities = probabilities + temp_weights(s) * reshape(cell2mat(scores(:, s)), [num_of_classes, size(scores, 1)])';
       end
       scores_array = [scores_array; bsxfun(@rdivide, probabilities, sum(probabilities,2))];
       [~, pred] = max(probabilities, [], 2);
       preds_array = [preds_array; pred];
        
       % Update current weights by the current performance
       [new_weights, ~] = update_ensemble_weights(ones(1, length(temp_ensemble_list)), scores, true, 0, 0, classes);
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

end

% Organize Results
writetable(array2table(preds_array),[save_path '\Predictions.csv'], 'WriteVariableNames', 0)
writetable(array2table(scores_array),[save_path '\Scores.csv'], 'WriteVariableNames', 0)
writetable(array2table(C_nodes),[save_path '\Changed_Nodes_Ensemble.csv'])
writetable(cell2table(bns_inds),[save_path '\BNs_In_Ensemble.csv'])
    

