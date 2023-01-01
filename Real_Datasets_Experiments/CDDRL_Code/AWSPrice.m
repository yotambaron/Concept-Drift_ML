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
alpha = 0.5;
lambda = 0.8;
tabu_flag = 1;
tabu_size = 10;
robust_flag = 0;
params_flag = 1;
UCL_steps = 3;

% Load Data
data_path = 'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data\AWSPrice';
save_path = 'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Results\AWSPrice\CDDRL\Tabu10+Params';
df = readtable([data_path '\AWSPrice_Discretization.csv']); % Discritizied database
df = table2array(df);

n_features = size(df, 2);
initial_df = df(1:stable,:);

% Learn First BN With MMHC
dag = Causal_Explorer('MMHC',initial_df-1,max(initial_df),'MMHC',[],100,'unif');
dag = full(dag);
dag = abs(dag);
bnet = mk_bnet(dag, max(df));
for n=1:n_features
    bnet.CPD{n} = tabular_CPD(bnet, n);
end
bnet = learn_params(bnet, initial_df');

% Run CDDRL
tic;
[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag);
%[C_nodes, BNlist, test_num] = CDDRL_dynamic_CB(bnet, df, Window, Jump, 'Chi-Squared', 'EWMA_Wmean', stable, lambda);
%[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic_UCLs_stable(bnet, df, Window, Jump, 'Chi-Squared', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag, UCL_steps);
time=toc;

% Save BNs
for bb=1:length(BNlist)
    BNs_path = [save_path '\BNs\BN' num2str(bb) '.csv'];
    dag_i = BNlist{bb}.dag;
    csvwrite(BNs_path, dag_i)
end

netowrks=(size(BNlist,2));
break_points = zeros(steps, 1);
break_points(1) = Jump;

for net=2:steps
    break_points(net) = Jump+Jump*(net-1);
end

BNlist = adjust_BNlist(BNlist, window_steps, 0);

% Predict and Calculate Scores
aucs = zeros(steps, 1) + 0.5;
FAs = zeros(steps, 1);
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
    else
       data = df((break_points(b-1)+1):break_points(b),:);
       true = df((break_points(b-1)+1):break_points(b),n) - 1;
    end
    [preds, scores] = BN_inference(bn, data, n);
    preds_array = [preds_array; preds];
    scores_array = [scores_array; scores];
end

% Organize Results
Dist_Array = array2table(Dist_Array);
UCL_Array = array2table(UCL_Array);
Dist_Array.Properties.VariableNames = {'day' 'hour' 'minute' 'Instance_type' 'Operating_System' 'Region' 'price'};
UCL_Array.Properties.VariableNames = {'day' 'hour' 'minute' 'Instance_type' 'Operating_System' 'Region' 'price'};

% Save Results
writetable(array2table(preds_array),[save_path '\predictions.csv'], 'WriteVariableNames', 0)
writetable(array2table(scores_array),[save_path '\scores.csv'], 'WriteVariableNames', 0)
writetable(array2table(C_nodes),[save_path '\Changed_Nodes.csv'])
writetable(array2table(time),[save_path '\Run_Time.csv'])
writetable(Dist_Array,[save_path '\Distances.csv'])
writetable(UCL_Array,[save_path '\UCLs.csv'])


