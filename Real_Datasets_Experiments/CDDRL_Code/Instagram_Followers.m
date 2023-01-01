clear all;

% Set Parameters
jump_weeks = 1;
window_weeks = 1;
stable_weeks = 10;
Jump = 1000;  % One week
Window = Jump * window_weeks; % Two weeks
stable = Jump * stable_weeks;
weeks = 70;
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
data_path = 'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data\Instagram_Followers';
save_path = 'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Results\Instagram_Followers\CDDRL\þþTabu10+Params_Jump1000';
df = readtable([data_path '\Instagram_Followers_Discretization.csv']); % Discritizied database
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
%[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic_UCLs_stable(bnet, df, Window, Jump, 'Chi-Squared', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag, UCL_weeks);
time1=toc;

% Save BNs
for bb=1:length(BNlist)
    BNs_path = [save_path '\BNs\BN' num2str(bb) '.csv'];
    dag_i = BNlist{bb}.dag;
    %csvwrite(BNs_path, dag_i)
end

netowrks=(size(BNlist,2));
break_points = zeros(weeks, 1);
break_points(1) = Jump;

for net=2:weeks
    break_points(net) = Jump+Jump*(net-1);
end

BNlist = adjust_BNlist(BNlist, window_weeks, 0);

% Predict and Calculate Scores
aucs = zeros(weeks, 1) + 0.5;
FAs = zeros(weeks, 1);
TPs = zeros(weeks, 1);
FPs = zeros(weeks, 1);
FNs = zeros(weeks, 1);
TNs = zeros(weeks, 1);
preds_array = [];
scores_array = [];

tic;

for b=1:weeks
    display('Predicting week: ')
    b
    bn = BNlist{b};
    if b==1
       data = df(1:break_points(b),:);
       true = df(1:break_points(b),n) - 1;
    else
       data = df((break_points(b-1)+1):break_points(b),:);
       true = df((break_points(b-1)+1):break_points(b),n)-1;
    end
    [preds, scores] = BN_inference(bn, data, n);
    pred = zeros(size(scores, 1), 1);
    pred(scores(:, 2) > threshold) = 1;
    preds_array = [preds_array;pred];
    scores_array = [scores_array;scores];
    [~,~,~,AUC] = perfcurve(true, scores(:, 2), 1);
    aucs(b) = AUC;
    [TPs(b),FPs(b),FNs(b),TNs(b)] = confusion_matrix(pred, true);
    [TP_precision(b,1),TP_recall(b,1),TN_precision(b,1),TN_recall(b,1),FA_rate(b,1),f1_p(b,1),f1_n(b,1),f1_a(b,1),gm(b,1),b_accuracy(b,1),y_score(b,1)] = score_matrices(TPs(b),FPs(b),FNs(b),TNs(b));
end

time2 = toc;
all_time = time1 + time2;
Times = array2table([time1, time2, all_time]);
Times.Properties.VariableNames = {'CDDRL_Run_Time' 'Inference_Run_time' 'Total_Run_Time'};
writetable(Times,[save_path '\Run_Time.csv'])

% Organize Results
%All_Results = array2table([weeks_col,TPs,FPs,FNs,TNs,FA_rate,TP_precision,TP_recall,TN_precision,TN_recall,f1_p,f1_n,f1_a,gm,b_accuracy,y_score,aucs]);
%All_Results.Properties.VariableNames = {'Week' 'TP' 'FP' 'FN' 'TN' 'FA' 'TP_Precision' 'TP_Recall' 'TN_Precision' 'TN_Recall' 'F1_pos' 'F1_neg' 'F1_Avg' 'Geometric_mean' 'Balanced_Accuracy' 'Y_score' 'AUC'};
%Dist_Array = array2table(Dist_Array);
%UCL_Array = array2table(UCL_Array);
%Dist_Array.Properties.VariableNames = {'day' 'hour' 'nsw_price' 'nsw_demand' 'vic_price' 'vic_demand' 'transfer' 'class'};
%CL_Array.Properties.VariableNames = {'day' 'hour' 'nsw_price' 'nsw_demand' 'vic_price' 'vic_demand' 'transfer' 'class'};

% Save Results
%writetable(All_Results,[save_path '\All_Results.csv'])
%writetable(Dist_Array,[save_path '\Distances.csv'])
%writetable(UCL_Array,[save_path '\UCLs.csv'])
%writetable(array2table(C_nodes),[save_path '\Changed_Nodes.csv'])
%writetable(array2table(time),[save_path '\Run_Time.csv'])
%writetable(array2table(preds_array),[save_path '\Predictions.csv'], 'WriteVariableNames', 0)
%writetable(array2table(scores_array),[save_path '\Scores.csv'], 'WriteVariableNames', 0)

