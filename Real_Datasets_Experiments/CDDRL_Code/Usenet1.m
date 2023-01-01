clear all;

% Set Parameters
jump_months = 1;
window_months = 1;
stable_months = 3;
month_obs = 100;
Jump = month_obs  * jump_months;  % Two months
Window = month_obs  * window_months; % Four months
stable = month_obs  * stable_months;
months = 15;
months_col = (1: months);
months_col = reshape(months_col, [months 1]);

% CDDRL Parameters
alpha = 0.5;
lambda = 0.8;
tabu_flag = 1;
tabu_size = 10;
robust_flag = 0;
params_flag = 1;
UCL_months = 3;
threshold = 0.5;

% Load Data
data_path = 'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data\Usenet';
save_path = 'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Results\Usenet\CDDRL\Tabu10+Params';
df = readtable([data_path '\usenet1.csv']); % Discritizied database
df = table2array(df);
df = df + 1;

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
%[C_nodes, BNlist, Dist_Array, UCL_Array] = CDDRL_dynamic_UCLs_stable(bnet, df, Window, Jump, 'Chi-Squared', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag, UCL_months);

%BN0 = bnet;
%DB = df;
%window = Window;
%jump = Jump;
%Dist_metric = 'Chi-Squared';
%Spc_Test = 'EWMA_Wmean';

time=toc;

% Save BNs
for bb=1:length(BNlist)
    BNs_path = [save_path '\BNs\BN' num2str(bb) '.csv'];
    dag_i = BNlist{bb}.dag;
    csvwrite(BNs_path, dag_i)
end

netowrks=(size(BNlist,2));
break_points = zeros(months, 1);
break_points(1) = Jump;

for net=2:months
    break_points(net) = Jump+Jump*(net-1);
end

BNlist = adjust_BNlist(BNlist, window_months, 0);

% Predict and Calculate Scores
aucs = zeros(months, 1) + 0.5;
FAs = zeros(months, 1);
TPs = zeros(months, 1);
FPs = zeros(months, 1);
FNs = zeros(months, 1);
TNs = zeros(months, 1);
preds_array = [];
scores_array = [];

for b=1:months
    display('Predicting month: ')
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

% Organize Results
All_Results = array2table([months_col,TPs,FPs,FNs,TNs,FA_rate,TP_precision,TP_recall,TN_precision,TN_recall,f1_p,f1_n,f1_a,gm,b_accuracy,y_score,aucs]);
All_Results.Properties.VariableNames = {'Week' 'TP' 'FP' 'FN' 'TN' 'FA' 'TP_Precision' 'TP_Recall' 'TN_Precision' 'TN_Recall' 'F1_pos' 'F1_neg' 'F1_Avg' 'Geometric_mean' 'Balanced_Accuracy' 'Y_score' 'AUC'};
Dist_Array = array2table(Dist_Array);
UCL_Array = array2table(UCL_Array);
%Dist_Array.Properties.VariableNames = {'AGE' 'PAYMENT_DAY' 'MONTHS_IN_RESIDENCE' 'MONTHS_IN_THE_JOB' 'QUANT_ADDITIONAL_CARDS_IN_THE_APPLICATION' 'PERSONAL_NET_INCOME' 'SEX_M' 'MARITAL_STATUS_Married'	'MARITAL_STATUS_Single' 'FLAG_RESIDENCIAL_PHONE_Y' 'RESIDENCE_TYPE_Rent' 'RESIDENCE_TYPE_Own' 'FLAG_RESIDENCE_TOWN_eq_WORKING_TOWN_Y' 'Client_Type'};
%UCL_Array.Properties.VariableNames = {'AGE' 'PAYMENT_DAY' 'MONTHS_IN_RESIDENCE' 'MONTHS_IN_THE_JOB' 'QUANT_ADDITIONAL_CARDS_IN_THE_APPLICATION' 'PERSONAL_NET_INCOME' 'SEX_M' 'MARITAL_STATUS_Married'	'MARITAL_STATUS_Single' 'FLAG_RESIDENCIAL_PHONE_Y' 'RESIDENCE_TYPE_Rent' 'RESIDENCE_TYPE_Own' 'FLAG_RESIDENCE_TOWN_eq_WORKING_TOWN_Y' 'Client_Type'};

% Save Results
writetable(All_Results,[save_path '\All_Results.csv'])
writetable(Dist_Array,[save_path '\Distances.csv'])
writetable(UCL_Array,[save_path '\UCLs.csv'])
writetable(array2table(C_nodes),[save_path '\Changed_Nodes.csv'])
writetable(array2table(time),[save_path '\Run_Time.csv'])
writetable(array2table(preds_array),[save_path '\Predictions.csv'])
writetable(array2table(scores_array),[save_path '\Scores.csv'])
