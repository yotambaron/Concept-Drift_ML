clear all;
window = 3000;
jump = 1000;
stable = window + jump;
n_permutations = 10;
time_steps = 30;
steps_col = (1:time_steps);
steps_col = reshape(steps_col,[time_steps 1]);
df_path = 'C:\Users\User\Desktop\yotam\MATLAB\Constrain-Based CDDRL\DBs\Sin';
save_path = 'C:\Users\User\Desktop\yotam\MATLAB\Constrain-Based CDDRL\Results\Sin';
res_cddrl = zeros(time_steps, 7);
results_rest = zeros(time_steps, 3);
tabu_flag = 0;
tabu_size = 10;
robust_flag = 0;
params_flag = 0;
UCL_days = 3;
alpha = 0.5;
threshold = 0.5;
lambda = 0.8;
all_scores = [];
t = 0;
cb = 0;
tests = 0;
PC_tests = 0;
all_times = zeros(3, 1);
cddrl = 1;

for p=1:n_permutations
    
    df = readtable([df_path '\\' 'Sin_data' num2str(p) '.csv']);
    df = table2array(df);
    n_features = size(df,2);    
    dag_list = cell(time_steps, 3);
    
    if cddrl
        initial_df = df(1:stable,:);
        dag = Causal_Explorer('MMHC',initial_df,max(initial_df),'MMHC',[],10,'unif');
        dag = full(dag);
        dag = abs(dag);
        bnet = mk_bnet(dag, max(initial_df));
        for n=1:n_features
            bnet.CPD{n} = tabular_CPD(bnet, n);
        end
        bnet.names = {'color', 'area', 'curvature', 'noise1', 'noise2', 'noise3', 'noise4', 'noise5', 'class'};
        bnet = learn_params(bnet, initial_df');
    
        %tic
        %[C_nodes, BNlist, test_num] = CDDRL_dynamic_exp(cb,bnet,df,window,jump,'Chi-Squared','EWMA_Wmean',0.5,0,1,0,stable,lambda,tabu_flag,tabu_size,robust_flag);
        %tests = tests + test_num;
        %t = t + toc;
        tic
        if cb
            [C_nodes, BNlist, test_num] = CDDRL_dynamic_CB(bnet, df, window, jump, 'Chi-Squared', 'EWMA_Wmean', stable, lambda);
            tests = tests + test_num;
        else
            [C_nodes, BNlist, ~, ~] = CDDRL_dynamic(bnet, df, window, jump, 'Chi-Squared', 'EWMA_Wmean', alpha, stable, lambda, tabu_flag, tabu_size, robust_flag, params_flag);
        end
        t = t + toc;
        
        netowrks=(size(BNlist,2) + 1);
        break_points = zeros(netowrks+1,1);
        break_points(1) = jump;
    
        for net=2:netowrks+1
            break_points(net) = jump+jump*(net-1);
        end
    
        BNlist = [ {BNlist{1}}, BNlist ];
        BNlist = [ {BNlist{1}}, BNlist ];
        BNlist = [ {BNlist{1}}, BNlist ];

        f1_bn_change = zeros(time_steps,7);
        scores_array = [];
        for b=1:time_steps
            bn = BNlist{b};
            if b==1
                data = df(1:break_points(b),:);
                true = df(1:break_points(b),n) - 1;
            else
                data = df((break_points(b-1)+1):break_points(b),:);
                true = df((break_points(b-1)+1):break_points(b),n) - 1;
            end
            [pred, scores] = BN_inference(bn,data,n);
            pred2 = zeros(jump,1);
            pred2(scores > threshold) = 1;
            scores_array = [scores_array;scores];
            for r=1:size(pred,1)
                if size(pred{r,1},1)>1
                    pred{r,1} = datasample([1,2],1);
                end
            end
            pred = cell2mat(pred) - 1;
            [TP,FP,FN,TN,Precision,Recall,F1] = Scoring(pred2, true);
            f1_bn_change(b,:) = [TP,FP,FN,TN,Precision,Recall,F1];
        end
        
        res_cddrl = res_cddrl + f1_bn_change;
        all_scores = [all_scores, scores_array]
    
        f1_bn_change
        p
    
    else
        [MMHC_dags, K2_dags, PC_dags, run_times, CI] = Learn_dags(df, time_steps, jump);
        PC_tests = PC_tests + CI;
        all_times = all_times + run_times;
    
        netowrks=(size(dag_list, 1)) - 1;
        break_points = zeros(netowrks+1,1);
        break_points(1) = jump;
    
        for net=2:netowrks+1
            break_points(net) = jump+jump*(net-1);
        end
   
        temp_results = Predict(df, MMHC_dags, time_steps, break_points, jump, threshold, 0);
        results_rest(:, 1) = results_rest(:, 1) + temp_results;
        temp_results = Predict(df, K2_dags, time_steps, break_points, jump, threshold, 0);
        results_rest(:, 2) = results_rest(:, 2) + temp_results;
        temp_results = Predict(df, PC_dags, time_steps, break_points, jump, threshold, 0);
        results_rest(:, 3) = results_rest(:, 3) + temp_results;
    
        results_rest
        p
    end
end

if cddrl
    res_cddrl = res_cddrl/n_permutations;
    CDDRL_RunTime = zeros(time_steps,1);
    CDDRL_RunTime(1,1) = t/n_permutations;
    CDDRL_Tests = zeros(time_steps,1);
    CDDRL_Tests(1,1) = tests/n_permutations;
    All_Results = array2table([steps_col,res_cddrl,CDDRL_RunTime,CDDRL_Tests]);
    All_Results.Properties.VariableNames = {'Step' 'TP' 'FP' 'FN' 'TN' 'Precision' 'Recall' 'F1' 'RunTime' 'Tests'};
    writetable(All_Results,[save_path '\try\Sin_CDDRL_S&S_Results.csv']);
else
    results_rest = results_rest/n_permutations;
    Rest_RunTime = all_times/n_permutations;
    PC_Tests = zeros(time_steps, 1);
    PC_Tests(1,1) = PC_tests/n_permutations;
    All_Results = array2table([steps_col, results_rest, PC_Tests]);
    All_Results.Properties.VariableNames = {'Step' 'F1_MMHC' 'F1_K2' 'F1_PC' 'PC_Tests'};
    All_RunTimes = array2table(Rest_RunTime);
    All_RunTimes.Properties.RowNames = {'MMHC', 'K2', 'PC'};
    writetable(All_Results,[save_path '\Sin_Rest_Results.csv']);
    writetable(All_RunTimes,[save_path '\Sin_Rest_RunTimes.csv'], 'WriteRowNames', 1);
end


