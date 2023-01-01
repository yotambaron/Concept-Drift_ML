clear all;

% Simulate and save data bases of all sizes for all bns changes
original_path = 'C:\Users\User\Desktop\yotam\MATLAB\Constrain-Based CDDRL\DBs\Sachs\Original\';
change_path = 'C:\Users\User\Desktop\yotam\MATLAB\Constrain-Based CDDRL\DBs\Sachs\Change\';
nets_path = 'C:\Users\User\Desktop\yotam\MATLAB\Constrain-Based CDDRL\Networks\Small\';
results_path = 'C:\Users\User\Desktop\yotam\MATLAB\Constrain-Based CDDRL\Results\Sachs\threshold\';

db_sizes = [500, 1500, 5000, 15000];
seed = 28;
to_save = 1;
sample = 0;
num_algos = 1;
name_original = 'Sachs_Original';
name_change = 'Sachs_Change';
var_names = {'Akt' 'Erk' 'Jnk' 'Mek' 'P38' 'PIP2' 'PIP3' 'PKA' 'PKC' 'Plcg' 'Raf'};
cvs = 10;

% Load bns
sachs_original = bif2bnt([nets_path char('sachs_original.bif')]);
sachs_change = bif2bnt([nets_path char('sachs_change.bif')]);
% Get DAGs
before = sachs_original.dag;
after = sachs_change.dag;
% Get orderings
order_before = sachs_original.order;
order_after = sachs_change.order;

% Sample from bns
if sample
    sample_from_bn(original_path, sachs_original, db_sizes, var_names, seed, name_original, to_save, cvs);
    sample_from_bn(change_path, sachs_change, db_sizes, var_names, seed, name_change, to_save, cvs);
end

% Result arrays for SHD, runtime and number of CI tests for
% add/remove/reverse and for every df size
SHDs_avg = zeros(num_algos, length(db_sizes));
EEs_avg = zeros(num_algos, length(db_sizes));
EEs2_avg = zeros(num_algos, length(db_sizes));
MEs_avg = zeros(num_algos, length(db_sizes));
MEs2_avg = zeros(num_algos, length(db_sizes));
WDs_avg = zeros(num_algos, length(db_sizes));
WDs4_avg = zeros(num_algos, length(db_sizes));
WDs5_avg = zeros(num_algos, length(db_sizes));
LogLike_avg = zeros(num_algos - 1, length(db_sizes));
run_time_avg = zeros(num_algos, length(db_sizes));
CI_tests_avg = zeros(2, length(db_sizes));
all_dags = cell(num_algos + 1, length(db_sizes) + 1);
all_dags = make_row_col_names(all_dags, db_sizes);
c_nodes = [1, 4, 8, 9];
%c_nodes = [1,2,3,4,5,6,7,8,9,10,11];

for i=1:length(db_sizes)
    
    % SHDs for current df size
    shd = zeros(cvs, num_algos);
    ee = zeros(cvs, num_algos);
    ee2 = zeros(cvs, num_algos);
    me = zeros(cvs, num_algos);
    me2 = zeros(cvs, num_algos);
    wd = zeros(cvs, num_algos);
    wd4 = zeros(cvs, num_algos);
    wd5 = zeros(cvs, num_algos);
    % Log likelihood for current df size
    loglike = zeros(cvs, num_algos);
    % Runtime for current df size
    time = zeros(cvs, num_algos);
    % Number of CI tests for current df size
    CIs = zeros(cvs, 2);
    % Dags for the current data size
    db_dags = cell(num_algos, cvs);
    
    for j=1:cvs
        fprintf('Starting CV %d for db size of: %d\n', j, db_sizes(i));
        samples = num2str(db_sizes(i));
        % Load dfs
        data = table2array(readtable([change_path samples '\df_Sachs_Change' num2str(j) '.csv']));

        % Learn new bns and save run time and number of CI tests
        % Relearn with CB approach
        tic 
        [g_new, CIs(j, 1), dag_steps_list] = CB_Relearning(before, data, c_nodes);
        time(j, 1) = toc;
        % Relearn with S&S approach
        %tic
        %[BestSol, ~, ~] = makeSSCV1_sa(before, data, c_nodes, 0.5, 0, 1, 0);
        %time(j, 2) = toc;
        % Learn structure with MMHC
        %tic
        %dag_mmhc = Causal_Explorer('MMHC',data-1,max(data),'MMHC',[],10,'unif');
        %time(j, 3) = toc;
        %dag_mmhc = pdag_to_dag(dag_mmhc);
        % Learn structure with K2 using ordering of previous dag
        %tic
        %dag_k2_bef = learn_struct_K2(data', max(data), order_before, 'verbose', 'yes');
        %time(j, 4) = toc;
        % Learn structure with K2 using ordering of after dag
        %tic
        %dag_k2_aft = learn_struct_K2(data', max(data), order_after, 'verbose', 'yes');
        %time(j, 5) = toc;
        % Learn structure with PC
        %tic
        %[dag_pc, ~, CIs(j, 2)] = learn_struct_pdag_pc_adaptive(data, 2, 2);
        %time(j, 6) = toc;
        
        % Save dags learned for the current cv and data size
        db_dags{1, j} = double(g_new);
        %db_dags{2, j} = BestSol.g;
        %db_dags{3, j} = dag_mmhc;
        %db_dags{4, j} = dag_k2_bef;
        %db_dags{5, j} = dag_k2_aft;
        %db_dags{6, j} = dag_pc;
        % Compute and save SHD scores
        [shd(j, 1), ee(j, 1), me(j, 1), wd(j, 1)] = SHD_yotam(after, g_new);
        [ee2(j, 1), me2(j, 1), wd4(j, 1), wd5(j, 1)] = CB_SHD_parts(dag_steps_list, after);
        %[shd(j, 2), ee(j, 2), me(j, 2), wd(j, 2)] = SHD_yotam(after, BestSol.g);
        %[shd(j, 3), ee(j, 3), me(j, 3), wd(j, 3)] = SHD_yotam(after, dag_mmhc);
        %[shd(j, 4), ee(j, 4), me(j, 4), wd(j, 4)] = SHD_yotam(after, dag_k2_bef);
        %[shd(j, 5), ee(j, 5), me(j, 5), wd(j, 5)] = SHD_yotam(after, dag_k2_aft);
        %shd(j, 6) = SHDL(dag_to_cpdag(after), prepare_pc_pdag(dag_pc));
        % Compute log likelihoods of current cv
        dag_list = cell(num_algos - 1, 1);
        dag_list{1, 1} = g_new;
        %dag_list{2, 1} = BestSol.g;
        %dag_list{3, 1} = dag_mmhc;
        %dag_list{4, 1} = dag_k2_bef;
        %dag_list{5, 1} = dag_k2_aft;
        %loglike_list = compute_loglike(data, dag_list);
        %for dag=1:num_algos-1
        %    loglike(j, dag) = loglike_list(dag, 1);
        %end
    end
    
    % Save all dags learned for the current data size
    for alg=2:(num_algos+1)
        all_dags{alg, i + 1} = cell(1, cvs);
        for cv=1:cvs
            all_dags{alg, i + 1}{1, cv} = db_dags{alg - 1, cv};
        end
    end
    % Save average shds and run time for all current cvs and df size
    for algo=1:num_algos
        SHDs_avg(algo, i) = mean(shd(:, algo));
        EEs_avg(algo, i) = mean(ee(:, algo));
        EEs2_avg(algo, i) = mean(ee2(:, algo));
        MEs_avg(algo, i) = mean(me(:, algo));
        MEs2_avg(algo, i) = mean(me2(:, algo));
        WDs_avg(algo, i) = mean(wd(:, algo));
        WDs4_avg(algo, i) = mean(wd4(:, algo));
        WDs5_avg(algo, i) = mean(wd5(:, algo));
        if algo ~= num_algos
            LogLike_avg(algo, i) = mean(loglike(:, algo));
        end
        run_time_avg(algo, i) = mean(time(:, algo));
    end
    % Save average number of CI tests for all current cvs and df size
    CI_tests_avg(1, i) = mean(CIs(:, 1));
    %CI_tests_avg(2, i) = mean(CIs(:, 2));
    
    % Save current data size results if asked to
    if to_save
        save([results_path '\DAGs_All.mat'], 'all_dags', '-mat');
        writetable(array2table(SHDs_avg), [results_path '\Sachs_Avg_SHDs.csv']);
        writetable(array2table(EEs_avg), [results_path '\Sachs_Avg_EEs.csv']);
        writetable(array2table(EEs2_avg), [results_path '\Sachs_Avg_EEs2.csv']);
        writetable(array2table(MEs_avg), [results_path '\Sachs_Avg_MEs.csv']);
        writetable(array2table(MEs2_avg), [results_path '\Sachs_Avg_MEs2.csv']);
        writetable(array2table(WDs_avg), [results_path '\Sachs_Avg_WDs.csv']);
        writetable(array2table(WDs4_avg), [results_path '\Sachs_Avg_WDs4.csv']);
        writetable(array2table(WDs5_avg), [results_path '\Sachs_Avg_WDs5.csv']);
        %writetable(array2table(LogLike_avg), [results_path '\Sachs_Avg_LogLike.csv']);
        writetable(array2table(run_time_avg), [results_path '\Sachs_Avg_Runtime.csv']);
        writetable(array2table(CI_tests_avg), [results_path '\Sachs_Avg_CI_Tests.csv']);
    end
end

% Save results
% Transform to table
SHDs_avg = array2table(SHDs_avg);
EEs_avg = array2table(EEs_avg);
EEs2_avg = array2table(EEs2_avg);
MEs_avg = array2table(MEs_avg);
MEs2_avg = array2table(MEs2_avg);
WDs_avg = array2table(WDs_avg);
WDs4_avg = array2table(WDs4_avg);
WDs5_avg = array2table(WDs5_avg);
LogLike_avg = array2table(LogLike_avg);
run_time_avg = array2table(run_time_avg);
CI_tests_avg = array2table(CI_tests_avg);

% Assign columns names
SHDs_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000'};
SHDs_avg.Properties.RowNames = {'CB' 'S&S' 'MMHC' 'K2_bef' 'K2_aft' 'PC'};
EEs_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000'};
EEs_avg.Properties.RowNames = {'CB' 'S&S' 'MMHC' 'K2_bef' 'K2_aft' 'PC'};
EEs2_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000'};
EEs2_avg.Properties.RowNames = {'CB'};
MEs_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000'};
MEs_avg.Properties.RowNames = {'CB' 'S&S' 'MMHC' 'K2_bef' 'K2_aft' 'PC'};
MEs2_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000'};
MEs2_avg.Properties.RowNames = {'CB'};
WDs_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000'};
WDs_avg.Properties.RowNames = {'CB' 'S&S' 'MMHC' 'K2_bef' 'K2_aft' 'PC'};
WDs4_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000'};
WDs4_avg.Properties.RowNames = {'CB'};
WDs5_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000'};
WDs5_avg.Properties.RowNames = {'CB'};
LogLike_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000'};
LogLike_avg.Properties.RowNames = {'CB' 'S&S' 'MMHC' 'K2_bef' 'K2_aft'};
run_time_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000'};
run_time_avg.Properties.RowNames = {'CB' 'S&S' 'MMHC' 'K2_bef' 'K2_aft' 'PC'};
CI_tests_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000'};
CI_tests_avg.Properties.RowNames = {'CB' 'PC'};

% Save final results if asked to
if to_save
    writetable(SHDs_avg, [results_path '\Sachs_Avg_SHDs.csv'], 'WriteRowNames', true);
    writetable(EEs_avg, [results_path '\Sachs_Avg_EEs.csv'], 'WriteRowNames', true);
    writetable(EEs2_avg, [results_path '\Sachs_Avg_EEs2.csv'], 'WriteRowNames', true);
    writetable(MEs_avg, [results_path '\Sachs_Avg_MEs.csv'], 'WriteRowNames', true);
    writetable(MEs2_avg, [results_path '\Sachs_Avg_MEs2.csv'], 'WriteRowNames', true);
    writetable(WDs_avg, [results_path '\Sachs_Avg_WDs.csv'], 'WriteRowNames', true);
    writetable(WDs4_avg, [results_path '\Sachs_Avg_WDs4.csv'], 'WriteRowNames', true);
    writetable(WDs5_avg, [results_path '\Sachs_Avg_WDs5.csv'], 'WriteRowNames', true);
    writetable(LogLike_avg, [results_path '\Sachs_Avg_LogLike.csv'], 'WriteRowNames', true);
    writetable(run_time_avg, [results_path '\Sachs_Avg_Runtime.csv'], 'WriteRowNames', true);
    writetable(CI_tests_avg, [results_path '\Sachs_Avg_CI_Tests.csv'], 'WriteRowNames', true);
end


