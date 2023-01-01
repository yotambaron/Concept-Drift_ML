clear all;

% Simulate and save data bases of all sizes for all bns changes
dbs_path = 'C:\Users\User\Desktop\yotam\MATLAB\Constrain-Based CDDRL\DBs\Asia\';
nets_path = 'C:\Users\User\Desktop\yotam\MATLAB\Constrain-Based CDDRL\Networks\Small\';
results_path = 'C:\Users\User\Desktop\yotam\MATLAB\Constrain-Based CDDRL\Results\Asia\threshold\';

db_sizes = [500, 1500, 5000, 15000];
seed = 28;
to_save = 1;
sample = 0;
num_algos = 6;
cvs = 10;

if sample
    [~, ~, ~, ~] = sample_asia_bn(dbs_path, nets_path, db_sizes, seed, to_save, cvs);
end

% Set parameters
% Changed nodes
c_nodes_add = 2;
c_nodes_remove = 8;
c_nodes_reverse = [5, 8];
% Result arrays for SHD, runtime and number of CI tests for
% add/remove/reverse and for every df size
SHDs_avg = zeros(3 * num_algos, length(db_sizes));
run_time_avg = zeros(3 * num_algos, length(db_sizes));
CI_tests_avg = zeros(6, length(db_sizes));

% Load bns
asia = bif2bnt([nets_path 'asia.bif']);
asia_add = bif2bnt([nets_path 'asia_add.bif']);
asia_remove = bif2bnt([nets_path 'asia_remove.bif']);
asia_reverse = bif2bnt([nets_path 'asia_reverse.bif']);
% Get DAGs
before = asia.dag;
after_add = asia_add.dag;
after_remove = asia_remove.dag;
after_reverse = asia_reverse.dag;
% Get orderings
order_before = asia.order;
order_after_add = asia_add.order;
order_after_remove = asia_remove.order;
order_after_reverse = asia_reverse.order;

for i=1:length(db_sizes)
    
    % SHDs for current df size
    shd_add = zeros(cvs, num_algos);
    shd_remove = zeros(cvs, num_algos);
    shd_reverse = zeros(cvs, num_algos);
    
    % Runtime for current df size
    time_add = zeros(cvs, num_algos);
    time_remove = zeros(cvs, num_algos);
    time_reverse = zeros(cvs, num_algos);
    
    % Number of CI tests for current df size
    CIs_add = zeros(cvs, 2);
    CIs_remove = zeros(cvs, 2);
    CIs_reverse = zeros(cvs, 2);
    
    for j=1:cvs
        samples = num2str(db_sizes(i));
        % Load dfs
        data_add = table2array(readtable([dbs_path '\Add\' samples '\df_asia_add' num2str(j) '.csv']));
        data_remove = table2array(readtable([dbs_path '\Remove\' samples '\df_asia_remove' num2str(j) '.csv']));
        data_reverse = table2array(readtable([dbs_path '\Reverse\' samples '\df_asia_reverse' num2str(j) '.csv']));
        
        % Learn new bns and save run time and number of CI tests
        % Relearn using CB
        tic
        [g_new_add, CIs_add(j, 1)] = CB_Relearning(before, data_add, c_nodes_add);
        time_add(j, 1) = toc;
        tic
        [g_new_remove, CIs_remove(j, 1)] = CB_Relearning(before, data_remove, c_nodes_remove);
        time_remove(j, 1) = toc;
        tic
        [g_new_reverse, CIs_reverse(j, 1)] = CB_Relearning(before, data_reverse, c_nodes_reverse);
        time_reverse(j, 1) = toc;
        
        % Relearn using S&S
        tic
        [BestSol_add, ~, ~] = makeSSCV1_sa(before, data_add, c_nodes_add, 0.5, 0, 1, 0);
        time_add(j, 2) = toc;
        tic
        [BestSol_remove, ~, ~] = makeSSCV1_sa(before, data_remove, c_nodes_remove, 0.5, 0, 1, 0);
        time_remove(j, 2) = toc;
        tic
        [BestSol_reverse, ~, ~] = makeSSCV1_sa(before, data_reverse, c_nodes_reverse, 0.5, 0, 1, 0);
        time_reverse(j, 2) = toc;
        
        % Learn structure with MMHC
        tic
        mmhc_add = Causal_Explorer('MMHC',data_add-1,max(data_add),'MMHC',[],10,'unif');
        time_add(j, 3) = toc;
        mmhc_add = pdag_to_dag(mmhc_add);
        tic
        mmhc_remove = Causal_Explorer('MMHC',data_remove-1,max(data_remove),'MMHC',[],10,'unif');
        time_remove(j, 3) = toc;
        mmhc_remove = pdag_to_dag(mmhc_remove);
        tic
        mmhc_reverse = Causal_Explorer('MMHC',data_reverse-1,max(data_reverse),'MMHC',[],10,'unif');
        time_reverse(j, 3) = toc;
        mmhc_reverse = pdag_to_dag(mmhc_reverse);
        
        % Learn structure with K2 using ordering of previous dag
        tic
        k2_bef_add = learn_struct_K2(data_add', max(data_add), order_before, 'verbose', 'yes');
        time_add(j, 4) = toc;
        tic
        k2_bef_remove = learn_struct_K2(data_remove', max(data_remove), order_before, 'verbose', 'yes');
        time_remove(j, 4) = toc;
        tic
        k2_bef_reverse = learn_struct_K2(data_reverse', max(data_reverse), order_before, 'verbose', 'yes');
        time_reverse(j, 4) = toc;
        
        % Learn structure with K2 using ordering of after dag
        tic
        k2_aft_add = learn_struct_K2(data_add', max(data_add), order_after_add, 'verbose', 'yes');
        time_add(j, 5) = toc;
        tic
        k2_aft_remove = learn_struct_K2(data_remove', max(data_remove), order_after_remove, 'verbose', 'yes');
        time_remove(j, 5) = toc;
        tic
        k2_aft_reverse = learn_struct_K2(data_reverse', max(data_reverse), order_after_reverse, 'verbose', 'yes');
        time_reverse(j, 5) = toc;
        
        % Learn structure with PC
        tic
        [pc_add, ~, CIs_add(j, 2)] = learn_struct_pdag_pc_adaptive(data_add, 2, 2);
        time_add(j, 6) = toc;
        tic
        [pc_remove, ~, CIs_remove(j, 2)] = learn_struct_pdag_pc_adaptive(data_remove, 2, 2);
        time_remove(j, 6) = toc;
        tic
        [pc_reverse, ~, CIs_reverse(j, 2)] = learn_struct_pdag_pc_adaptive(data_reverse, 2, 2);
        time_reverse(j, 6) = toc;
        
        % Compute and save SHD scores
        % Compute CB
        shd_add(j, 1) = SHD(after_add, g_new_add);
        shd_remove(j, 1) = SHD(after_remove, g_new_remove);
        shd_reverse(j, 1) = SHD(after_reverse, g_new_reverse);
        % Compute S&S
        shd_add(j, 2) = SHD(after_add, BestSol_add.g);
        shd_remove(j, 2) = SHD(after_remove, BestSol_remove.g);
        shd_reverse(j, 2) = SHD(after_reverse, BestSol_reverse.g);
        % Compute MMHC
        shd_add(j, 3) = SHD(after_add, mmhc_add);
        shd_remove(j, 3) = SHD(after_remove, mmhc_remove);
        shd_reverse(j, 3) = SHD(after_reverse, mmhc_reverse);
        % Compute K2 with before ordering
        shd_add(j, 4) = SHD(after_add, k2_bef_add);
        shd_remove(j, 4) = SHD(after_remove, k2_bef_remove);
        shd_reverse(j, 4) = SHD(after_reverse, k2_bef_reverse);
        % Compute K2 with after ordering
        shd_add(j, 5) = SHD(after_add, k2_aft_add);
        shd_remove(j, 5) = SHD(after_remove, k2_aft_remove);
        shd_reverse(j, 5) = SHD(after_reverse, k2_aft_reverse);
        % Compute PC
        shd_add(j, 6) = SHDL(dag_to_cpdag(after_add), prepare_pc_pdag(pc_add));
        shd_remove(j, 6) = SHDL(dag_to_cpdag(after_remove), prepare_pc_pdag(pc_remove));
        shd_reverse(j, 6) = SHDL(dag_to_cpdag(after_reverse), prepare_pc_pdag(pc_reverse));
    end
    
    % Save average shds and run time for all current cvs and df size
    for k=1:num_algos
        SHDs_avg(k * 3 - 2, i) = mean(shd_add(:, k));
        SHDs_avg(k * 3 - 1, i) = mean(shd_remove(:, k));
        SHDs_avg(k * 3, i) = mean(shd_reverse(:, k));
        
        run_time_avg(k * 3 - 2, i) = mean(time_add(:, k));
        run_time_avg(k * 3 - 1, i) = mean(time_remove(:, k));
        run_time_avg(k * 3, i) = mean(time_reverse(:, k));
    end
    
    % Save average number of CI tests for all current cvs and df size
    CI_tests_avg(1, i) = mean(CIs_add(:, 1));
    CI_tests_avg(2, i) = mean(CIs_remove(:, 1));
    CI_tests_avg(3, i) = mean(CIs_reverse(:, 1));
    CI_tests_avg(4, i) = mean(CIs_add(:, 2));
    CI_tests_avg(5, i) = mean(CIs_remove(:, 2));
    CI_tests_avg(6, i) = mean(CIs_reverse(:, 2));
end

% Save results
% Transform to table
SHDs_avg = array2table(SHDs_avg);
run_time_avg = array2table(run_time_avg);
CI_tests_avg = array2table(CI_tests_avg);
% Assign columns and rows names
SHDs_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000'};
SHDs_avg.Properties.RowNames = {'Add_edge_CB' 'Remove_edge_CB' 'Reverse_edge_CB' 'Add_edge_S&S' 'Remove_edge_S&S' 'Reverse_edge_S&S' 'Add_edge_MMHC' 'Remove_edge_MMHC' 'Reverse_edge_MMHC' 'Add_edge_K2_bef' 'Remove_edge_K2_bef' 'Reverse_edge_K2_bef' 'Add_edge_K2_aft' 'Remove_edge_K2_aft' 'Reverse_edge_K2_aft' 'Add_edge_PC' 'Remove_edge_PC' 'Reverse_edge_PC'};
run_time_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000'};
run_time_avg.Properties.RowNames = {'Add_edge_CB' 'Remove_edge_CB' 'Reverse_edge_CB' 'Add_edge_S&S' 'Remove_edge_S&S' 'Reverse_edge_S&S' 'Add_edge_MMHC' 'Remove_edge_MMHC' 'Reverse_edge_MMHC' 'Add_edge_K2_bef' 'Remove_edge_K2_bef' 'Reverse_edge_K2_bef' 'Add_edge_K2_aft' 'Remove_edge_K2_aft' 'Reverse_edge_K2_aft' 'Add_edge_PC' 'Remove_edge_PC' 'Reverse_edge_PC'};
CI_tests_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000'};
CI_tests_avg.Properties.RowNames = {'Add_edge_CB' 'Remove_edge_CB' 'Reverse_edge_CB' 'Add_edge_PC' 'Remove_edge_PC' 'Reverse_edge_PC'};
% Save if asked to
if to_save
    writetable(SHDs_avg, [results_path '\Asia_Avg_SHDs.csv'], 'WriteRowNames', true);
    writetable(run_time_avg, [results_path '\Asia_Avg_Runtime.csv'], 'WriteRowNames', true);
    writetable(CI_tests_avg, [results_path '\Asia_Avg_CI_Tests.csv'], 'WriteRowNames', true);
end
