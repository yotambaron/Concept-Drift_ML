clear all;

% Simulate and save data bases of all sizes for all bns changes
original_path = 'C:\Users\User\Desktop\yotam\MATLAB\Constrain-Based CDDRL\DBs\Andes\Original\';
change_path = 'C:\Users\User\Desktop\yotam\MATLAB\Constrain-Based CDDRL\DBs\Andes\Change\';
nets_path = 'C:\Users\User\Desktop\yotam\MATLAB\Constrain-Based CDDRL\Networks\Large\';
results_path = 'C:\Users\User\Desktop\yotam\MATLAB\Constrain-Based CDDRL\Results\Andes\threshold\';

db_sizes = [500, 1500, 5000, 15000, 50000];
seed = 28;
to_save = 1;
sample = 0;
num_algos = 1;
name_original = 'Andes_Original';
name_change = 'Andes_Change';
var_names = {'GOAL_2' 'SNode_3' 'SNode_4' 'SNode_5' 'SNode_6' 'SNode_7' 'DISPLACEM0' 'RApp1' 'GIVEN_1' 'RApp2' 'SNode_8' 'SNode_9' 'SNode_10' 'SNode_11' 'SNode_12' 'SNode_13' 'SNode_14' 'SNode_15' 'SNode_16' 'SNode_17' 'SNode_18' 'SNode_19' 'NEED1' 'SNode_20' 'GRAV2' 'SNode_21' 'VALUE3' 'SNode_24' 'SLIDING4' 'SNode_25' 'CONSTANT5' 'SNode_26' 'KNOWN6' 'VELOCITY7' 'SNode_47' 'RApp3' 'KNOWN8' 'RApp4' 'SNode_27' 'COMPO16' 'GOAL_48' 'TRY12' 'TRY11' 'GOAL_49' 'CHOOSE19' 'GOAL_50' 'SYSTEM18' 'SNode_51' 'KINEMATI17' 'SNode_52' 'IDENTIFY10' 'GOAL_53' 'IDENTIFY9' 'SNode_28' 'TRY13' 'TRY14' 'TRY15' 'VAR20' 'SNode_29' 'SNode_31' 'GIVEN21' 'SNode_33' 'SNode_34' 'VECTOR27' 'APPLY32' 'GOAL_56' 'CHOOSE35' 'GOAL_57' 'MAXIMIZE34' 'SNode_59' 'AXIS33' 'SNode_60' 'WRITE31' 'GOAL_61' 'WRITE30' 'GOAL_62' 'RESOLVE37' 'GOAL_63' 'NEED36' 'SNode_64' 'SNode_41' 'SNode_42' 'IDENTIFY39' 'SNode_43' 'RESOLVE38' 'GOAL_66' 'SNode_67' 'IDENTIFY41' 'SNode_54' 'RESOLVE40' 'GOAL_69' 'SNode_70' 'IDENTIFY43' 'SNode_55' 'RESOLVE42' 'GOAL_72' 'SNode_73' 'KINE29' 'SNode_74' 'VECTOR44' 'SNode_75' 'EQUATION28' 'GOAL_79' 'RApp5' 'GOAL_80' 'RApp6' 'GOAL_81' 'TRY25' 'TRY24' 'GOAL_83' 'CHOOSE47' 'GOAL_84' 'SYSTEM46' 'SNode_86' 'NEWTONS45' 'SNode_156' 'DEFINE23' 'GOAL_98' 'IDENTIFY22' 'SNode_37' 'TRY26' 'SNode_38' 'SNode_40' 'SNode_44' 'SNode_46' 'NULL48' 'SNode_65' 'SNode_68' 'SNode_71' 'FIND49' 'GOAL_87' 'NORMAL50' 'SNode_88' 'STRAT_90' 'NORMAL52' 'INCLINE51' 'SNode_91' 'HORIZ53' 'BUGGY54' 'SNode_92' 'IDENTIFY55' 'SNode_93' 'WEIGHT56' 'SNode_94' 'WEIGHT57' 'SNode_95' 'SNode_97' 'FIND58' 'GOAL_99' 'IDENTIFY59' 'SNode_100' 'FORCE60' 'SNode_102' 'APPLY61' 'GOAL_103' 'CHOOSE62' 'GOAL_104' 'SNode_106' 'SNode_152' 'WRITE63' 'GOAL_107' 'WRITE64' 'GOAL_108' 'GOAL_109' 'GOAL65' 'GOAL_110' 'GOAL66' 'GOAL_111' 'NEED67' 'RApp7' 'RApp8' 'SNode_112' 'GOAL68' 'GOAL_113' 'GOAL_114' 'SNode_115' 'VECTOR69' 'SNode_116' 'SNode_117' 'VECTOR70' 'SNode_118' 'EQUAL71' 'SNode_119' 'SNode_120' 'GOAL72' 'GOAL_121' 'SNode_122' 'VECTOR73' 'SNode_123' 'NEWTONS74' 'SNode_124' 'SUM75' 'SNode_125' 'GOAL_126' 'GOAL_127' 'RApp9' 'RApp10' 'SNode_128' 'GOAL_129' 'GOAL_130' 'SNode_131' 'SNode_132' 'SNode_133' 'SNode_134' 'SNode_135' 'SNode_154' 'SNode_136' 'SNode_137' 'GOAL_142' 'GOAL_143' 'GOAL_146' 'RApp11' 'RApp12' 'RApp13' 'GOAL_147' 'TRY76' 'GOAL_149' 'APPLY77' 'GOAL_150' 'GRAV78' 'SNode_151' 'GOAL_153' 'SNode_155'};
cvs = 10;

% Load bns
andes_original = bif2bnt([nets_path char('andes_original.bif')]);
andes_change = bif2bnt([nets_path char('andes_change.bif')]);
% Get DAGs
before = andes_original.dag;
after = andes_change.dag;
% Get orderings
order_before = andes_original.order;
order_after = andes_change.order;

% Sample from bns
if sample
    sample_from_bn(original_path, andes_original, db_sizes, var_names, seed, name_original, to_save, cvs);
    sample_from_bn(change_path, andes_change, db_sizes, var_names, seed, name_change, to_save, cvs);
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
c_nodes = [8, 42, 44, 55, 66, 81, 82, 99, 103, 135, 136, 146, 172, 193, 210, 213];
%c_nodes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223];

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
        data = table2array(readtable([change_path samples '\df_Andes_Change' num2str(j) '.csv']));

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
        writetable(array2table(SHDs_avg), [results_path '\Andes_Avg_SHDs.csv']);
        writetable(array2table(EEs_avg), [results_path '\Andes_Avg_EEs.csv']);
        writetable(array2table(EEs2_avg), [results_path '\Andes_Avg_EEs2.csv']);
        writetable(array2table(MEs_avg), [results_path '\Andes_Avg_MEs.csv']);
        writetable(array2table(MEs2_avg), [results_path '\Andes_Avg_MEs2.csv']);
        writetable(array2table(WDs_avg), [results_path '\Andes_Avg_WDs.csv']);
        writetable(array2table(WDs4_avg), [results_path '\Andes_Avg_WDs4.csv']);
        writetable(array2table(WDs5_avg), [results_path '\Andes_Avg_WDs5.csv']);
        %writetable(array2table(LogLike_avg), [results_path '\Andes_Avg_LogLike.csv']);
        writetable(array2table(run_time_avg), [results_path '\Andes_Avg_Runtime.csv']);
        writetable(array2table(CI_tests_avg), [results_path '\Andes_Avg_CI_Tests.csv']);
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
SHDs_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000' 'Obs_50000'};
SHDs_avg.Properties.RowNames = {'CB' 'S&S' 'MMHC' 'K2_bef' 'K2_aft' 'PC'};
EEs_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000' 'Obs_50000'};
EEs_avg.Properties.RowNames = {'CB' 'S&S' 'MMHC' 'K2_bef' 'K2_aft' 'PC'};
EEs2_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000' 'Obs_50000'};
EEs2_avg.Properties.RowNames = {'CB'};
MEs_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000' 'Obs_50000'};
MEs_avg.Properties.RowNames = {'CB' 'S&S' 'MMHC' 'K2_bef' 'K2_aft' 'PC'};
MEs2_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000' 'Obs_50000'};
MEs2_avg.Properties.RowNames = {'CB'};
WDs_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000' 'Obs_50000'};
WDs_avg.Properties.RowNames = {'CB' 'S&S' 'MMHC' 'K2_bef' 'K2_aft' 'PC'};
WDs4_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000' 'Obs_50000'};
WDs4_avg.Properties.RowNames = {'CB'};
WDs5_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000' 'Obs_50000'};
WDs5_avg.Properties.RowNames = {'CB'};
LogLike_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000' 'Obs_50000'};
LogLike_avg.Properties.RowNames = {'CB' 'S&S' 'MMHC' 'K2_bef' 'K2_aft'};
run_time_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000' 'Obs_50000'};
run_time_avg.Properties.RowNames = {'CB' 'S&S' 'MMHC' 'K2_bef' 'K2_aft' 'PC'};
CI_tests_avg.Properties.VariableNames = {'Obs_500' 'Obs_1500' 'Obs_5000' 'Obs_15000' 'Obs_50000'};
CI_tests_avg.Properties.RowNames = {'CB' 'PC'};Andes
% Save final results if asked to
if to_save
    writetable(SHDs_avg, [results_path '\Andes_Avg_SHDs.csv'], 'WriteRowNames', true);
    writetable(EEs_avg, [results_path '\Andes_Avg_EEs.csv'], 'WriteRowNames', true);
    writetable(EEs2_avg, [results_path '\Andes_Avg_EEs2.csv'], 'WriteRowNames', true);
    writetable(MEs_avg, [results_path '\Andes_Avg_MEs.csv'], 'WriteRowNames', true);
    writetable(MEs2_avg, [results_path '\Andes_Avg_MEs2.csv'], 'WriteRowNames', true);
    writetable(WDs_avg, [results_path '\Andes_Avg_WDs.csv'], 'WriteRowNames', true);
    writetable(WDs4_avg, [results_path '\Andes_Avg_WDs4.csv'], 'WriteRowNames', true);
    writetable(WDs5_avg, [results_path '\Andes_Avg_WDs5.csv'], 'WriteRowNames', true);
    writetable(LogLike_avg, [results_path '\Andes_Avg_LogLike.csv'], 'WriteRowNames', true);
    writetable(run_time_avg, [results_path '\Andes_Avg_Runtime.csv'], 'WriteRowNames', true);
    writetable(CI_tests_avg, [results_path '\Andes_Avg_CI_Tests.csv'], 'WriteRowNames', true);
end


