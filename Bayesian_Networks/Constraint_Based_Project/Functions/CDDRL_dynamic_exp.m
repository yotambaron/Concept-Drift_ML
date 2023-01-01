function [C_nodes, BNlist, test_num] = CDDRL_dynamic_exp(cb,BN0,DB,window,jump,Dist_metric,Spc_Test,alpha,my,complexity,recures,stable,lambda,tabu_flag,tabu_size,robust_flag,cpts,score)
 
rng(28);
ns = max(DB);
%%STAGE 2-Detect changes

test_num = 0;
Dag = BN0.dag;
N = size(Dag,1);
DB_stable = DB(1:stable,:);
[mus,stds] = Compute_Means_Sds(DB_stable,BN0,window,jump,Dist_metric);

switch Spc_Test,
    case 'Shewhart_mean',func=str2func('Shewhart_mean_cddrl');
    case 'Shewhart_max',func=str2func('Shewhart_max_cddrl');
    case 'Shewhart_maxK',func=str2func('Shewhart_maxk_cddrl');
    case 'Shewhart_Wmean',func=str2func('Shewhart_Wmean_cddrl');    
    case 'EWMA_mean',func=str2func('EWMA_mean_cddrl');
    case 'EWMA_max',func=str2func('EWMA_max_cddrl');
    case 'EWMA_maxK',func=str2func('EWMA_maxk_cddrl');
    case 'EWMA_Wmean',func=str2func('EWMA_Wmean_cddrl');
    %case 'Hotelling', func=str2func('detectChanged_Hotelling');
    %case 'MEWMA', func=str2func('detectChanged_MEWMA');    
    otherwise, error('invalid input');       
end

%%Stage 3-ReLearing
       
T_vals = window:jump:size(DB,1);  % The time stamps in the database
Ntests = length(T_vals);
BNlist = cell(1,Ntests);
obs_to_add = window-jump;
Dist_Array = zeros(Ntests,N);
UCL_Array = zeros(Ntests,N);
[~, initial_score]  = bdeuFirst(DB_stable, Dag, ns); % Score the first BN
Bestsol = cell(Ntests,1);
Z=zeros(2,N); % Save all past distances for every node separately

for p = 1:Ntests
    p
    c_nodes = [];
    if p == 1
        temp_DB = DB(1:T_vals(p),:);
        BN_normal = BN0;
        temp_test = test_change_1Metric_dynamic(temp_DB,BN_normal,Dist_metric); % Get the distances vector of each node
    else
        temp_DB = DB((T_vals(p-1)+1-obs_to_add):T_vals(p),:);
        temp_test = test_change_1Metric_dynamic(temp_DB,BN_normal,Dist_metric); % Get the distances vector of each node
    end
    if strcmp(Spc_Test,'EWMA_Wmean')||strcmp(Spc_Test,'Shewhart_Wmean')
        [Change_indices,Z,UCL] = func(temp_test,BN_normal,lambda,mus,stds,Z); % Check which nodes were changed in the current timestamp
        Dist_Array(p,:) = Z(2,:);
        UCL_Array(p,:) = UCL(1,:);
    else
        [Change_indices] = func(temp_test,mus,stds);
    end

    
    for j=1:N
        if Change_indices{1,j}==1
            c_nodes(end+1) = j;
            Z(:,j)=0; % Zero the distance of the the changed node
        end
    end
    
    C_nodes{p,1} = c_nodes;
    if (~isempty(c_nodes))
        if cb
            [new_g, tests] = CB_Relearning(Dag, temp_DB, c_nodes);
            test_num = test_num + tests;
            bnet = mk_bnet(new_g, ns);
            for n=1:length(ns)
               bnet.CPD{n} = tabular_CPD(bnet, n);
            end
        bnet = learn_params(bnet,temp_DB');
        
        else
        if p == 1
            if (tabu_flag)
                [BestSol,~,cpts] = makeSSCV1_sa_Tabu(Dag,temp_DB,c_nodes,alpha,my,complexity,recures,tabu_size,cpts,score);
            else 
                if (robust_flag)
                    [BestSol,~,cpts] = makeSSCV1_sa_Robust(Dag,temp_DB,c_nodes,alpha,my,complexity,recures,cpts,score);
                else
                    [BestSol,~,cpts] = makeSSCV1_sa(Dag,temp_DB,c_nodes,alpha,my,complexity,recures,cpts,score);
                end
            end
        else
            if (tabu_flag)
                [BestSol,~,cpts] = makeSSCV1_sa_Tabu(Dag,temp_DB,c_nodes,alpha,my,complexity,recures,tabu_size);
            else
                if (robust_flag)
                    [BestSol,~,cpts] = makeSSCV1_sa_Robust(Dag,temp_DB,c_nodes,alpha,my,complexity,recures);
                %%%%%[BestSol,itarations,cpts] = makeSSCV1_sa(Dag,temp_DB,extended_temp_DB,c_nodes,alpha,my,complexity,recures);
                else
                    [BestSol,~,cpts] = makeSSCV1_sa(Dag,temp_DB,c_nodes,alpha,my,complexity,recures);
                end
            end
        end
        
        %%%Assign BN object after each iteration
        bnet = mk_bnet(BestSol.g, ns);
        x = BestSol.s;
        Bestsol{p,1} = x;
        
        %%%Make CPTs the right size - if another category was added to the node
        CPT = cpts(1,:);
        for i=1:size(CPT,2)
            while size(CPT{i},2)<ns(i)
                new_col = zeros(size(CPT{i},1),1);
                CPT{i} = horzcat(CPT{i},new_col);
            end
        end
        %%%Make CPTs the right size - if another category was added to the node's parents
        for i=1:size(CPT,2)
            vec = ns'.*bnet.dag(:,i);
            vec = vec(vec>0);
            while size(CPT{i},1)<prod(vec)
                new_row = zeros(1,ns(i));
                CPT{i} = vertcat(CPT{i},new_row);
            end
        end
        %%%Make counts of occurences in the data into probabilities
        for i=1:size(CPT,1)
            for j=1:size(CPT,2)
                for k=1:size(CPT{i,j},1)
                    CPT{i,j}(k,:)=CPT{i,j}(k,:)/sum(CPT{i,j}(k,:));
                end
            end
        end
        %%%Replace devision by 0 (in the CPTs) to a value of 0
        for i=1:size(CPT,2)
            for j=1:size(CPT{i},1)
                for k=1:size(CPT{i},2)
                    if isnan(CPT{i}(j,k))
                        CPT{i}(j,k) = 0;
                    end
                end
            end
        end
        %%% Save the corrected CPTs to the bnet    
        for n=1:length(ns)
            bnet.CPD{n} = tabular_CPD(bnet, n, [CPT{1,n}]);
        end
        end
        
        
        BNlist{1,p} = bnet; % Save the best current BN structure
        
        if p==1
            BN_normal = BN0;
        else
            BN_normal = BNlist{1,p};
            %BN_normal = learn_params(BN_normal,learn_params_DB');
        end
        
        Dag = BN_normal.dag;
        
    else % No nodes were changed
        BNlist{1,p}=BN_normal;
        if p==1
            Bestsol{p,1} = initial_score;
        else
            Bestsol{p,1} = Bestsol{p-1,1};
        end
        %itarations = 0;
        %cpts = [];
    end  
end
test_num = test_num / Ntests;
