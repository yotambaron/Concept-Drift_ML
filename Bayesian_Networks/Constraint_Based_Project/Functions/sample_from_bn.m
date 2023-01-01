function [] = sample_from_bn(save_path,bn,db_sizes,var_names,seed,name,save,cvs)

for i=1:length(db_sizes)
    for j=1:cvs
        rng(seed);
        % Create simulated df
        samples = db_sizes(i);
        db_simulated = cell(samples, size(bn.dag, 1));
    
        % Sample bnet
        for k=1:samples
            db_simulated(k, :) = sample_bnet(bn);
        end
    
        % Transform the dfs to tables
        db_simulated = array2table(db_simulated);
    
        % Set variables' names
        db_simulated.Properties.VariableNames = var_names;

        % Save dfs
        if save
            writetable(db_simulated,[save_path '\' num2str(samples) '\df_' name num2str(j) '.csv']);
         end
        seed = seed + 1;
    end
end

end


