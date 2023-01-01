function [ loglike_list ] = compute_loglike( data, dag_list )

% This function recieves data and a list of dags, it learns parameters of
% each dag and computes its loglikelihood using the inputed data.

loglike_list = zeros(length(dag_list), 1);

% Loop through all the inputed dags
for dag=1:length(dag_list)
    
    % Create bn with and learn parameters with data
    bnet = mk_bnet(dag_list{dag, 1}, max(data));

    for n=1:length(max(data))
        bnet.CPD{n} = tabular_CPD(bnet, n);
    end
    bnet = learn_params(bnet, data');
    
    % Calculate the log likelihood of each sample and save it
    engine = jtree_inf_engine(bnet);
    for row=1:size(data, 1)
        evidence = data(row, :);
        evidence = num2cell(evidence);
        [~, ll] = enter_evidence(engine, evidence);
        loglike_list(dag, 1) = loglike_list(dag, 1) + ll;
    end

end

