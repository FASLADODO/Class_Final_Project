function [ Y_hat ] = nb_predict(mdl, X)
% Using the naive bayes model to predict the testing data X label
    load NB_index.mat;
    
    X_presence = full(X);
    X_presence(X_presence ~= 0) = 1;
    X_MI = X_presence(:, index);
    
    [Y_hat,Posterior,Cost] = predict(mdl,X_MI);
end