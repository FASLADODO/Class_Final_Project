function [ Y_hat ] = svm_predict(mdl, X)
% Using the naive bayes model to predict the testing data X label
    load NB_index.mat;
    
    X_presence = full(X);
    X_presence(X_presence ~= 0) = 1;
    X_MI = X_presence(:, index);
    
    Y_Prob = predict(mdl,X_MI);
    Y_hat = zeros(length(Y_Prob), 1);
    Y_hat(Y_Prob > 0.59) = 1;
end