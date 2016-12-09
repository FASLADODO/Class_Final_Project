function [ Y_hat ] = svm_predict(mdl, X)
% Using the naive bayes model to predict the testing data X label
    load SVM_index.mat;
    
    X_presence = full(X);
    X_presence(X_presence ~= 0) = 1;
    X_MI = X_presence(:, index);
    
    Y_Prob = predict(mdl,X_MI);
    Y_hat = zeros(length(Y_Prob), 1);
    ranges = linspace(0,1,10000);
    threshold = ranges(max(find(arrayfun(@(x) mean(Y_Prob > x) > 0.556,ranges))));
    Y_hat(Y_Prob > threshold) = 1;
end