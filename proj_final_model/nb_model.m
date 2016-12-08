function [ mdl ] = nb_model(X, Y, index)
% using the training data X and Y to generate Naive Bayes classification
% model
    
    X_presence = full(X);
    X_presence(X_presence ~= 0) = 1;
    X_MI = full(X_presence(:, index));
    Y_MI = full(Y);
    mdl = fitcnb(X_MI,Y_MI,'Distribution','mn');
end