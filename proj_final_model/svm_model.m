function [ mdl ] = svm_model(X, Y, index)
    % using the training data X and Y to generate SVM classification
    % model
    X_presence = full(X);
    X_presence(X_presence ~= 0) = 1;
    X_MI = full(X_presence(:, index));
    Y_MI = full(Y);
    mdl = fitrlinear(X_MI, Y_MI, 'Regularization', 'lasso', 'Solver', 'sparsa');
end