function [ mdl ] = knn_model(X, Y, coeff)
    % using the training data X and Y to generate KNN classification
    % model
    X = full(X);
    Y = full(Y);
    X0 = bsxfun(@minus, X, mean(X,1));
    X_hat = X0*coeff;
    mdl = fitcknn(X_hat, Y);
    mdl.BreakTies = 'nearest';
    mdl.IncludeTies = false;
    mdl.Distance = 'correlation';
    mdl.NumNeighbors = 50;
end