function [Y_hat] = KNN_predict(mdl, X, coeff)

    X = full(X);
    X0 = bsxfun(@minus, X, mean(X,1));
    X_hat = X0*coeff;
    Y_hat = predict(mdl, X_hat);
end