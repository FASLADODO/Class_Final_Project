clear
clc

load ../final_project_kit/train_set/words_train.mat;
[chi, df] = chi2feature(X,Y);

counter=0;
res = zeros(1,100);
% for per = linspace(0.01,0.025)
% change the percentage here before max(chi) to decide how many features to keep
% it seems the precision changes every time I ran this though
ind = chi>0.0109*max(chi);
X_hat = X(:,ind);
save('chi_feature.mat', 'X_hat');
N = size(X_hat, 1);
K  = 10;
Indices = crossvalind('Kfold', N, K);
Y = full(Y);
precision = 0;
threshold = 0.6;
for k = 1:K
    X_train = X_hat(Indices ~= k, :);
    X_test = X_hat(Indices == k, :);
    Y_train = Y(Indices ~= k);
    Y_test = Y(Indices == k);
    mdl = fitrlinear(X_train, Y_train, 'Regularization', 'lasso', 'Solver', 'sparsa');
    YHat = predict(mdl,X_test);
    YHat(YHat > threshold) = 1;
    YHat(~(YHat > threshold)) = 0;
    precision = precision + mean(YHat == Y_test);
end
counter = counter+1;
precision = precision / K;
% end