%% Preparation data
clear all
addpath('four_model_project_code');
% load final_project_kit/train_set/raw_tweets_train.mat
load ../final_project_kit/train_set/words_train.mat
% load ../final_project_kit/train_set/train_raw_img.mat
% load final_project_kit/train_set/train_cnn_feat.mat
% load final_project_kit/train_set/train_img_prob.mat
% load final_project_kit/train_set/train_color.mat
% load final_project_kit/train_set/train_tweet_id_img.mat
%% display image
image(reshape_img(train_img(5, :)));
%% Characteristic of X
perc_non_zero_element = nnz(X) / 4500 / 10000 * 100;
%% Covariance & variance of origianl matrix
% (very slow, no need to run, need it to seek if the SVD has the same variance)
% http://stats.stackexchange.com/questions/22569/pca-and-proportion-of-variance-explained
cov_train_X = cov(X);
variance = trace(cov_train_X);
%% PCA with 90% reconstruction accuracy ## need 763 PCs ##
% http://stackoverflow.com/questions/3181593/matlab-is-running-out-of-memory-but-it-should-not-be/3181851#3181851
X0 = bsxfun(@minus, X, mean(X,1));
[U,S,PC] = svds(X0, 4500);
varPC = diag(S'*S)' / (size(X,1)-1);
total_variance = sum(varPC);
varPC = varPC / total_variance;
perc_explained = zeros(size(X, 1), 1);
perc_explained(1) = varPC(1);
for i = 2:length(varPC)
    perc_explained(i) = perc_explained(i-1) + varPC(i);
end
%% PCA using 763 PCs
X0 = bsxfun(@minus, X, mean(X,1));
[U,S,princ_loading] = svds(X0, 763);
X_hat = X0*princ_loading;
% save('X_hat_PC_763.mat','X_hat');
%% PCA with pca function
X_hat = full(X);
[coeff,X_hat,latent] = pca(X_hat,'NumComponents',763);
% save('X_hat_PC_763.mat','X_hat');
% save('coeff_PC_763.mat','coeff');
%% 10 fold + Logistic Full data
addpath('four_model_project_code/liblinear');
N = size(X, 1);
K  = 10;
Indices = crossvalind('Kfold', N, K);
precision = 0;
% run 1 fold
% % X_train = X_hat(Indices ~= 1, :);
% % X_test = X_hat(Indices == 1, :);
% X_train = X(Indices ~= 1, :);
% X_test = X(Indices == 1, :);
% Y_train = Y(Indices ~= 1);
% Y_test = Y(Indices == 1);
% [ p,  predicted_label] = logistic( X_train, Y_train, X_test, Y_test );
%run 10 fold cross-validation
for k = 1:K
%     X_train = X_hat(Indices ~= k, :);
%     X_test = X_hat(Indices == k, :);
    X_train = X(Indices ~= k, :);
    X_test = X(Indices == k, :);
    Y_train = Y(Indices ~= k);
    Y_test = Y(Indices == k);
    [ p,  predicted_label] = logistic( X_train, Y_train, X_test, Y_test );
    precision = precision + p;
end
precision = precision / K;
%% K-means with PCA
load X_hat_PC_763.mat
clusters = [100, 500, 1000, 2000];
precision = zeros(length(clusters), 1);
N = size(X_hat, 1);
K  = 10;
Indices = crossvalind('Kfold', N, K);
for i = 1:length(clusters)
    c = clusters(i);
    for k = 1:K
        X_train = X_hat(Indices ~= k, :);
        X_test = X_hat(Indices == k, :);
        Y_train = Y(Indices ~= k);
        Y_test = Y(Indices == k);
        precision(i) = precision(i) + k_means(X_train, Y_train, X_test, Y_test, c);
    end
end
precision = precision / K;
% save('precision_K_means_set_1.mat','precision');
%% KNN with 10-fold cross validation PCA and prior, K = 50 is the best
% add KL Divergence
load X_hat_PC_763.mat
% set 1
% Ks = [1, 3, 5, 10, 50, 100, 500, 1000, 2000, 3000];
% distFuncs = {'euclidean', 'cityblock', 'chebychev', 'correlation'};

% set 2
Ks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400 500];
distFuncs = {'correlation'};

% set 3
% Ks = [30, 40, 50];
% distFuncs = {'correlation'};
precision = zeros(length(Ks), length(distFuncs));
N = size(X_hat, 1);
K  = 10;
Indices = crossvalind('Kfold', N, K);
% X_train = X_hat(Indices ~= 1, :);
% X_test = X_hat(Indices == 1, :);
% Y_train = Y(Indices ~= 1);
% Y_test = Y(Indices == 1);
% mdl = fitcknn(X_train, Y_train);
% func = distFuncs{4};
% mdl.Distance = func;
% mdl.NumNeighbors = 100;
% mdl.BreakTies = 'nearest';
% mdl.IncludeTies = false;
% p = mean(predict(mdl,X_test) == Y_test);
for k = 1:K
    X_train = X_hat(Indices ~= k, :);
    X_test = X_hat(Indices == k, :);
    Y_train = Y(Indices ~= k);
    Y_test = Y(Indices == k);
    mdl = fitcknn(X_train, Y_train);
    mdl.BreakTies = 'nearest';
    mdl.IncludeTies = false;
%     set 3
%     prior = full([1 - mean(Y_train == 1); mean(Y_train == 1)]);
%     mdl.Prior = prior;
    for i = 1:length(Ks)
        c = Ks(i);
        mdl.NumNeighbors = c;
        for j = 1:length(distFuncs)
            func = distFuncs{j};
            mdl.Distance = func;
            precision(i, j) = precision(i, j) + mean(predict(mdl,X_test) == Y_test);
        end
    end
end
precision = precision / K;
% save('precision_KNN_set_1.mat','precision');
% save('precision_KNN_set_2.mat','precision');
% save('precision_KNN_prior_set_3.mat','precision');
%% KNN Model building
load X_hat_PC_763.mat
mdl = fitcknn(X_hat, Y);
mdl.BreakTies = 'nearest';
mdl.IncludeTies = false;
mdl.Distance = 'correlation';
mdl.NumNeighbors = 50;
% save('KNN_nearest_correlation_C_50.mat','mdl');
%% KNN with Gaussian Kernel PCA 10-fold cross-validation
load X_hat_PC_763.mat
% Set 1
% Ks = [100, 500, 1000, 2000, 3000, 4500];
% distFuncs = {'euclidean', 'cityblock', 'chebychev', 'correlation'};
% sigmas = [5, 10, 50, 100];
% Set 2
% Ks = [1, 5, 10, 50, 100, 200];
% distFuncs = {'euclidean', 'cityblock', 'chebychev', 'correlation'};
% sigmas = 1:5;
% Set 3
Ks = [10, 20, 30, 40, 50, 60, 70, 80];
distFuncs = {'euclidean', 'cityblock', 'chebychev', 'correlation'};
sigmas = [0.0001, 0.001, 0.01, 0.1, 1];

precision = zeros(length(Ks), length(distFuncs), length(sigmas));
N = size(X_hat, 1);
K  = 10;
Indices = crossvalind('Kfold', N, K);
% k = 1;
% X_train = X_hat(Indices ~= k, :);
% X_test = X_hat(Indices == k, :);
% Y_train = Y(Indices ~= k);
% Y_test = Y(Indices == k);
% mdl = fitcknn(X_train, Y_train);
% s = 1;
% sigma = sigmas(s);
% weight_func = @(d)(exp(- d.^2 / (2 * sigma^2)));

% d = 1;
% func = distFuncs{d};
% mdl.Distance = func;
% c = 5;
% mdl.NumNeighbors = Ks(c);
% mdl.BreakTies = 'nearest';
% mdl.IncludeTies = false;
% mdl.DistanceWeight = weight_func;
% set 3
% mdl.Prior = prior;
% precision(c, d, s) = mean(predict(mdl,X_test) == Y_test);
for k = 1:K
    X_train = X_hat(Indices ~= k, :);
    X_test = X_hat(Indices == k, :);
    Y_train = Y(Indices ~= k);
    Y_test = Y(Indices == k);
    mdl = fitcknn(X_train, Y_train);
    mdl.BreakTies = 'nearest';
    mdl.IncludeTies = false;
%     prior = full([1 - mean(Y_train == 1); mean(Y_train == 1)]);
%     mdl.Prior = prior;
    for s = 1:length(sigmas)
        sigma = sigmas(s);
        weight_func = @(d)(exp(- d.^2 / (2 * sigma^2)));
        mdl.DistanceWeight = weight_func;
        for d = 1:length(distFuncs)
            func = distFuncs{d};
            mdl.Distance = func;
            for c = 1:length(Ks)
                mdl.NumNeighbors = Ks(c);
                precision(c, d, s) = precision(c, d, s) + mean(predict(mdl,X_test) == Y_test);
            end
        end
    end
end
precision = precision / K;
% save('precision_KNN_Gaussian_kernel_set_1.mat','precision');
% save('precision_KNN_Gaussian_kernel_set_2.mat','precision');
% save('precision_KNN_Gaussian_kernel_set_3.mat','precision');
%% SVM use fitrlinear
load X_hat_PC_763.mat
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
precision = precision / K;
% k = @(x,x2) kernel_intersection(x, x2);
% precision = kernel_libsvm(X_train, Y_train, X_test, Y_test, k);
% precision = 1 - precision;
% k = @(x,x2) kernel_poly(x, x2, 1);
% precision(1) = precision(1) + kernel_libsvm(X_train, Y_train, X_test, Y_test, k);
% k = @(x,x2) kernel_poly(x, x2, 2);
% precision(2) = precision(2) + kernel_libsvm(X_train, Y_train, X_test, Y_test, k);
% k = @(x,x2) kernel_poly(x, x2, 3);
% precision(3) = precision(3) + kernel_libsvm(X_train, Y_train, X_test, Y_test, k);
% for k = 1:K
%     X_train = X_hat(Indices ~= k, :);
%     X_test = X_hat(Indices == k, :);
%     Y_train = Y(Indices ~= k);
%     Y_test = Y(Indices == k);
%     
%     k = @(x,x2) kernel_poly(x, x2, 1);
%     precision(1) = precision(1) + kernel_libsvm(X_train, Y_train, X_test, Y_test, k);
%     k = @(x,x2) kernel_poly(x, x2, 2);
%     precision(2) = precision(2) + kernel_libsvm(X_train, Y_train, X_test, Y_test, k);
%     k = @(x,x2) kernel_poly(x, x2, 3);
%     precision(3) = precision(3) + kernel_libsvm(X_train, Y_train, X_test, Y_test, k);
%     k = @(x,x2) kernel_gaussian(x, x2, 20);
%     precision(4) = precision(4) + kernel_libsvm(X_train, Y_train, X_test, Y_test, k);
%     k = @(x,x2) kernel_intersection(x, x2);
%     precision(5) = precision(5) + kernel_libsvm(X_train, Y_train, X_test, Y_test, k);
% end
% precision = 1 - precision / K;
%% Train SVM
load X_hat_PC_763.mat
Y_hat = full(Y);
mdl = fitrlinear(X_hat, Y_hat, 'Regularization', 'lasso', 'Solver', 'sparsa');
YHat = predict(mdl,X_hat);
YHat(YHat > 0.6) = 1;
YHat(~(YHat > 0.6)) = 0;
precision = mean(YHat == Y_hat);
% save('SVM_fitrlinear_model.mat','mdl');
%% Gausian Kernel with SVM
% load X_hat_PC_763.mat
addpath('four_model_project_code/libsvm');
% use original data
X_hat = X;
N = size(X_hat, 1);
K  = 10;
Indices = crossvalind('Kfold', N, K);
sigmas = [0.01, 0.1, 1, 10, 100, 1000];
precision = zeros(length(sigmas), 1);
k = 1;
X_train = X_hat(Indices ~= k, :);
X_test = X_hat(Indices == k, :);
Y_train = Y(Indices ~= k);
Y_test = Y(Indices == k);

for i = 1:length(sigmas)
    s = sigmas(i);
    k = @(x,x2) kernel_gaussian(x, x2, s);
    precision(i) = precision(i) + kernel_libsvm(X_train, Y_train, X_test, Y_test, k);
end
%% GMM
load X_hat_PC_763.mat
N = size(X_hat, 1);
K  = 10;
Indices = crossvalind('Kfold', N, K);
clusters = [10, 50, 100, 500, 1000];
precision = zeros(length(clusters), 1);
k = 1;
X_train = X_hat(Indices ~= k, :);
X_test = X_hat(Indices == k, :);
Y = full(Y);
Y_train = Y(Indices ~= k);
Y_test = Y(Indices == k);
i = 4;
c = clusters(i);
precision(i) = precision(i) + GMM(X_train,Y_train,X_test,Y_test, c);
%% GMM with Baysian Inference (Has bug, needs to find TA)
load X_hat_PC_763.mat
N = size(X_hat, 1);
K  = 10;
Indices = crossvalind('Kfold', N, K);
clusters = [5, 10, 50, 100];
precision = zeros(length(clusters), 1);
k = 1;
X_train = X_hat(Indices ~= k, :);
X_test = X_hat(Indices == k, :);
Y = full(Y);
Y_train = Y(Indices ~= k);
Y_test = Y(Indices == k);

category = 2;
prior = zeros(1, category);
for l = 1:category
    letter_mask = Y_train == l;
    prior(l) = mean(letter_mask);
end
i = 4;
c = clusters(i);
precision(i) = precision(i) + GMM_BI(X_train,Y_train,X_test,Y_test, c, prior);

%% Method Summary
% Baysian
% HW 02 Decision Tree, Q3
% HW 05 Supervised Neural Network, Q1
% HW 06(b) Perception, Q2
% HW 07 Auto-encoder with logistic & k-means Q5

% WORKING ON 
% HW 07 Baysian Inference with GMM Q3

% FINISHED
% HW 06(b) Kernel with SVM, Q1
% HW 07 PCA with logistic & k-means
% HW 07 GMM Q2
% HW 02 KNN, kernel method, Q2