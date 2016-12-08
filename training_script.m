%% Preparation data
clear all
addpath('four_model_project_code');
% load final_project_kit2/train_set/raw_tweets_train.mat
load ../final_project_kit2/train_set/words_train.mat
% load ../final_project_kit2/train_set/train_raw_img.mat
% load final_project_kit2/train_set/train_cnn_feat.mat
% load final_project_kit2/train_set/train_img_prob.mat
% load final_project_kit2/train_set/train_color.mat
% load final_project_kit2/train_set/train_tweet_id_img.mat
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
%% SVM model building
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
%% Mutual Information Feature Selection Frequency
% http://stackoverflow.com/questions/13603882/feature-selection-and-reduction-for-text-classification
clear all
addpath('four_model_project_code');
addpath('PRML_MATLAB_TOOL_BOX/chapter01');
load ../final_project_kit/train_set/words_train.mat
mi = zeros(size(X, 2), 1);
for i = 1:size(X, 2)
    mi(i) = mutInfo(Y, X(:, i));
end
[sortedMI,sortedIndex] = sort(mi,'descend');

top_words = [1000, 1250, 1500];
%top_words = 526;
precision = zeros(length(top_words), 1);
N = size(X, 1);
K  = 10;
Indices = crossvalind('Kfold', N, K);
thresholds = 0.61;

for i = 1:length(top_words)
    t = top_words(i);
    index = sortedIndex(1:t);
    X_MI = full(X(:, index));
    Y_MI = full(Y);
    for k = 1:K
        X_train = X_MI(Indices ~= k, :);
        X_test = X_MI(Indices == k, :);
        Y_train = Y_MI(Indices ~= k);
        Y_test = Y_MI(Indices == k);
        mdl = fitrlinear(X_train, Y_train, 'Regularization', 'lasso', 'Solver', 'sparsa');
        YHat = predict(mdl,X_test);
        YPred = zeros(length(YHat), 1);
        YPred(YHat > thresholds) = 1;
        precision(i) = precision(i) + mean(YPred == Y_test);
    end
end
precision = precision / K;
% save('MI_Frequency_X_Y.mat','index', 'thresholds');
% save('MI_Frequency_MI_SortedIndex.mat','sortedIndex', 'sortedMI');
%% MI Frequency SVM model (can be used)
clear all
load MI_Frequency_X_Y.mat
threshold = 0.61;
mdl = fitrlinear(X_MI, Y_MI, 'Regularization', 'lasso', 'Solver', 'sparsa');
YHat = predict(mdl,X_MI);
YPred = zeros(length(YHat), 1);
YPred(YHat > threshold) = 1;
precision = mean(YPred == Y_MI);
% save('MI_Frequency_SVM_model.mat','mdl');
%% LASSO feature selection
% lasso feature selection matlab (google)

%% SVM using MI frequency
clear all
load 'MI_X_Y.mat'
N = size(X_MI, 1);
K  = 10;
Indices = crossvalind('Kfold', N, K);
thresholds = 0.6:0.01:0.7;
precision = zeros(length(thresholds), 1);
for k = 1:K
    X_train = X_MI(Indices ~= k, :);
    X_test = X_MI(Indices == k, :);
    Y_train = Y_MI(Indices ~= k);
    Y_test = Y_MI(Indices == k);
    mdl = fitrlinear(X_train, Y_train, 'Regularization', 'lasso', 'Solver', 'sparsa');
    YHat = predict(mdl,X_test);
    for t = 1:length(thresholds)
        threshold = thresholds(t);
        YPred = zeros(length(YHat), 1);
        YPred(YHat > threshold) = 1;
        precision(t) = precision(t) + mean(YPred == Y_test);
    end
end
precision = precision / K;
%% MI  + PCA Feature generation (can be subsituted by the follow block)
clear all
load ../final_project_kit/train_set/words_train.mat
load MI_Frequency_MI_SortedIndex.mat

t = 526;
index = sortedIndex(1:t);
X_MI = full(X(:, index));
Y_MI = full(Y);
[coeff,score_526,latent,tsquared,explained_526,mu] = pca(X_MI,'NumComponents',87);
for i = 2 : length(explained_526)
    explained_526(i) = explained_526(i) + explained_526(i-1);
end
save('MI_PCA_526.mat','coeff', 'index');
% t = 744;
% index = sortedIndex(1:t);
% X_MI = full(X(:, index));
% Y_MI = full(Y);
% [coeff,score_744,latent,tsquared,explained_744,mu] = pca(X_MI,'NumComponents',105);
% for i = 2 : length(explained_744)
%     explained_744(i) = explained_744(i) + explained_744(i-1);
% end
% save('MI_PCA_744.mat','coeff', 'index');
% t = 1350;
% index = sortedIndex(1:t);
% X_MI = full(X(:, index));
% Y_MI = full(Y);
% [coeff,score_1350,latent,tsquared,explained_1350,mu] = pca(X_MI,'NumComponents',167);
% for i = 2 : length(explained_1350)
%     explained_1350(i) = explained_1350(i) + explained_1350(i-1);
% end
% save('MI_PCA_1350.mat','coeff', 'index');
%% MI + PCA dataset generation
clear all
load ../final_project_kit/train_set/words_train.mat
load MI_Frequency_MI_SortedIndex.mat

t = 4748;
index = sortedIndex(1:t);
X_MI = full(X(:, index));
Y_MI = full(Y);
[ coeff ] = get_pca(X_MI);

save('MI_PCA.mat','coeff', 'index');
%% SVM + PCA SVM (CAN BE USED)
% t = 4748, thresholds = 0.62, accuracy = 0.7929
% t = 5284, thresholds = 0.60, accuracy = 0.7838
clear all
load ../final_project_kit/train_set/words_train.mat
load MI_Frequency_MI_SortedIndex.mat
load MI_PCA.mat

X_MI = full(X(:, index));
Y_MI = full(Y);
X0 = bsxfun(@minus, X_MI, mean(X_MI,1));
X_MI = X0*coeff;

N = size(X_MI, 1);
K  = 10;
Indices = crossvalind('Kfold', N, K);
thresholds = 0.55:0.01:0.7;
precision = zeros(length(thresholds), 1);
for t = 1:length(thresholds)
    threshold = thresholds(t);
    for k = 1:K
        X_train = X_MI(Indices ~= k, :);
        X_test = X_MI(Indices == k, :);
        Y_train = Y_MI(Indices ~= k);
        Y_test = Y_MI(Indices == k);
        mdl = fitrlinear(X_train, Y_train, 'Regularization', 'lasso', 'Solver', 'sparsa');
        YHat = predict(mdl,X_test);
        YHat(YHat > threshold) = 1;
        YHat(~(YHat > threshold)) = 0;
        precision(t) = precision(t) + mean(YHat == Y_test);
    end
end
precision = precision / K;
%% Mutual Information Feature Selection SVM
clear all
addpath('four_model_project_code');
addpath('PRML_MATLAB_TOOL_BOX/chapter01');
load ../final_project_kit/train_set/words_train.mat
X_presence = full(X);
X_presence(X_presence ~= 0) = 1;
mi = zeros(size(X_presence, 2), 1);
for i = 1:size(X_presence, 2)
    mi(i) = mutInfo(Y, X_presence(:, i));
end
[sortedMI,sortedIndex] = sort(mi,'descend');
% top_words = [413, 712, 1304, 1913, 3095, 4666];
% precision = zeros(length(top_words), 1);
top_words = 3095;
N = size(X, 1);
K  = 10;
Indices = crossvalind('Kfold', N, K);
% thresholds = 0.55:0.01:0.65;
thresholds = 0.59;
precision = zeros(length(thresholds), 1);

% for i = 1:length(top_words)
    i = 1;
    top_word = top_words(i);
    index = sortedIndex(1:top_word);
    X_MI = full(X_presence(:, index));
    Y_MI = full(Y);
    for k = 1:K
        X_train = X_MI(Indices ~= k, :);
        X_test = X_MI(Indices == k, :);
        Y_train = Y_MI(Indices ~= k);
        Y_test = Y_MI(Indices == k);
        mdl = fitrlinear(X_train, Y_train, 'Regularization', 'lasso', 'Solver', 'sparsa');
        YHat = predict(mdl,X_test);
        for t = 1:length(thresholds)
%             t = 1;
            threshold = thresholds(t);
            YPred = zeros(length(YHat), 1);
            YPred(YHat > threshold) = 1;
            precision(t) = precision(t) + mean(YPred == Y_test);
%             precision(i) = precision(i) + mean(YPred == Y_test);
        end
    end
% end
precision = precision / K;
% save('MI_Feature_SVM_index_thresholds.mat','index', 'thresholds');
% save('MI_Feature_sortMI_sortIndex.mat','sortedMI', 'sortedIndex');
%% Mutual Information Feature Selection SVM Model building
clear all
load ../final_project_kit/train_set/words_train.mat
load MI_Feature_SVM_index_thresholds.mat

X_presence = full(X);
X_presence(X_presence ~= 0) = 1;
X_MI = full(X_presence(:, index));
Y_MI = full(Y);

mdl = fitrlinear(X_MI, Y_MI, 'Regularization', 'lasso', 'Solver', 'sparsa');
YHat = predict(mdl,X_MI);
YPred = zeros(length(YHat), 1);
YPred(YHat > thresholds) = 1;
precision = mean(YPred == Y_MI);
% save('MI_Feature_SVM_model.mat','mdl');
%% Naive Baysian with Frequency
% https://www.mathworks.com/help/stats/fitcnb.html
% section Train Naive Bayes Classifiers using multinomial Predictors
clear all
load ../final_project_kit/train_set/words_train.mat
load MI_Frequency_MI_SortedIndex.mat

% top_words = [435, 807, 1350, 1913, 3186, 4748, 5248];
top_words = 3186;
precision = zeros(length(top_words), 1);
N = size(X, 1);
K  = 10;
Indices = crossvalind('Kfold', N, K);

for i = 1:length(top_words)
%     i = 1;
    top_word = top_words(i);
    index = sortedIndex(1:top_word);
    X_MI = full(X(:, index));
    Y_MI = full(Y);
    for k = 1:K
        X_train = X_MI(Indices ~= k, :);
        X_test = X_MI(Indices == k, :);
        Y_train = Y_MI(Indices ~= k);
        Y_test = Y_MI(Indices == k);
        Mdl = fitcnb(X_train,Y_train,'Distribution','mn');
        [label,Posterior,Cost] = predict(Mdl,X_test);
        precision(i) = precision(i) + mean(label == Y_test);
    end
end
precision = precision / K;
% save('NB_Frequency_index.mat','index');
%% Naive Baysian with Frequency Model Building
clear all
load ../final_project_kit/train_set/words_train.mat
load NB_Frequency_index.mat

X_MI = full(X(:, index));
Y_MI = full(Y);
Mdl = fitcnb(X_MI,Y_MI,'Distribution','mn');
[label,Posterior,Cost] = predict(Mdl,X_MI);
precision = mean(label == Y_MI);
% save('NB_Frequency_model.mat','Mdl');
%% Naive Baysian with Feature
% https://www.mathworks.com/help/stats/fitcnb.html
% section Train Naive Bayes Classifiers using multinomial Predictors
clear all
load ../final_project_kit/train_set/words_train.mat
load MI_Feature_sortMI_sortIndex.mat

X_presence = full(X);
X_presence(X_presence ~= 0) = 1;
% top_words = [413, 712, 1304, 1913, 3095, 4666, 5252];
top_words = 3095;
precision = zeros(length(top_words), 1);
N = size(X, 1);
K  = 10;
Indices = crossvalind('Kfold', N, K);

for i = 1:length(top_words)
    top_word = top_words(i);
    index = sortedIndex(1:top_word);
    X_MI = full(X_presence(:, index));
    Y_MI = full(Y);
    for k = 1:K
        X_train = X_MI(Indices ~= k, :);
        X_test = X_MI(Indices == k, :);
        Y_train = Y_MI(Indices ~= k);
        Y_test = Y_MI(Indices == k);
        Mdl = fitcnb(X_train,Y_train,'Distribution','mn');
        [label,Posterior,Cost] = predict(Mdl,X_test);
        precision(i) = precision(i) + mean(label == Y_test);
    end
end
precision = precision / K;
% save('NB_Feature_index.mat','index');
%% Naive Baysian with Feature Model Building
clear all
load ../final_project_kit/train_set/words_train.mat
load NB_Feature_index.mat

X_presence = full(X);
X_presence(X_presence ~= 0) = 1;
X_MI = full(X_presence(:, index));
Y_MI = full(Y);
Mdl = fitcnb(X_MI,Y_MI,'Distribution','mn');
[label,Posterior,Cost] = predict(Mdl,X_MI);
precision = mean(label == Y_MI);
% save('NB_Feature_model.mat','Mdl');
%% chi^2 feature Frequency Selection Naive Bayes
clear all
addpath('chi2feature');
load ../final_project_kit/train_set/words_train.mat

X_chi = full(X);
Y_chi = full(Y);
[chi, df] = chi2feature(X_chi,Y_chi);

[sortedChi2,sortedIndex] = sort(chi,'descend');

% top_words = [153, 428, 838, 1481, 2055, 3225, 4897, 5249];
top_words = 3225;
precision = zeros(length(top_words), 1);
N = size(X, 1);
K  = 10;
Indices = crossvalind('Kfold', N, K);

for i = 1:length(top_words)
    top_word = top_words(i);
    index = sortedIndex(1:top_word);
    X_chi = full(X(:, index));
    Y_chi = full(Y);
    for k = 1:K
        X_train = X_chi(Indices ~= k, :);
        X_test = X_chi(Indices == k, :);
        Y_train = Y_chi(Indices ~= k);
        Y_test = Y_chi(Indices == k);
        Mdl = fitcnb(X_train,Y_train,'Distribution','mn');
        [label,Posterior,Cost] = predict(Mdl,X_test);
        precision(i) = precision(i) + mean(label == Y_test);
    end
end
precision = precision / K;
% save('chi2_Frequency_sortChi2_sortIndex.mat','sortedChi2', 'sortedIndex');
% save('NB_chi2_Frequency_index.mat','index');
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
%% combining methods
% SVM + PCA SVM
% Mutual Information Feature Selection SVM Model building (thrid best)
% Naive Baysian with Frequency Model Building (second best)
% Naive Baysian with Feature Model Building (BEST)


% Naive Baysian with MI Feature Selection