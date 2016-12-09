%% Load the data
clear all
clear all
cnn_feat = 0;
prob_feat = 0;
color_feat = 0;
raw_imgs = 0;
raw_tweets = 0;
load words_train.mat
%% SVM PreProcess
% Mutual Information Feature Selection
X_presence = full(X);
X_presence(X_presence ~= 0) = 1;
mi = zeros(size(X_presence, 2), 1);
for i = 1:size(X_presence, 2)
    mi(i) = mutInfo(Y, X_presence(:, i));
end
[sortedMI,sortedIndex] = sort(mi,'descend');
% select the top 3095 words that has the largest Mutual Information
top_word = 3095;
index = sortedIndex(1:top_word);
% save this index to be used in model building
save('SVM_index.mat','index');

%% SVM prediction
N = size(X,1);
K = 10;
Indices = crossvalind('kfold', N, K);
svm_training_features = round(linspace(100,5000,10));
cv_precision = zeros(numel(svm_training_features),K);

priors = full([sum(Y == 0)/size(Y,1) sum(Y == 1)/size(Y,1)]);
ranges = linspace(0,1,10000);

for j = 1:numel(svm_training_features)
    index = sortedIndex(1:svm_training_features(j));
    for k = 1:K
        svm_mdl = svm_model(X_presence(Indices~=k,:), full(Y(Indices~=k)), index);
        Y_Prob = predict(svm_mdl,X_presence(Indices==k,index));
        YHat = zeros(length(Y_Prob), 1);
        threshold = ranges(max(find(arrayfun(@(x) mean(Y_Prob > x) > priors(2),ranges))));
        YHat(Y_Prob > threshold) = 1;
        cv_precision(j,k) = mean(full(Y(Indices==k)) == YHat);
    end
end
svm_cv_precision = sum(cv_precision,2)/K;
% save('svm_cv_precision.mat', 'svm_cv_precision');

%% SVM Model generating
% load SVM_index.mat
[~, index] = max(svm_cv_precision);
index = sortedIndex(1:svm_training_features(index));
save('SVM_index.mat', 'index');

mdl = svm_model(X, Y, index);
% save the SVM model
save('SVM_model.mat','mdl');
%% SVM prediction
[Y_hat] = predict_labels(X, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets);
precision = mean(full(Y) == Y_hat);