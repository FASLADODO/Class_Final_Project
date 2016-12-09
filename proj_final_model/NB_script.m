%% Load the data
clear all
% Ignore all provided features except word counts
cnn_feat = 0;
prob_feat = 0;
color_feat = 0;
raw_imgs = 0;
raw_tweets = 0;
load words_train.mat

%% Naive Bayes PreProcess
% Mutual Information Feature Selection
X_presence = full(X);
X_presence(X_presence ~= 0) = 1;
mi = zeros(size(X_presence, 2), 1);
for i = 1:size(X_presence, 2)
    mi(i) = mutInfo(Y, X_presence(:, i));
end
[sortedMI,sortedIndex] = sort(mi,'descend');
% select the top 3095 words that has the largest Mutual Information
% verified using cv
top_word = 3095;
index = sortedIndex(1:top_word);
% save this index to be used in model building
save('NB_index.mat','index');
%% Naive Bayes Model generating
load NB_index.mat
mdl = nb_model(X, Y, index);
% save the Naive Bayes model
save('NB_model.mat','mdl');
%% Naive Bayes prediction
N = size(X,1);
K = 10;
Indices = crossvalind('kfold', N, K);
cv_precision = zeros(K,1);
for k = 1:K
    nb_mdl = nb_model(X(Indices~=k,:), full(Y(Indices~=k)), index);
    YHat = nb_predict(nb_mdl, X(Indices==k,:));
    cv_precision(k) = mean(full(Y(Indices==k)) == YHat);
end
nb_cv_precision = sum(cv_precision)/K;
save('nb_cv_precision.mat','nb_cv_precision');

%Chose NB as final model
[Y_hat] = predict_labels(X, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets);
precision = mean(full(Y) == Y_hat);