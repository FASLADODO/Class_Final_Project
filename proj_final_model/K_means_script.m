%% Load the data
clear all
cnn_feat = 0;
prob_feat = 0;
color_feat = 0;
raw_imgs = 0;
raw_tweets = 0;
load words_train.mat
%% K_means PreProcess
% PCA Feature Selection with 90% reconstruction accuracy
[ coeff ] = get_pca(X);
save('K_means_coeff.mat','coeff');
%% K_means model building + Prediction
[Y_hat] = predict_labels(X, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets);
precision = mean(full(Y) == Y_hat);