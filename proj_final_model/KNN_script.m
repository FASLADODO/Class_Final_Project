%% Load the data
clear all
cnn_feat = 0;
prob_feat = 0;
color_feat = 0;
raw_imgs = 0;
raw_tweets = 0;
load words_train.mat
%% KNN PreProcess
% PCA Feature Selection with 90% reconstruction accuracy
[ coeff ] = get_pca(X);
save('KNN_coeff.mat','coeff');
%% KNN Model generating
load KNN_coeff.mat
mdl = knn_model(X, Y, coeff);
% save the KNN model
save('KNN_model.mat','mdl');
%% KNN prediction
[Y_hat] = predict_labels(X, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets);
precision = mean(full(Y) == Y_hat);