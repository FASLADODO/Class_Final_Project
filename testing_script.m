%% Preparation data
clear all
addpath('leaderboard_model_code');
%% load testing
load ../final_project_kit/train_set_unlabeled/raw_tweets_train_unlabeled.mat
load ../final_project_kit/train_set_unlabeled/words_train_unlabeled.mat
load ../final_project_kit/train_set_unlabeled/train_unlabeled_raw_img.mat
load ../final_project_kit/train_set_unlabeled/train_unlabeled_cnn_feat.mat
load ../final_project_kit/train_set_unlabeled/train_unlabeled_img_prob.mat
load ../final_project_kit/train_set_unlabeled/train_unlabeled_color.mat
load ../final_project_kit/train_set_unlabeled/train_unlabeled_tweet_id_img.mat
%% load training
load ../final_project_kit/train_set/words_train.mat
cnn_feat = 0;
prob_feat = 0;
color_feat = 0;
raw_imgs = 0;
raw_tweets = 0;
Y_hat = full(Y);
[Y_pred] = predict_labels(X, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets);
precision = mean( Y_hat == Y_pred );