%% Preparation data
clear all
addpath('leaderboard_model_code');
%% load testing
load final_project_kit/train_set_unlabeled/raw_tweets_train_unlabeled.mat
load final_project_kit/train_set_unlabeled/words_train_unlabeled.mat
load final_project_kit/train_set_unlabeled/train_unlabeled_raw_img.mat
load final_project_kit/train_set_unlabeled/train_unlabeled_cnn_feat.mat
load final_project_kit/train_set_unlabeled/train_unlabeled_img_prob.mat
load final_project_kit/train_set_unlabeled/train_unlabeled_color.mat
load final_project_kit/train_set_unlabeled/train_unlabeled_tweet_id_img.mat