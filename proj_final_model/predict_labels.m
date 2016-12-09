function [Y_hat] = predict_labels(word_counts, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets)
% Inputs:   word_counts     nx10000 word counts features
%           cnn_feat        nx4096 Penultimate layer of Convolutional
%                               Neural Network features
%           prob_feat       nx1365 Probabilities on 1000 objects and 365
%                               scene categories
%           color_feat      nx33 Color spectra of the images (33 dim)
%           raw_imgs        nx30000 raw images pixels
%           raw_tweets      nx1 cells containing all the raw tweets in text
% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 0 for sad)
%% Remove comments for the model that you need
%% Naive Bayes prediction
% load NB_model.mat
% Y_hat = nb_predict(mdl, word_counts);
%% SVM prediction
% load SVM_model.mat
% Y_hat = svm_predict(mdl, word_counts);
%% KNN prediction
% load KNN_coeff.mat
% load KNN_model.mat
% Y_hat = KNN_predict(mdl, word_counts, coeff);
%% K_means Prediction
% load K_means_coeff.mat
% load words_train.mat
% Y_hat = k_means(X, Y, word_counts, coeff);
end
