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

load NB_chi2_Feature_model.mat
load NB_chi2_Feature_index.mat

X_presence = full(word_counts);
X_presence(X_presence ~= 0) = 1;
X_chi = full(X_presence(:, index));

[Y_hat,Posterior,Cost] = predict(Mdl,X_chi);
end
