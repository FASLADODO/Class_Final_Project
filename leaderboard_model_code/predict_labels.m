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

load SVM_fitrlinear_model.mat
load coeff_PC_763.mat
word_counts_full = full(word_counts);

X0 = bsxfun(@minus, word_counts_full, mean(word_counts_full,1));
score = X0*coeff;

Y_hat = predict(mdl,score);
threshold = 0.6;
mask = Y_hat > threshold;
Y_hat(mask) = 1;
Y_hat(~mask) = 0;
end
