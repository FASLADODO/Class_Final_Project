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

% load('nbBernMdl.mat');
load('nbBernTopCols.mat');
load('nbBernMdlGram.mat');
% load('nbBernGramTopCols.mat');
load('neg_features_filtered_cols.mat');
load('topwords');
load('gram_keys');
word_counts = CountNegationWords(word_counts, raw_tweets, topwords);
word_counts = word_counts(:, neg_features_filtered_cols);
word_counts = double(word_counts > 0);
word_counts = [word_counts zeros(size(word_counts,1), numel(gram_keys))];

map = make_grams(raw_tweets);
for i = 1:numel(gram_keys)
    key = gram_keys(i);
    if map.isKey(key{1})
        rows = map(key{1});
        for j = 1:numel(rows)
            word_counts(rows(j), i+2338) = 1;
        end
    end
end

word_counts = word_counts(:, nbBernTopCols);

% Y_hat = nbBernPred(nbBernMdl, word_counts');
Y_hat = nbBernPred(nbBernMdlGram, word_counts');
Y_hat = (Y_hat - 1)';

end
