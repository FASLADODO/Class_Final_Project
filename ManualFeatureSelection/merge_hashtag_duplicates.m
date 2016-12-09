function [colsToRemove, indicatorCols] = merge_hashtag_duplicates(X)
load topwords.mat

%%Find all the hashtags in top words
hashtags_idxs = strncmpi(topwords,'#',1);
hashtags = topwords(hashtags_idxs);

%%Remove the hashtag character and search for matching non-hashtag words
strippedWords = arrayfun(@(x)x{1}(2:end), hashtags, 'UniformOutput', false);
stripped_idxs = arrayfun(@(x)find(strcmp(topwords,x),1),strippedWords, 'UniformOutput', false);
nonhashtag_idxs = cell2mat(stripped_idxs);

%%Get the indices for the matching hashtag words
hashtagWords = arrayfun(@(x)['#' x{1}], topwords(nonhashtag_idxs), 'UniformOutput', false);
matching_hashtag_idxs = cell2mat(arrayfun(@(x)find(strcmp(topwords,x),1),hashtagWords, 'UniformOutput', false));

colsToRemove = [nonhashtag_idxs; matching_hashtag_idxs];

indicatorCols = zeros(size(X,1), size(nonhashtag_idxs,2));

for feature = 1:numel(nonhashtag_idxs)
    indicatorCols(:,feature) = X(:,nonhashtag_idxs(feature)) + X(:,matching_hashtag_idxs(feature));
end



