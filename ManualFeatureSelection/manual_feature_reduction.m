function X_filtered = manual_feature_reduction(X)
load stopwords.mat
load topwords.mat
x_stop = cell2mat(arrayfun(@(x)find(strcmp(topwords,x),1),stopwords, 'UniformOutput', false));

[x_users, users_indicator] = group_user_names_into_one_feature(X);
[x_dupes, dupesCols] = merge_hashtag_duplicates(X);
full_x = full(X);
emptyFeatures = find(~any(full_x,1) == 1);
cols_to_remove = [x_stop; x_users; x_dupes; emptyFeatures'];
cols_to_remove = unique(cols_to_remove);

cols_to_keep = ones(size(X,2),1);
cols_to_keep(cols_to_remove) = 0;
cols_to_keep = logical(cols_to_keep);
X_filtered = X(:, cols_to_keep);
X_filtered = [X_filtered users_indicator];
X_filtered = [X_filtered dupesCols];
end