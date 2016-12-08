function [cols_to_remove, indicatorFeatureCol] = group_user_names_into_one_feature(X)
X = full(X);
load('topwords.mat');
% topwords = get_column_of_words_from_csv('../final_project_kit2/topwords.csv','\n');
cols_to_remove = find(strncmpi(topwords,'@',1));
indicatorFeatureCol = logical(sum(X(:,cols_to_remove) ~= 0, 2));
end