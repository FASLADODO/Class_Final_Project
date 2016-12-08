function colsToRemove = remove_stop_words(stopWordsFile, stopWordsDelimiter)

load('StopWords1.mat');
% stopwords = get_column_of_words_from_csv(stopWordsFile,stopWordsDelimiter);
load('topwords.mat');
% topwords = get_column_of_words_from_csv('../final_project_kit2/topwords.csv','\n');
idxs = arrayfun(@(x)find(strcmp(topwords,x),1),stopwords, 'UniformOutput', false);
colsToRemove = cell2mat(idxs);
end