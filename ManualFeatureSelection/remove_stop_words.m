function colsToRemove = remove_stop_words()
load stopwords.mat
load topwords.mat
idxs = arrayfun(@(x)find(strcmp(topwords,x),1),stopwords, 'UniformOutput', false);
colsToRemove = cell2mat(idxs);
end