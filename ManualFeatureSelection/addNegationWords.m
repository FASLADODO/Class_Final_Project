function [new_features_X, new_features_names] = addNegationWords(X, raw_tweets) 
topwords = get_column_of_words_from_csv('../final_project_kit2/topwords.csv','\n');
raw = cell(size(raw_tweets{1,2},1),1);

new_features_X = full(X);

for row = 1 : size(raw_tweets{1,2},1)
    raw(row) = textscan(raw_tweets{1,2}{row}, '%s');
    sentence = raw{row};
    new_sentence = sentence;
    for word = 1: numel(raw{row})
        if strncmpi(sentence{word,1}, '"',1)
            new_sentence{word, 1} = cellstr(sentence{word,1}(2:end));
        end
        if strcmp(sentence{word,1}(length(sentence{word,1})), '"')
            second_to_last = length(sentence{word,1}) - 1;
            new_sentence{word,1} = cellstr(sentence{word,1}(1:second_to_last));
        end
    end
    raw{row} = new_sentence;
end

mapObj = containers.Map;
negationWords = {'not', 'doesn"t', 'doesnt', 'can"t', 'cant', 'cannot',...
                'isn"t', 'isnt', 'aren"t', 'arent', 'no', 'hadn"t', 'hadnt',...
                'don"t', 'dont', 'won"t', 'wont'};

for row = 1 : numel(raw)
    sentence = raw{row};
    new_sentence = sentence;
    for word = 2: numel(raw{row})
        if ismember(sentence{word-1,1}, negationWords)
            neg_word = negationWords(strncmp(sentence{word-1,1}, negationWords, length(sentence{word-1,1})));
            new_sentence{word,1} = strcat(neg_word,'_',sentence{word,1});
            word_idx = cell2mat(arrayfun(@(x) strcmp(x{1},sentence{word,1}), ...
                 topwords, 'UniformOutput', false));
            if ~mapObj.isKey(new_sentence{word,1})
               newWord = containers.Map('KeyType', 'double', 'ValueType', 'any'); 
               mapObj(new_sentence{word,1}{1}) = newWord;
            end
            wordKey = mapObj(new_sentence{word,1}{1});
            if ~wordKey.isKey(row)
                wordKey(row) = 0;
            end
            wordKey(row) = wordKey(row) + 1;
            mapObj(new_sentence{word,1}{1}) = wordKey;
            new_features_X(row, word_idx) = max(new_features_X(row, word_idx) - 1, 0);
        end
    end
    raw{row} = new_sentence;
            
end

new_features_names = keys(mapObj);
numNewFeatures = numel(new_features_names);
numOfCol = size(X,2);

for key = 1:numNewFeatures
    key_cell = new_features_names(key);
    new_feature = mapObj(key_cell{1});
    rows_to_update = keys(new_feature);
    for row = 1 :numel(rows_to_update)
        row_num = rows_to_update{row};
        new_features_X(row_num, numOfCol + key) = new_feature(row_num);
    end
end



%Create arbritary space less than total number of words
% allWords = cell(10000,1);

% counter = 1;
% for row = 1:numel(raw)
%     for cell = 1:numel(raw{1,row})
%         allWords(counter) = cellstr(raw{1,row}{cell});
%         counter = counter + 1;
%     end
% end

%Clean up quotation marks
% allWords(strncmpi(allWords,'"',1)) = arrayfun(@(x)x{1}(2:end), allWords(strncmpi(allWords,'"',1)), 'UniformOutput', false);
% lastCharacters = arrayfun(@(x)x{1}(length(x{1})) == '"',allWords);
% allWords(lastCharacters) = arrayfun(@(x)x{1}(1:(length(x{1})-1)), allWords(lastCharacters),'UniformOutput', false);
% 
% allWords = unique(allWords);

end
