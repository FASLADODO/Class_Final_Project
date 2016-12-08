function [X_negated] = CountNegationWords(X, raw_tweets, topwords)
X_negated = [full(X) zeros(size(full(X)))];
raw = cell(size(raw_tweets,1),1);

%Clean the quotation marks off
for row = 1 : size(raw_tweets,1)
    raw(row) = textscan(raw_tweets{row}, '%s');
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

negationWords = {'not', 'doesn"t', 'doesnt', 'can"t', 'cant', 'cannot',...
                'isn"t', 'isnt', 'aren"t', 'arent', 'no', 'hadn"t', 'hadnt',...
                'don"t', 'dont', 'won"t', 'wont'};
            
punctuation = {'.', '?', ',', ';', ':', '"','!'};

for row = 1 : numel(raw)
    negated = false;
    sentence = raw{row};
    for word = 1: numel(raw{row})
        word_str = sentence{word,1};
        if negated
            if logical(sum(strcmp(topwords,word_str)))
                col_idx = 10000+find(strcmp(topwords,word_str));
                X_negated(row, col_idx) =  X_negated(row, col_idx) + 1;
                X_negated(row, col_idx-10000) =  max(X_negated(row, col_idx-10000) - 1, 0);
            end
        end
        word_len = length(word_str);
        if strcmp(word_str, 'not')
            negated = ~negated;
        elseif (word_len >= 3) && (strcmp(word_str(word_len-2:end), 'n"t'))
            negated = ~negated;
        elseif ismember(word_str, negationWords)
            negated =~ negated;
        elseif ismember(word_str, punctuation)
            negated = false;
        end
        
    end            
end

assert(sum(sum(X_negated,1),2)/sum(sum(X,1),2) < 1.01 ,'X_negated has too many counted words');
assert(sum(sum(X_negated,1),2)/sum(sum(X,1),2) >= 1 ,'X_negated has too few counted words');


end