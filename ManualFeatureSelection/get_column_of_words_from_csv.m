function listOfWords = get_column_of_words_from_csv(fileAddress, delimiter)
fid = fopen(fileAddress);
wordCell = textscan(fid, repmat('%s',1,1), 'delimiter',delimiter, 'CollectOutput',true);
listOfWords = wordCell{1};
end