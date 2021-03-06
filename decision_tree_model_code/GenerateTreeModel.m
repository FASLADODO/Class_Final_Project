function [treeModel, precision] =  GenerateTreeModel(X, Y)

X = full(X);
Y = full(Y);

N = size(X, 1);
K  = 10;
numOfTrees = [10, 25];
Indices = crossvalind('Kfold', N, K);

precision = zeros(K,numel(numOfTrees));

paroptions = statset('UseParallel',true);

for t = 1 : numel(numOfTrees)
    for k = 1:K
        Mdl = TreeBagger(numOfTrees(t),X(Indices ~= k,:),Y(Indices ~= k),...
                'OOBPrediction','On', 'Method','classification', ...
                 'Options', paroptions);
        Yfit = predict(Mdl, X(Indices == k,:));
        precision(k, t) = mean(cellfun(@str2double,Yfit) == Y(Indices == k));
    end
end

modelPerformance = mean(precision,1);
[precision, idx] = max(modelPerformance);
treeModel = TreeBagger(numOfTrees(idx),X, Y, 'OOBPrediction','On', ...
            'Method','classification','OOBPredictorImportance', 'on', ...
            'Options', paroptions);

end
