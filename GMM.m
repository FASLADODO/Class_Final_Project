function [precision] = GMM(train_x,train_y,test_x,test_y, K)

label = zeros(K, 1);
option = statset('MaxIter', 1000);
GMModel = fitgmdist(train_x, K, 'CovarianceType', 'diagonal', 'Options', option, 'RegularizationValue', 1e-5);
C = GMModel.mu;
IDX = cluster(GMModel,train_x);
for j = 1:K
    c = find(IDX == j);
    table = tabulate(train_y(c));
    [~,index] = max(table(:,2));
    label(j) = table(index,1);
end

cpre = zeros(size(test_y,1),1);
for j = 1:size(test_y)
    m = bsxfun(@minus,C,double(test_x(j,:)));
    for k = 1:K
        dis(k) = norm(m(k,:));
    end
    [~,ind] = min(dis,[],2);
    cpre(j) = label(ind);
end
precision = mean(cpre == test_y);