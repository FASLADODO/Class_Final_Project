function [cpre] = k_means(X, Y, X_test, coeff)
% X and Y and input data used to build the model and X_test are testing
% data
X = full(X);
X0 = bsxfun(@minus, X, mean(X,1));
train_x = X0*coeff;

train_y = Y;

X_test = full(X_test);
X_test0 = bsxfun(@minus, X_test, mean(X_test,1));
test_x = X_test0*coeff;

K = 500;
% separate into k clusters and assign labels to each cluster
label = zeros(K,1);
[IDX,C] = kmeans(train_x, K, 'MaxIter', 500);
for j = 1:K
    c = find(IDX == j);
    table = tabulate(train_y(c));
    [~,index] = max(table(:,2));
    label(j) = table(index,1);
end

% assign test points to clusters
cpre = zeros(size(test_x,1),1);
for j = 1:size(size(test_x, 1))
    m = bsxfun(@minus,C,double(test_x(j,:)));
    for k = 1:K
        dis(k) = norm(m(k,:));
    end
    [~,ind] = min(dis,[],2);
    cpre(j) = label(ind);
end
