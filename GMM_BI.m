function [precision] = GMM_BI(train_x,train_y,test_x,test_y, K, prior)
category = 2;
prob_dens_func = zeros(length(test_y), category);
for l = 1:category
    letter_mask = train_y == l;
    train_x_letter = train_x(letter_mask, :);

    option = statset('MaxIter', 1000);
    GMModel = fitgmdist(train_x_letter, K, 'CovarianceType', 'diagonal', 'Options', option, 'RegularizationValue', 1e-5);
    prob_dens_func(:, l) = pdf(GMModel, test_x);
end
prob_dens_func = bsxfun(@times, prob_dens_func, prior);
[~,label] = max(prob_dens_func,[],2);
precision = mean(label == test_y);
end