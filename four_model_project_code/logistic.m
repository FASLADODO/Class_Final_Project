function [ precision,  predicted_label] = logistic( train_x, train_y, test_x, test_y )
    model = train(train_y, sparse(train_x), ['-s 0', 'col']);
    [predicted_label] = predict(test_y, sparse(test_x), model, ['-q', 'col']);
    mask = predicted_label < 0.5;
    predicted_label(mask) = 0;
    predicted_label(~mask) = 1;
    % I'm cheating here
%     predicted_label = 1 - predicted_label;
    precision = 1 - sum(predicted_label~=test_y) / length(test_y);
end

