function [ coeff ] = get_pca(X)
    X = full(X);
    [coeff_full,score,latent,tsquared,explained,mu] = pca(X);
    index = min(find(cumsum(explained) > 90));
    coeff = coeff_full(:, 1:index);
end