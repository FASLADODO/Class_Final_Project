function [ coeff ] = get_pca(X_MI)
    [coeff_full,score,latent,tsquared,explained,mu] = pca(X_MI);
    index = min(find(cumsum(explained) > 90));
    coeff = coeff_full(:, 1:index);
end