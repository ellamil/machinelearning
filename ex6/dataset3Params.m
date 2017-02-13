function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))

C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
sig_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
error_mat = zeros(length(C_vec),length(sig_vec));
for c = 1:length(C_vec)
    for s = 1:length(sig_vec) 
        model = svmTrain(X, y, C_vec(c), @(x1, x2) gaussianKernel(x1, x2, sig_vec(s)));
        predictions = svmPredict(model, Xval);
        error_mat(c,s) =  mean(double(predictions ~= yval));
    end
end

% [C_ind, sigma_ind] = find(error_mat == min(min(error_mat)));
% C = C_vec(C_ind); 
% sigma = sig_vec(sigma_ind);

[minColumnError, minColumnErrorIndex] = min(error_mat);
[minError, minErrorIndex] = min(minColumnError);
C = C_vec(minColumnErrorIndex(minErrorIndex));
sigma = sig_vec(minErrorIndex);

% =========================================================================

end