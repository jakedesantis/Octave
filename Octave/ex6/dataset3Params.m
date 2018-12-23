function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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
%


%Cs = [0.1, 0.15, 0.2, 0.25, 0.3];
%sigmas = [0.0015, 0.0018, 0.002, 0.0022, 0.0025];

Cs = [0.01,0.1,1.0,10.0];
sigmas = [0.01,0.1,1.0,10.0];


e = zeros(length(Cs), length(sigmas))

for i = [1:length(Cs)]
  for j = [1:length(sigmas)]
    C_trial = Cs(i)
    sigma_trial = sigmas(j)
    model= svmTrain(X, y, C_trial, @(x1, x2) gaussianKernel(x1, x2, sigma_trial));
    predictions = svmPredict(model, Xval);
    e(i,j) = mean(double(predictions ~= yval))

  end
end


[Cindex,Sindex] = indexesOfMin(e);

C = Cs(Cindex)
sigma = sigmas(Sindex)
%sigma = Cs(Cindex)
%C = sigmas(Sindex)
C = 1
sigma = 0.1




% =========================================================================

end
