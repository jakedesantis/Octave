function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

%disp(idx);

%fprintf('Program paused. Press enter to continue. (jake)\n');
%pause;

for k = 1:K
  sum_k = zeros(1,n);
  counter = 0;
  for i = 1:m
    if idx(i) == k
      sum_k = sum_k + X(i,:);
      counter = counter + 1;
%      pause(0.5);
%      fprintf('k is now %f\n', k);
%      fprintf('i is now %f\n', i);
%      fprintf('idx is now %f\n', idx(i));
%      fprintf('sum_k is now %f\n');
%      disp(sum_k);
%      fprintf('centroids is now %f\n');
%      disp(centroids);
    end
  end
  centroids(k,:) = sum_k/counter;
end

% =============================================================


end

