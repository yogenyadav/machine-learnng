function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

size(y)
size(all_theta)

%y is of 5000x1 with values like this [1;1;1...;2;2;2...;3;3;3...;4;4;4...;....]
%lets say a = [1;1;1;2;2;2;3;3;3;4;4;4;5;5;5;1;1;1;2;2;2;3;3;3;4;4;4;5;5;5]
%(a == 4) will return a matrix where all the 4's are set to 1 and and others to 0 like this 
%[0;0;0;0;0;0;0;0;0;1;1;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0]
%here this means training for classifier 4

for c = 1:num_labels
     % Set Initial theta
     initial_theta = zeros(n + 1, 1);
     
     % Set options for fminunc
     options = optimset('GradObj', 'on', 'MaxIter', 50);
 
     % Run fmincg to obtain the optimal theta
     % This function will return theta and the cost 
     % (y == c) ...train for classifier 1 then 2 then 3 then 4...so on
     [theta] = ...
         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
     
     % set c'th row in all_theta with values from theta'
     % all_theta is of size 10x401 and theta is of size 401x1
     all_theta(c,:) = theta';
end

% =========================================================================


end
