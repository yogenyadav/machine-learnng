function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

sum_of_square_of_difference = 0;
for x = 1:m
    sum_of_square_of_difference = sum_of_square_of_difference + ((theta(1) + theta(2) * X(x,2)) - y(x)) ^ 2;
end

J = sum_of_square_of_difference/(2*m);


% =========================================================================

end
