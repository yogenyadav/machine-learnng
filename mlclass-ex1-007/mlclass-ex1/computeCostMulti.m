function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%sum_of_square_of_difference = 0;
%hypothesis_value = 0;
%for x = 1:m
%	for z=1:length(theta)
%    	hypothesis_value = hypothesis_value + theta(z)*X(x,z);
%    end
%    sum_of_square_of_difference = sum_of_square_of_difference + ((hypothesis_value - y(x)) ^ 2);
%end

%J = sum_of_square_of_difference/(2*m);

h = theta' * X';
diff = h - y';
diff_sqr = diff .^ 2;
sum_diff_sqr = sum(diff_sqr);
J = sum_diff_sqr/(2*m);

% =========================================================================

end
