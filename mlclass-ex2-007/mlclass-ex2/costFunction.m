function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% block below is vectorized sigmoid value for each hypothesis for each training set
x = theta' * X'; %1xn * nxm = 1xm matrix
x = e .^ (-x);
x = 1 + x;
x = 1 ./ x;

x2 = zeros(1, size(x,2)); %1xm matrix
for i=1:m
	x2(i) = log(x(i)) * y(i);
end

z = zeros(1, size(x,2)); %1xm matrix
for i=1:m
	z(i) = 1 - x(i);
end

y2 = 1 .- y;

x3 = zeros(1, size(x,2)); %1xm matrix
for i=1:m
	x3(i) = log(z(i)) * y2(i);
end

J = -1*sum(x2+x3)/m;


x3 = x - y'; %1xm matrix
x4 = x3 * X; %1xm * mxn = 1xn matrix
grad = x4'/m; %nx1 matrix

% =============================================================

end
