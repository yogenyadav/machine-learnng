function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

n = length(theta);
sum_theta_sq = 0;
for i=2:n
	sum_theta_sq = sum_theta_sq + theta(i) ^ 2;
end

J = -1*sum(x2+x3)/m;
J = J+lambda*sum_theta_sq/(2*m);

x3 = x - y'; %1xm matrix
x4 = x3 * X; %1xm * mxn = 1xn matrix
grad = x4'/m; %nx1 matrix
for i=2:n
	grad(i) = grad(i) + lambda*theta(i)/m;
end

% =============================================================

end
