function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


XX = X * theta; 											%mxn * nx1 = mx1
size(XX);
h_of_theta_xx = sigmoid(XX);
size(h_of_theta_xx);
log_h_of_theta_xx = log(h_of_theta_xx);
y_log_h_of_theta_xx = y .* log_h_of_theta_xx; 				%mx1 
size(y_log_h_of_theta_xx);

yy = 1 .- y;
h_of_theta_xx2 = 1 .- h_of_theta_xx;
log_h_of_theta_xx2 = log(h_of_theta_xx2);
y_log_h_of_theta_xx2 = yy .* log_h_of_theta_xx2;
size(y_log_h_of_theta_xx2);

J = -1*sum(y_log_h_of_theta_xx + y_log_h_of_theta_xx2)/m;
temp = theta;
temp(1) = 0;
temp = temp .^ 2;
J = J+lambda*sum(temp)/(2*m);


h_of_theta_x_minus_y = h_of_theta_xx - y; 					%mx1
size(h_of_theta_x_minus_y);
h_of_theta_x_minus_y_xj = h_of_theta_x_minus_y' * X; 		%1xm * mxn = 1xn
size(h_of_theta_x_minus_y_xj);
grad = h_of_theta_x_minus_y_xj'/m; 							%nx1
size(grad);
temp = theta;
temp(1) = 0;
grad = grad + (lambda*temp ./ m);


% =============================================================

grad = grad(:);

end
