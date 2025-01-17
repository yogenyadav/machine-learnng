function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    sum_for_theta0 = 0;
    sum_for_theta1 = 0;
    for x = 1:m
        sum_for_theta0 = sum_for_theta0 + ((theta(1) + theta(2) * X(x,2)) - y(x)) * X(x,1);
        sum_for_theta1 = sum_for_theta1 + ((theta(1) + theta(2) * X(x,2)) - y(x)) * X(x,2);
    end
    d0 = (sum_for_theta0 * alpha)/m;
    d1 = (sum_for_theta1 * alpha)/m;

    theta(1) = theta(1) - d0;
    theta(2) = theta(2) - d1;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
