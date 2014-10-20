function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

sum_for_theta = theta;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    h = theta' * X';
    diff = h - y';
    diff_2 = diff * X;

    theta = theta - (diff_2'*alpha)/m;


%    for x = 1:m
%        for z=1:length(theta)
%            sum_for_theta(z) = sum_for_theta(z) + theta(z)*X(x,z);
%        end
%        for z=1:length(theta)
%            sum_for_theta(z) = (sum_for_theta(z) - y(x)) * X(x,z);
%        end
%    end
%    derivative = zeros(length(theta), 1);
%    for z=1:length(theta)
%        ee = (sum_for_theta(z) * alpha)/m;
%        derivative(z) = (sum_for_theta(z) * alpha)/m;
%    end

%    for z=1:length(theta)
%        theta(z) = theta(z) - derivative(z);
%    end


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
