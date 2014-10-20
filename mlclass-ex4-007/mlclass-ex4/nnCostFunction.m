function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

size(y);
%step-1:
%transform y such that y becomes 5000x10 matrix of 5000 examples and 10 classifiers
%column-1 is set to 1s if example represents 1
%column-2 is set to 1s if example represents 2...so on
%column-10 is set to 1s if example represents 0
z = zeros(m, num_labels);
for i=1:num_labels
	z(:,i) = y==i;
end
y = z;
size(y);

%step-2:
%add bias unit to X
X = [ones(m, 1) X];

%step-3:
%calculate hypothesis for layer-2
h_of_theta_x = sigmoid(X * Theta1');
size(h_of_theta_x);

%step-4:
%add bias unit to output of layer-2 which is h_of_theta_x
h_of_theta_x = [ones(size(h_of_theta_x, 1), 1) h_of_theta_x];
size(h_of_theta_x);

%step-5:
%calculate hypothesis for layer-3
h_of_theta_x = sigmoid(h_of_theta_x * Theta2');
size(h_of_theta_x);

%step-5
%calculate cost which is....
% final layer hypothisis values
% apply cost logistic regression function
J = (-1/m) * sum(sum(y .* log(h_of_theta_x) + (1-y) .* log(1-h_of_theta_x)));
size(J);

regularization = (lambda/(2*m)) * (sum(sum((Theta1(:,2:end)).^2)) + sum(sum((Theta2(:,2:end)).^2)));

J = J + regularization;

%J =J + (lambda/(2*m)) * (sum(sum(theta_1(:,2:end).^2,2)) + sum(sum(theta_2(:,2:end).^2,2)));

%this can be done like this as well.....
%X = [ones(m, 1) X];
%transform y such that y becomes 5000x10 matrix of 5000 examples and 10 classifiers
%column-1 is set to 1s if example represents 1
%column-2 is set to 1s if example represents 2...so on
%column-10 is set to 1s if example represents 0
%y = eye(num_labels)(y,:);
% 
% 
%a1 = X;
% 
%z2 = a1 * Theta1';
%a2 = sigmoid(z2);
% 
%n = size(a2, 1);
%a2 = [ones(n,1) a2];
% 
%z3 = a2 * Theta2';
%a3 = sigmoid(z3);
% 
%J = ((1/m) * sum(sum((-y .* log(a3))-((1-y) .* log(1-a3)))));

a1 = X;
z2 = a1 * Theta1';
a2 = sigmoid(z2);

n = size(a2, 1);
a2 = [ones(n,1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

delta_layer_3 = a3 - y;
size(delta_layer_3);

delta_layer_2 = (delta_layer_3 * Theta2(:,2:end)) .* sigmoidGradient(z2);
size(delta_layer_2);

delta_cap2 = delta_layer_3' * a2; 
delta_cap1 = delta_layer_2' * a1;

Theta1_grad = ((1/m) * delta_cap1) + ((lambda/m) * (Theta1));
Theta2_grad = ((1/m) * delta_cap2) + ((lambda/m) * (Theta2));
 
Theta1_grad(:,1) -= ((lambda/m) * (Theta1(:,1)));
Theta2_grad(:,1) -= ((lambda/m) * (Theta2(:,1)));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
