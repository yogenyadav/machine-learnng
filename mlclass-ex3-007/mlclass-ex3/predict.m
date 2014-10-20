function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

%Theta1 - for layer 2, its 25x401 - parameters for a unit is in corresponding row
%Theta2 - for layer 3, its 10x26

% each unit of layer 2 should be sum of products of training example and Theta values for that unit in Theta1
% therefore by transposing Theta' and multipying with X you can get that result.
layer2_output = sigmoid(X * Theta1'); %5000x401 * 401x25 = 5000x25

% add bias unit of all 1s
layer2_output = [ones(m, 1), layer2_output]; %5000x26

% similar to how layer 2 calculation was done, done for layer 3
layer3_output = sigmoid(layer2_output * Theta2'); %5000x26 * 26x10 = 5000x10

% rest is similar to done in predictOneVsAll
[max_prob, ix] = max(layer3_output, [], 2);

%return the predicted digit in p
p = ix;


% =========================================================================


end
