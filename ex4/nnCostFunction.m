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

% --------- No regularization ---------

% agregamos Xo y llamamos sigmoid. layer 1 to layer 2
tmp = [ones(size(X, 1), 1) X] * Theta1';    % z2
tmp = sigmoid(tmp);                         % a2

% agregamos ao y llamamos sigmoid. layer 2 to layer 3
tmp = [ones(size(tmp, 1), 1) tmp] * Theta2';    % z3
tmp = sigmoid(tmp);                           % a3

% transformando output
y2 = zeros(m, num_labels);
y2(sub2ind(size(y2), (1:length(y))', y)) = 1;

J = sum(sum(-y2 .* log(tmp) - (1 - y2) .* log(1 - tmp), 2)) / m;

% --------- Regularization ---------

Theta1_reg = Theta1;
Theta2_reg = Theta2;
% Not counting the bias unit - funciona para cualquier numero de unidades 
Theta1_reg(:, 1) = 0;
Theta2_reg(:, 1) = 0;
J = J + (lambda / (2 * m)) * (sum(sum((Theta1_reg .* Theta1_reg) , 2)) + ...
    sum(sum((Theta2_reg .* Theta2_reg) , 2)));

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

% Pasos 1 a 4
Theta2_t = Theta2';
for t = 1:m
    % paso 1
    a_1 = [1; X(t, :)'];           % es un vector columna
    z_2 = Theta1 * [a_1];  % es un vector columna
    a_2 = [1; sigmoid(z_2)];       % es un vector columna  
    z_3 = Theta2 * [a_2];  % es un vector columna
    a_3 = sigmoid(z_3);       % es un vector columna 
    
    % paso 2
    delta_3 = a_3 - y2(t, :)';
    
    % paso 3
    delta_2 = Theta2_t(2:end, :) * delta_3 .* sigmoidGradient(z_2); %Remove theta 0
    
    % paso 4
    Theta2_grad = Theta2_grad + delta_3 * [a_2]'; %se agrega el 1
    Theta1_grad = Theta1_grad + delta_2 * [a_1]';
    
endfor

% paso 5
tmp = lambda / m;

Theta2_grad = Theta2_grad / m + tmp * Theta2_reg;
Theta1_grad = Theta1_grad / m + tmp * Theta1_reg;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
