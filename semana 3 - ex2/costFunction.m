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

% ==== For loop version ====
#{
for i=1:m
  gz = sigmoid(X(i,:)*theta);
  J = J + (-1/m)*( y(i)*log(gz) + (1-y(i)) * log(1-gz) );       %perfecto
endfor

for i=1:m
    gz = sigmoid(X(i,:)*theta);
      for j=1:size(grad,1)
          grad(j) = grad(j) + (1/m) * (gz-y(i)) * X(i,j) ;       %perfecto
      endfor
endfor
#} 

% ==== Vectorized version ====

hx = sigmoid(X * theta);
m = length(X);

J = (-y' * log(hx) - (1 - y')*log(1 - hx)) / m;
grad = X' * (hx - y) / m;


% =============================================================

end
