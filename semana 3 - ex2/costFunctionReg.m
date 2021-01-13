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

% costo no regularizado
J_noReg = 0;
J_Reg = 0;
for i=1:m
  gz = sigmoid(X(i,:)*theta);
  J_noReg = J_noReg + (-1/m)*( y(i)*log(gz) + (1-y(i)) * log(1-gz) );       %perfecto
endfor

% costo regularizado
for i=2:size(theta,1)
  J_Reg = J_Reg + (lambda/(2*m))*theta(i)^2;           %perfecto
endfor

J = J_noReg + J_Reg;

% Gradiente
for i=1:m
    gz = sigmoid(X(i,:)*theta);
      for j=1:size(grad,1)
          grad(j) = grad(j) + (1/m) * (gz-y(i)) * X(i,j) ;       %perfecto
      endfor
endfor

for j=1:size(grad,1)
      grad(j) = grad(j) + (lambda/m) * theta(j)^2 ;       %perfecto
endfor

% =============================================================

end
