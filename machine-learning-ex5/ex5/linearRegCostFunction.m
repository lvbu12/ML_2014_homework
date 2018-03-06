function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
y_predict=X*theta;
theta_tmp=[0;theta(2:end,:)];
J=(y_predict-y)'*(y_predict-y)/(2*m)+lambda*(theta_tmp'*theta_tmp)/(2*m);
%grad(1,:)=sum(y_predict-y)/m;
grad=((y_predict-y)'*X)'/m+lambda/m*theta_tmp;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
