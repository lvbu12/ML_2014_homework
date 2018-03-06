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
nx=length(theta);
z=X*theta;
y_pred=sigmoid(z);
grad(1)=(y_pred-y)'*X(:,1)/m;
J=-(y'*log(y_pred)+(1-y)'*log(1-y_pred))/m+lambda/2*(theta(2:nx))'*(theta(2:nx))/m;
X_tmp=X(:,2:nx);
grad_tmp=((y_pred-y)'*X_tmp)'/m;
grad(2:nx)=grad_tmp+(lambda/m).*theta(2:nx);






% =============================================================

end
