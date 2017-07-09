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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


%J=(1/(2*m))*sum(((X*theta)-y).^2)+((lambda/(2*m))*sum(thetar.^2))

theta1=theta(2:size(theta));

thetar= [0;theta1];

J=(1/(2*m))*sum(((X*theta)-y).^2)+((lambda/(2*m))*sum(thetar.^2));


%Theta1_grad=(1/m)*(sum(((X*theta)-y)));

grad=(1/m)*(X'*((X*theta)-y));
theta_reg=theta;
theta_reg(1,1)=0;

grad_reg=(lambda/m)*theta_reg;

grad=grad+grad_reg;

%{
Theta2_grad=Theta2_grad./m;
Theta1_grad=Theta1_grad./m;
   
  
Theta2_grad(:, 2:end)=Theta2_grad(:, 2:end)+((lambda/m) * Theta2(:, 2:end));
Theta1_grad(:, 2:end)=Theta1_grad(:, 2:end)+((lambda/m) * Theta1(:, 2:end));
  
  %grad=[Theta1_lambda(:) ; Theta2_lambda(:)];
  
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%}
% =========================================================================

grad = grad(:);

end
