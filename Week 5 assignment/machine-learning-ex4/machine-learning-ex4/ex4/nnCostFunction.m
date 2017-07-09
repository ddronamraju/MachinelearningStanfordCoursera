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

% Add ones to the X data matrix
% Size of X= 5000x400
X = [ones(m, 1) X];
% After adding bias Size of X= 5000x401

% This for loop is written to convert y into a matrix of veectors containing binary values. For example if the output is 10, the first column will have 10th row as 1 and remaining zeroes
yc=zeros(num_labels,m);
% Size of yc= 10x5000

for i=1:m
  yc(y(i),i)=1;
end

% This set of code is directly copied from the previous exercise to compute the output
a1=X;
z2=a1*Theta1';
a2=sigmoid(z2);

a2 = [ones(m, 1) a2];

z3=a2*Theta2';
a3=sigmoid(z3);

% You need to return the following variables correctly 
 

J=(1/m)*sum(sum((-yc').*log(a3)-(1-yc').*log(1-(a3))));

thetarr1= Theta1(:,2:size(Theta1,2));

thetarr2= Theta2(:,2:size(Theta2,2));
 
% regularization formula 
Regfact = lambda*(sum(sum(thetarr1.^2))+sum(sum(thetarr2.^2)))/(2*m);

J=J+Regfact;

for t=1:m
%step 1
  %Get only the row for the t'th training example
  a1_b = X(t,:);
 
  z2_b=Theta1*a1_b';
  a2_b=sigmoid(z2_b);
  
  a2_b= [1;a2_b];
  z3_b=Theta2*a2_b;
  a3_b=sigmoid(z3_b);
  
  %step 2
  d3=a3_b-yc(:,t);
  
  %step 3
  d2=(Theta2'*d3).*sigmoidGradient([1;z2_b]);
  
  %step 4
  Theta2_grad = Theta2_grad + d3 * a2_b'; 
  Theta1_grad = Theta1_grad + d2(2:end) * a1_b; 
 end
  %step 5

  Theta2_grad=Theta2_grad./m;
  Theta1_grad=Theta1_grad./m;
   
  
  Theta2_grad(:, 2:end)=Theta2_grad(:, 2:end)+((lambda/m) * Theta2(:, 2:end));
  Theta1_grad(:, 2:end)=Theta1_grad(:, 2:end)+((lambda/m) * Theta1(:, 2:end));
  
  %grad=[Theta1_lambda(:) ; Theta2_lambda(:)];
  
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
