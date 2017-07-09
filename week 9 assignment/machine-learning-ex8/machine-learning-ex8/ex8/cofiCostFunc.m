function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%Implementation Tip:You can use the R matrix to set selected entries to 0.
%For example, R .* M will do an element-wise multiplication between M
%and R; since R only has elements with values either 0 or 1, this has the
%effect of setting the elements of M to 0 only when the corresponding value
%in R is 0. Hence, sum(sum(R.*M)) is the sum of all the elements of M for
%which the corresponding element in R equals 1.

J=(1/2)*sum(sum(R.*((X*Theta'-Y).^2)));
J=J+(lambda*sum(sum(Theta.^2))/2)+(lambda*sum(sum(X.^2))/2);

%You should come up with a way to compute all the derivatives
%associated with x(i) (i.e., the derivative terms associated with
%the feature vector x(i)) at the same time.

%when you consider the features for the i-th movie,
%you only need to be concern about the users who had given ratings to the
%movie, and this allows you to remove all the other users from Theta and Y.

%following expression gives all users that rated movie i
for i=1:size(R,1)
  idx=find(R(i,:)==1);
   %Apply idx to X and Y to give you only the set of movies that have rated the i-th movie.
  
  Thetatemp = Theta(idx,:);
  Ytemp = Y(i,idx);
  X_grad(i,:) = ((X(i,:)*Thetatemp')-Ytemp)*Thetatemp;
  X_grad(i,:) = X_grad(i,:)+lambda*X(i,:);
end

for j=1:size(R,2)
 idt=find(R(:,j)==1);
 %Apply idx to Theta and Y to give you only the set of users which have been rated by jth user.
  Xtemp = X(idt,:);
  Ytemp = Y(idt,j);
  Theta_grad(j,:) = ((Xtemp*Theta(j,:)')-Ytemp)'*Xtemp;
  Theta_grad(j,:) = Theta_grad(j,:)+lambda*Theta(j,:);
end
% =============================================================


grad = [X_grad(:); Theta_grad(:)];



end
