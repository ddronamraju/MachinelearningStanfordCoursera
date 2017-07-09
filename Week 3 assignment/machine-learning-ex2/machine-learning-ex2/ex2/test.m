%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);

% Put some labels
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;


%% =========== Part 1: Regularized Logistic Regression ============
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic
%  regression to classify the data points.
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Initialize some useful values
m = length(y); % number of training examples
J=0;
  %J=(1/100)*sum(-y'*log(sigmoid(X))-((1-y)'*log(1-sigmoid(X))));
  %J=y'*log(sigmoid(X));


theta1=theta(2:size(theta));

thetar= [0;theta1];


J=(1/m)*sum(-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta)))+((lambda/(2*m))*sum(thetar.^2));

disp(J);
