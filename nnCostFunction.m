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
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Fix Y vector
yMatrix = zeros(num_labels,m);
for i = 1:num_labels,
    yMatrix(i,:) = (y==i);
end;
       
% Step 1: Feedforward and Cost Function
X = [ones(m,1) X];

a1 = X;
z2 = a1*Theta1'
a2 = sigmoid(z2)

a2 = [ones(m, 1) a2];
z3 = a2*Theta2'
a3 = sigmoid(z3)
h = a3;

J = sum(sum((-1*yMatrix)'.*(log(h))-(1-yMatrix)'.*log(1-(h)), 2))/m

% Step 2: Add regularization
Reg = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

J = J + Reg;

% Step 3: Backpropagation
for t = 1:m,
    % BP Step 1:
    a1 = X(t,:);
    z2 = Theta1 * a1';
    a2 = sigmoid(z2);
    
    a2 = [1; a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    % BP Step 2:
    delta_3 = a3 - (yMatrix(:,t));
    
    % BP Step 3:
    z2 = [1; z2];
    delta_2 = (Theta2'*delta_3).*sigmoidGradient(z2);
    
    % BP Step 4:
    delta_2 = delta_2(2:end);
    Theta2_grad = Theta2_grad + delta_3*(a2)';
    Theta1_grad = Theta1_grad + delta_2*(a1);
  
end;

% BP Step 5:
Theta2_grad = (1/m).*Theta2_grad;
Theta1_grad = (1/m).*Theta1_grad;

    
% Step 4: Gradient Regularization
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
% -------------------------------------------------------------

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
