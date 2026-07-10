function [ solution_heat, output_posterior_mean ] = posterior_mean_l1(x_eval,t, epsilon)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here



% definition gaussian
gaussian = @(y, x, t) 1/sqrt(4*pi*t*epsilon) * exp( -(x-y).^2 / (4*t*epsilon));
%definition laplace
b=2*epsilon; mu=0;
%laplace = @(y) 1/(2*b) * exp( - abs(y-mu)/(b) );
laplace =  @(y) exp( - abs(y-mu)/(b));

%definition posterior
posterior_density = @(y,x,t) gaussian(y,x,t) .* laplace(y) ;

posterior_mean_to_integrate = @(y,x,t) y.*gaussian(y,x,t) .* laplace(y) ;

for i=1:size(x_eval,2)
   solution_heat(i) = integral(@(y) posterior_density(y,x_eval(i),t) , -Inf, Inf, 'AbsTol', 1e-16);
   output_posterior_mean(i) =  integral(@(y) posterior_mean_to_integrate(y,x_eval(i),t) , -Inf, Inf, 'AbsTol', 1e-16)/   solution_heat(i);
end


% example
% x_eval = linspace(-4,4,1000);
%[proba, post_mean] = posterior_mean_l1(x_eval,1,1/2); plot(x_eval,post_mean); grid on
