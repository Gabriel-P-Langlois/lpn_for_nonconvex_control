% Script for the Metropolis Adjusted Langevin Algorithm (MALA).
% Written by Gabriel P. Langlois

% It is currently adapted to the distribution
% propotional to exp(-Phi(y|x,t)/\epsilon), where
%               Phi(y|x,t) = (t/2)H^*((x-y)/t) + J(y)
% where
%               (t/2)H^*((x-y)/t) = (0.5/t)*norm(x-y)^2
% and
%               J(y) = m*(||y||^2)/2.

%% Log journal
% Adjustement for the step size? How well should it scale?
% Acceptance rate - add a measure of it.
% At equal iterations, the error doubles every time the dimension goes up
% by 4. 

%% PARAMETERS - Energy, prior, and distribution
n = 64^2;
x = randn(n,1);
t = 1;
epsilon = 1;

% Quadratic prior of strong convexity m
m = 1;

%% MAP AND PM ESTIMATES (if available)
% Quadratic prior of strong convexity m
u_map = x/(1+m*t);      % True value of the MAP estimate.
u_pm = x/(1+m*t);       % True value of the PM estimate.

%% PARAMETERS - MALA Algorithm
step_size = 3*(t*epsilon/((n^(1/3))*(1+m*t)));
burn_in = 40000;                         % Initial burn-in (samples to be eliminated)
iter = 500000;                           % Number of iterations

%% INITIALIZATION AND PRECOMPUTED QUANTITIES
y1 = x; %y1 = u_map;                                % Initial step.
val1 = y1 + 0.5*step_size*gradF(y1,x,t,epsilon,m);  % grad descent step
pi1 = F(y1,x,t,epsilon,m);                          % distribution at step

y2 = zeros(n,1);                        % Next step
u_empirical = zeros(n,1);               % Empirical pm estimator.
grad_prior_emp = zeros(n,1);            % Empirical E{\gradJ(y)}
empirical_var = 0;

%% ALGORITHM
rng(2);                                         % Random seed control
acceptance_ratio = 0;
tic
for k=1:1:(burn_in-1)
    y2 = val1 + sqrt(step_size)*randn(n,1);     % candidate step
    val2 = y2 + 0.5*step_size*gradF(y2,x,t,epsilon,m);             
    pi2 = F(y2,x,t,epsilon,m);
    if(rand <= min(1,exp((pi2 - pi1) - (0.5/step_size)*(norm(y1 - val2)^2 - norm(y2 - val1)^2))))
        y1 = y2;                                % Accept
        val1 = val2;                            % Store this value
        pi1 = pi2;
    end
end

for k=burn_in:1:(burn_in+iter-1)
    y2 = val1 + sqrt(step_size)*randn(n,1);     % candidate step
    val2 = y2 + 0.5*step_size*gradF(y2,x,t,epsilon,m);             
    pi2 = F(y2,x,t,epsilon,m);
    if(rand <= min(1,exp((pi2 - pi1) -(0.5/step_size)*(norm(y1 - val2)^2 - norm(y2 - val1)^2))))         
        y1 = y2;
        val1 = val2;                            % Store this value
        pi1 = pi2;
        acceptance_ratio = acceptance_ratio + 1/iter;
    end
    empirical_var = empirical_var + (norm(y1-u_pm)^2)/iter;
    u_empirical = u_empirical + y1/iter;
    grad_prior_emp = grad_prior_emp + grad_prior(y1,m)/iter;
end
toc

%% ERROR ANALYSIS
error_l2_av = (norm(u_pm-u_empirical))/n; % Adjusted Euclidean norm
error_l2_norm = norm(u_pm-u_empirical);   % Euclidean norm
error_linf = norm(u_pm-u_empirical,inf);  % Infinity norm
scaled_empirical_var = empirical_var/(n*t); % The true value of this will always be smaller or equal than epsilon.

% Same error as above, which is correct.
error_l2_av_grad_prior = (norm(u_pm-(x-t*grad_prior_emp)))/n; % Adjusted Euclidean norm
error_l2_norm_grad_prior = norm(u_pm-(x-t*grad_prior_emp));   % Euclidean norm
error_linf_grad_prior = norm(u_pm-(x-t*grad_prior_emp),inf);  % Infinity norm

disp(['After ',num2str(burn_in+iter),' iterations and discarding the first ',num2str(burn_in), ...
    ' samples, we get the following error estimates for empirical approximation of u_pm: '])
disp(' ')
disp(['The l2 norm of the error divided by the dimension is '...
    ,num2str(error_l2_av),'.'])
disp(['The l2 norm of the error is '...
    ,num2str(error_l2_norm),'.'])
disp(['The linfinity norm of the error is '...
    ,num2str(error_linf),'.'])
disp(['The empirical variance divided by the upper bound for the true variance is '...
    ,num2str(scaled_empirical_var),'.'])
disp(' ')
disp(['The acceptance ratio in the MALA algorithm is ',num2str(acceptance_ratio),'.'])
disp(' ')

plot(x,u_empirical,'linewidth',1,'DisplayName',['Estimate of the posterior mean.'])
hold
plot(x,x)

%% HELPER FUNCTIONS
function val = F(y,x,t,epsilon,m)
    val = -((0.5/t)*norm((x-y))^2 + 0.5*m*(norm(y)^2))/epsilon; 
end

function vector = gradF(y,x,t,epsilon,m)
    vector = -((y-x)/t + m*y)/epsilon;
end

function prior_grad_vector = grad_prior(y,m)
    prior_grad_vector = m*y;
end