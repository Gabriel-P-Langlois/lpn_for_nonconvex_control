% Script for the Metropolis Adjusted Langevin Algorithm (MALA).
% Written by Gabriel P. Langlois

% It is currently adapted to the distribution
% propotional to exp(-Phi(y|x,t)/\epsilon), where
%               Phi(y|x,t) = (0.5/t)*norm(x-y)^2 + J_mu(y), where
%               J_mu(y) = Moreau-Yosida approx of
%               \sum_{j=1}^{n} \lambda_j\abs{y_j}

%% PARAMETERS - Energy, prior, and distribution
x = (-5.0:0.01:5.0)';
n = length(x);
t = 1.25;
epsilon = 0.05;
lambda = 2*ones(n,1);
mu = 0.01; % Moreau-Yosida approx

%% MAP AND PM ESTIMATES
u_map_approx = max(0,x-lambda*(t)) + min(0,x+lambda*(t));
u_pm_approx = pm_est_l1norm(x,t,epsilon,lambda); % Not the real PM, only a approximation

%% PARAMETERS - MALA Algorithm
step_size = (1/(n^(1/3)))*(t*epsilon); % Constant, and purely approximate.
burn_in = 10000;                         % Initial burn-in (samples to be eliminated)
iter = 100000;                           % Number of iterations

%% INITIALIZATION AND PRECOMPUTED QUANTITIES
y1 = x; y1 = u_map;                                 % Initial step. Set this to u_map if available.
val1 = y1 + 0.5*step_size*gradF(y1,x,t,epsilon,lambda,mu);  % grad descent step
F1 = F(y1,x,t,epsilon,lambda,mu);                          % distribution at step

step2 = zeros(n,1);                        % Next step
u_empirical = zeros(n,1);               % Empirical pm estimator.
grad_prior_emp = zeros(n,1);            % Empirical E{\gradJ(y)}
empirical_var = 0;

%% ALGORITHM
rng(2);                                         % Random seed control
acceptance_ratio = 0;
tic
for k=1:1:(burn_in-1)
    step2 = val1 + sqrt(step_size)*randn(n,1);      % candidate step
    val2 = step2 + 0.5*step_size*gradF(step2,x,t,epsilon,lambda,mu);             
    F2 = F(step2,x,t,epsilon,lambda,mu);
    if(rand <= min(1,exp(-((2*step_size)*(F1-F2) + norm(y1 - val2)^2 - norm(step2 - val1)^2)/(2*step_size))))
        y1 = step2;                                 % Accept
        val1 = val2;
        F1 = F2;
    end
end

for k=burn_in:1:(burn_in+iter-1)
    step2 = val1 + sqrt(step_size)*randn(n,1);      % candidate step
    val2 = step2 + 0.5*step_size*gradF(step2,x,t,epsilon,lambda,mu);             
    F2 = F(step2,x,t,epsilon,lambda,mu);
    if(rand <= min(1,exp(-((2*step_size)*(F1-F2) + norm(y1 - val2)^2 - norm(step2 - val1)^2)/(2*step_size))))       
        y1 = step2;                                 % Accept
        val1 = val2;
        F1 = F2;
        acceptance_ratio = acceptance_ratio + 1/iter;
    end
    empirical_var = empirical_var + (norm(y1-u_pm_approx)^2)/iter;
    u_empirical = u_empirical + y1/iter;        % Store this value
    grad_prior_emp = grad_prior_emp + grad_prior(x+sqrt(t*epsilon)*y1,lambda,mu)/iter;
end
toc
u_empirical = x + sqrt(t*epsilon)*u_empirical;
%% ERROR ANALYSIS
error_l2_av = (norm(u_pm_approx-u_empirical))/n; % Adjusted Euclidean norm
error_l2_norm = norm(u_pm_approx-u_empirical);   % Euclidean norm
error_linf = norm(u_pm_approx-u_empirical,inf);  % Infinity norm
scaled_empirical_var = empirical_var/(n*t); % The true value of this will always be smaller or equal than epsilon.

error_l2_av_grad_prior = (norm(u_pm_approx-(x-t*grad_prior_emp)))/n; % Adjusted Euclidean norm
error_l2_norm_grad_prior = norm(u_pm_approx-(x-t*grad_prior_emp));   % Euclidean norm
error_linf_grad_prior = norm(u_pm_approx-(x-t*grad_prior_emp),inf);  % Infinity norm

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

%disp(num2str(error_l2_av_grad_prior))
%disp(num2str(error_l2_norm_grad_prior))
%disp(num2str(error_linf_grad_prior))
%disp(' ')

% Diagnosis tools
plot(x,u_pm_approx);
hold
plot(x,u_map_approx);
plot(x,u_empirical);
plot(x,grad_prior_emp);

%% HELPER FUNCTIONS
function vector = pm_est_l1norm(x,t,epsilon,lambda)   % For the thresholding l1norm operator
    v_plus  = (x + t*lambda);
    v_minus = (-x + t*lambda);
    ratio_plus  = erfcx(v_plus./sqrt(2*t*epsilon))./(erfcx(v_plus./sqrt(2*t*epsilon)) + erfcx(v_minus./sqrt(2*t*epsilon)));
    ratio_minus = erfcx(v_minus./sqrt(2*t*epsilon))./(erfcx(v_plus/sqrt(2*t*epsilon)) + erfcx(v_minus./sqrt(2*t*epsilon)));
    vector = v_plus.*ratio_plus - v_minus.*ratio_minus;
end

function val = F(y,x,t,epsilon,lambda,mu)
    val = -(0.5*norm(y)^2 + prior(x+sqrt(t*epsilon)*y,lambda,mu)/epsilon);
    
    function prior_val = prior(y,lambda,mu)
        u_map_min = max(0,y - mu*lambda) + min(0,y + mu*lambda);
        prior_val = (0.5/mu)*(norm(y-u_map_min)^2) + lambda.*abs(u_map_min);
    end
end

function vector = gradF(y,x,t,epsilon,lambda,mu)
    vector = -(y + sqrt(t/epsilon)*grad_prior(x+sqrt(t*epsilon)*y,lambda,mu));
    
    function prior_grad_vector = grad_prior(y,lambda,mu)
        u_map_min = max(0,y - mu*lambda) + min(0,y + mu*lambda);
        prior_grad_vector = (y - u_map_min)/mu;
    end
end

function prior_grad_vector = grad_prior(y,lambda,mu)
    u_map_min = max(0,y - mu*lambda) + min(0,y + mu*lambda);
    prior_grad_vector = (y - u_map_min)/mu;
end