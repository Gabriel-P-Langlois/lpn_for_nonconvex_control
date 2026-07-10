% Script for the Metropolis Adjusted Langevin Algorithm (MALA) to compute
% the (approximate) posterior mean estimate of the distribution
% exp(-Phi(y|x,t)/\epsilon) where Phi(y|x,t) = (t/2)H^*((x-y)/t) + J(y).

% Written by Gabriel P. Langlois

%% Log
% Optimizing for speed
% 275 seconds with v5normal vs 342 seconds, which is an improvement.
% 50000
% 200000

%% Load images
load('lenna_256x256_d');
load('lenna_256x256_d_n_g');
true_image = lenna_256x256_d;
noisy_image = lenna_256x256_d_n_g;
clear lenna_256x256_d lenna_256x256_d_n_g

%% Parameters and choice of prior
x = noisy_image(:);
n = length(true_image(1,:));
t = 0.2;
epsilon = 1;
m = 1; % Quadratic prior of strong convexity m

%% Map and PM estimates
% Quadratic prior of strong convexity m
u_map = x/(1+m*t);      % True value of the MAP estimate.
u_pm = x/(1+m*t);       % True value of the PM estimate.

%% PARAMETERS - MALA Algorithm
step_size = 0.190375*(t*epsilon/((n^(1/3))*(1+m*t)));
burn_in = 50000;                         % Initial burn-in (samples to be eliminated)
iter = 200000;                           % Number of iterations

%% INITIALIZATION AND PRECOMPUTED QUANTITIES
y1 = x; %y1 = u_map;                                % Initial step.
val1 = y1 + 0.5*step_size*gradF(y1,x,t,epsilon,m);  % grad descent step
pi1 = F(y1,x,t,epsilon,m);                          % distribution at step

y2 = zeros(n^2,1);                        % Next step
u_empirical = zeros(n^2,1);               % Empirical pm estimator.
grad_prior_emp = zeros(n^2,1);            % Empirical E{\gradJ(y)}
empirical_var = 0;

%% ALGORITHM
rng(2,'v5normal');                             % Random seed control set to v5normal, which is the fastest according to my experiments
acceptance_ratio = 0;
tic
for k=1:1:(burn_in-1)
    y2 = val1 + sqrt(step_size)*randn(n^2,1);     % candidate step
    val2 = y2 + 0.5*step_size*gradF(y2,x,t,epsilon,m);             
    pi2 = F(y2,x,t,epsilon,m);
    if(rand <= min(1,exp((pi2 - pi1) - (0.5/step_size)*(norm(y1 - val2)^2 - norm(y2 - val1)^2))))
        y1 = y2;                                % Accept
        val1 = val2;                            % Store this value
        pi1 = pi2;
    end
end

for k=burn_in:1:(burn_in+iter-1)
    y2 = val1 + sqrt(step_size)*randn(n^2,1);     % candidate step
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
error_l2_av = (norm(u_pm-u_empirical))/(n^2);   % Adjusted Euclidean norm
error_l2_norm = norm(u_pm-u_empirical);         % Euclidean norm
error_linf = norm(u_pm-u_empirical,inf);        % Infinity norm
scaled_empirical_var = empirical_var/((n^2)*t); % The true value of this will always be smaller or equal than epsilon.

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

%% Display denoised images
true_pm_denoised_image = reshape(u_pm,n,n);
mala_pm_denoised_image = reshape(u_empirical,n,n);

figure(1)
subplot(1,4,1)
imshow(true_image)
title('Uncorrupted image of Lenna')
subplot(1,4,2)
imshow(noisy_image)
title('Noisy (Gaussian) image of Lenna')
subplot(1,4,3)
imshow(true_pm_denoised_image)
title('Denoised (true pm estimator) of the image')
subplot(1,4,4)
imshow(mala_pm_denoised_image)
title('Denoised (mala pm estimator) of the image')

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