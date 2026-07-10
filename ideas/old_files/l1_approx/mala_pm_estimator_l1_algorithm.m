% Script for the Metropolis Adjusted Langevin Algorithm (MALA) to compute
% the (approximate) posterior mean estimate of the distribution
% exp(-Phi(y|x,t)/\epsilon) where Phi(y|x,t) = (t/2)H^*((x-y)/t) + J(y).

% Written by Gabriel P. Langlois

%% Log
% Optimizing for speed
% 715 seconds with v5normal
% 50000
% 200000

%% Load images
load('lenna_256x256_d');
load('lenna_256x256_d_n_g');
load('lenna_256x256_d_n_sp');
true_image = lenna_256x256_d;
g_noisy_image = lenna_256x256_d_n_g;
sp_noisy_image = lenna_256x256_d_n_sp;
clear lenna_256x256_d lenna_256x256_d_n_g lenna_256x256_d_n_sp

%% PARAMETERS - Energy, prior, and distribution
x = g_noisy_image(:);
n = length(true_image(1,:));
t = 0.2;
epsilon = 1.25;

% Moreau-Yosida approx, thresholding operator
lambda = 0.1*ones(n^2,1);
mu = 0.00001;

%% True and approximate MAP AND PM ESTIMATES

u_map = max(0,x-lambda*(t+mu)) + min(0,x+lambda*(t+mu));
u_map_approx = max(0,x-lambda*t) + min(0,x+lambda*t);
u_pm_approx = pm_est_l1norm(x,t,epsilon,lambda); % Not the real PM, only a approximation

%% PARAMETERS - MALA Algorithm
step_size = 0.0375*(t*epsilon/((n^(2/3))));
burn_in = 50000;                         % Initial burn-in (samples to be eliminated)
iter = 200000;                           % Number of iterations

%% INITIALIZATION AND PRECOMPUTED QUANTITIES
y1 = x; %y1 = u_map;                                 % Initial step. Set this to u_map if available.
val1 = y1 + 0.5*step_size*gradF(y1,x,t,epsilon,lambda,mu);  % grad descent step
pi1 = F(y1,x,t,epsilon,lambda,mu);                          % distribution at step

y2 = zeros(n^2,1);                        % Next step
u_empirical = zeros(n^2,1);               % Empirical pm estimator.
grad_prior_emp = zeros(n^2,1);            % Empirical E{\gradJ(y)}
empirical_var = 0;

%% ALGORITHM
rng(2,'v5normal');                                         % Random seed control
acceptance_ratio = 0;
tic
for k=1:1:(burn_in-1)
    y2 = val1 + sqrt(step_size)*randn(n^2,1);     % candidate step
    val2 = y2 + 0.5*step_size*gradF(y2,x,t,epsilon,lambda,mu);             
    pi2 = F(y2,x,t,epsilon,lambda,mu);
    if(rand <= min(1,exp((pi2 - pi1) - (0.5/step_size)*(norm(y1 - val2)^2 - norm(y2 - val1)^2))))
        y1 = y2;                                % Accept
        val1 = val2;                            % Store this value
        pi1 = pi2;
    end
end

for k=burn_in:1:(burn_in+iter-1)
    y2 = val1 + sqrt(step_size)*randn(n^2,1);     % candidate step
    val2 = y2 + 0.5*step_size*gradF(y2,x,t,epsilon,lambda,mu);             
    pi2 = F(y2,x,t,epsilon,lambda,mu);
    if(rand <= min(1,exp((pi2 - pi1) -(0.5/step_size)*(norm(y1 - val2)^2 - norm(y2 - val1)^2))))         
        y1 = y2;
        val1 = val2;                            % Store this value
        pi1 = pi2;
        acceptance_ratio = acceptance_ratio + 1/iter;
    end
    empirical_var = empirical_var + (norm(y1-u_pm_approx)^2)/iter;
    u_empirical = u_empirical + y1/iter;        % Store this value
    grad_prior_emp = grad_prior_emp + grad_prior(y1,lambda,mu)/iter;
end
toc

%% ERROR ANALYSIS
error_l2_av_grad_prior = (norm(u_pm_approx-(x-t*grad_prior_emp)))/(n^2);    % Adjusted Euclidean norm
error_l2_norm_grad_prior = norm(u_pm_approx-(x-t*grad_prior_emp));          % Euclidean norm
error_linf_grad_prior = norm(u_pm_approx-(x-t*grad_prior_emp),inf);         % Infinity norm
scaled_empirical_var = empirical_var/((n^2)*t); % The true value of this will always be smaller or equal than epsilon.
disp(['After ',num2str(burn_in+iter),' iterations and discarding the first ',num2str(burn_in), ...
    ' samples, we get the following error estimates for empirical approximation of u_pm: '])
disp(' ')
disp(['The l2 norm of the error divided by the dimension is '...
    ,num2str(error_l2_av_grad_prior),'.'])
disp(['The l2 norm of the error is '...
    ,num2str(error_l2_norm_grad_prior),'.'])
disp(['The linfinity norm of the error is '...
    ,num2str(error_linf_grad_prior),'.'])
disp(['The empirical variance divided by the upper bound for the true variance is '...
    ,num2str(scaled_empirical_var),'.'])
disp(' ')
disp(['The acceptance ratio in the MALA algorithm is ',num2str(acceptance_ratio),'.'])
disp(' ')

%% Display denoised images
true_pm_denoised_image = reshape(u_pm_approx,n,n);
mala_pm_denoised_image = reshape(x-t*grad_prior_emp,n,n);

figure(1)
subplot(1,4,1)
imshow(true_image)
title('Uncorrupted image of Lenna')
subplot(1,4,2)
imshow(reshape(x,n,n))
title('Noisy image of Lenna')
subplot(1,4,3)
imshow(true_pm_denoised_image)
title('Denoised (true pm estimator) of the image')
subplot(1,4,4)
imshow(mala_pm_denoised_image)
title('Denoised (mala pm estimator) of the image')

%% HELPER FUNCTIONS
function val = F(y,x,t,epsilon,lambda,mu)
    val = -((0.5/t)*norm((x-y))^2 + prior(y,lambda,mu))/epsilon; 
    
    function prior_val = prior(y,lambda,mu)
        prior_val = (0.5/mu)*norm(y-(max(0,y-lambda*(mu)) + min(0,y+lambda*(mu))))^2 + lambda.*abs(max(0,y-lambda*(mu)) + min(0,y+lambda*(mu)));
    end
end

function vector = gradF(y,x,t,epsilon,lambda,mu)
    vector = -((y-x)/t + grad_prior(y,lambda,mu))/epsilon;
end

function prior_grad_vector = grad_prior(y,lambda,mu)
    prior_grad_vector = (y-(max(0,y-lambda*(mu)) + min(0,y+lambda*(mu))))/mu;
end

function vector = pm_est_l1norm(x,t,epsilon,lambda)
    v_plus  = (x + t*lambda);
    v_minus = (-x + t*lambda);
    ratio_plus  = erfcx(v_plus./sqrt(2*t*epsilon))./(erfcx(v_plus./sqrt(2*t*epsilon)) + erfcx(v_minus./sqrt(2*t*epsilon)));
    ratio_minus = erfcx(v_minus./sqrt(2*t*epsilon))./(erfcx(v_plus/sqrt(2*t*epsilon)) + erfcx(v_minus./sqrt(2*t*epsilon)));
    vector = v_plus.*ratio_plus - v_minus.*ratio_minus;
end