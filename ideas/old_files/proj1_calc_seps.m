%% Script that computes S_eps(x,t).
%  Written by Gabriel P. Langlois

%% Notes: Observation: When I sample from N(u_map,t*epsilon) exactly, the
% value of Phi(y) is almost always the exact same!

%% Initialization -- Free parameters
% Randomization and reproducibility
rng(2,'v5normal');

% Variables - scale, noise, and smoothing parameters
scale = 255;            % grayscale = [0,255]
m = 10000;                 % m: Number of samples
sigma = 10/scale;
lambda = 32/scale;      % See table above

% Load images
image = imresize(imread('barbara_512x512.png'),[256,256]); % Barbara
image_d = im2double(image);
image_d_n_g = imnoise(image_d,'gaussian',0,(sigma)^2);

%% Initialization -- Do not modify
% Parameters of TV and probability distribution
x = image_d_n_g;
epsilon = 2*(sigma^2)/lambda;
t = (sigma^2)/epsilon;

% Parameters for the algorithm
size = length(image_d);             % Size of the image (size x size)
R = size^2;       

% MAP estimate
u_map = reshape(SB_ATV_mod2(x(:),lambda*0.5),size,size);

%% Algorithms

% 1) Sample from a standard gaussian and evaluate exp(-(1/epsilon)*J(x+sqrt(t*epsilon)*z))

J_terms1 = zeros(m,1);
for i=1:1:m
   J_terms1(i) = ATV(x + sqrt(t*epsilon)*randn(size,size));
end
min_terms1 = min(J_terms1);
scaled_terms1 = J_terms1 - min_terms1;
sum_terms1 = sum(exp(-scaled_terms1/epsilon));
Seps1 = min_terms1 - epsilon*log(sum_terms1);

% 2) Sample using IS from a Gaussian centered at u_map

norm_terms2 = zeros(m,1);
J_terms2 = zeros(m,1);
temp2 = zeros(m,1);
for i=1:1:m
   temp2 = randn(size,size);
   J_terms2(i) = ATV(u_map + sqrt(t*epsilon)*temp2);
   norm_terms2(i) = -epsilon*0.5*(norm(temp2,'fro')^2) + epsilon*0.5*(norm(temp2 - (x-u_map)/sqrt(t*epsilon),'fro')^2);
end
min_terms2 = min(J_terms2 + norm_terms2);
scaled_terms2 = J_terms2 + norm_terms2 - min_terms2;
sum_terms2 = sum(exp(-scaled_terms2/epsilon));
Seps2 = min_terms2 - epsilon*log(sum_terms2);

%% %% Algorithm 2 - u_map version
% J_umap = ATV(u_map);
% Szero = (0.5/t)*norm(x-u_map,'fro')^2 + J_umap;
% 
% % Sample from a Gaussian centered around u_map with variance t*epsillon
% % and evaluate exp(-(1/epsilon)*(Phi(y_i) - Szero))*ind([0,1]^n);
% 
% temp = zeros(size,size);      % random number
% J_terms = zeros(m,1);
% norm_terms1 = zeros(m,1);
% norm_terms2 = zeros(m,1);
% exp_terms = zeros(m,1);
% 
% test = zeros(m,1);
% test2 = zeros(m,1);
% 
% temp2 = zeros(size,size);
% J_terms2 = zeros(m,1);
% 
% % Iterations
% i = 1;
% while i <= m
%     
%     i = i + 1;
% end
% 
% % for i=1:1:m
% %     % Sample until one admissible value is sampled
% %     temp = u_map + sqrt(t*epsilon)*randn(size,size);
% %     temp2 = x + sqrt(t*epsilon)*randn(size,size);
% %     
% %     J_terms(i) = ATV(temp);
% %     norm_terms1(i) = (0.5/t)*norm(x-temp,'fro')^2;
% %     norm_terms2(i) = (0.5/t)*norm(u_map-temp,'fro')^2;
% %     exp_terms(i) = norm_terms1(i) + J_terms(i) - norm_terms2(i);
% %     
% %     J_terms2(i) = ATV(temp2);
% %     
% %     test(i) = dot(x(:)-temp(:),(x(:)-u_map(:))/t);
% %     test2(i) = -(t/2)*(norm((x(:)-u_map(:))/t,'fro')^2);
% % end
% 
% %% Test 1
% average_val = (1/m)*sum(exp_terms);
% diff = exp_terms - average_val;
% min_terms = min(diff);
% scaled_terms = diff-min_terms;
% 
% average_val2 = (1/m)*sum(J_terms2);
% diff2 = J_terms2 - average_val2;
% min_terms2 = min(diff2);
% scaled_terms2 = diff2-min_terms2;
% 
% Seps1 = average_val + min_terms;
% Seps2 = average_val2 + min_terms2;

%% Test 2
%scaled_terms = diff - min_terms;
%sum_exp = sum(exp(-(1/epsilon)*scaled_terms));
%Seps1 = average_val + min_terms - epsilon*log((1/m)*sum_exp);

% %% Algorithm 2 - Breg_div
% 
% % Load u_pm
% J_upm = ATV(u_pm);
% 
% temp = zeros(size,size);      % random number
% J_val2 = zeros(m,1);
% exp_terms2 = zeros(m,1);
% 
% % Iterations
% for i=1:1:m
%     temp = u_pm + sqrt(t*epsilon)*randn(size,size);
%     J_val2(i) = ATV(temp);
%     exp_terms2(i) = ((0.5/t)*norm(x-temp,'fro')^2 - (0.5/t)*norm(u_pm-temp,'fro')^2) + (J_val2(i)-J_upm);
% end
% 
% min_terms2 = min(exp_terms2);
% scaled_terms2 = exp_terms2 - min_terms2;
% sum_exp2 = sum(exp(-(1/epsilon)*scaled_terms2));
% Seps2 = J_upm + min_terms2 - epsilon*log((1/m)*sum_exp2);
% 
% % Tests
% 

%% Helper functions %%%
function val = P_ratio(x,t,temp_rand,y_lm,temp_ind,size,A) % P_ratio(x,t,temp_rand,v)
%   P_ratio computes the Metropolis-Hasting step in the MCMC algorithm
%   
%   Input:
%   A:          Image proposal
%   x:          Noisy image
%   t:          Smoothing parameter
%   epsilon:    Smoothing parameter
%   size:       length(x)
%   temp_ind:   Indices at perturbation 
%   temp_rand:  Perturbation itself
%
%   Output:
%
%   val = Acceptance probability of the algorithm.

val = (temp_rand/t)*(0.5*temp_rand + y_lm - x);
if(temp_ind(1)>1)
    val = val + abs(A(temp_ind(1)-1,temp_ind(2))-(y_lm+temp_rand)) - ...
        abs(A(temp_ind(1)-1,temp_ind(2))-y_lm);
end
if(temp_ind(1)<size)
    val = val + abs(A(temp_ind(1)+1,temp_ind(2))-(y_lm+temp_rand)) - ...
        abs(A(temp_ind(1)+1,temp_ind(2))-y_lm);
end
if(temp_ind(2)>1)
    val = val + abs(A(temp_ind(1),temp_ind(2)-1)-(y_lm+temp_rand)) - ...
        abs(A(temp_ind(1),temp_ind(2)-1)-y_lm);
end
if(temp_ind(2)<size)
    val = val + abs(A(temp_ind(1),temp_ind(2)+1)-(y_lm+temp_rand)) - ...
        abs(A(temp_ind(1),temp_ind(2)+1)-y_lm);
end
end

function val = ATV(y)
size = length(y);
val = 0;
for i=1:1:size
    for j=1:1:size
        if(i>1)
            val = val + abs(y(i-1,j)-y(i,j));
        end
        if(i<size)
            val = val + abs(y(i+1,j)-y(i,j));
        end
        if(j>1)
            val = val + abs(y(i,j-1)-y(i,j));
        end
        if(j<size)
            val = val + abs(y(i,j+1)-y(i,j));
        end
    end
end
val = 0.5*val;
end

function u = SB_ATV_mod2(g,mu)
% Split Bregman Anisotropic Total Variation Denoising
%
%   u = arg min_u 1/2||u-g||_2^2 + mu*ATV(u)
%   
%   g : noisy image
%   mu: regularisation parameter
%   u : denoised image
%
% Refs:
%  *Goldstein and Osher, The split Bregman method for L1 regularized problems
%   SIAM Journal on Imaging Sciences 2(2) 2009
%  *Micchelli et al, Proximity algorithms for image models: denoising
%   Inverse Problems 27(4) 2011
%
% Benjamin Trémoulhéac
% University College London
% b.tremoulheac@cs.ucl.ac.uk
% April 2012

g = g(:);
n = length(g);
[B Bt BtB] = DiffOper(sqrt(n));
b = zeros(2*n,1);
d = b;
u = g;
err = 1;k = 1;
tol = 1e-6;
lambda = 1;
while err > tol
    up = u;
    [u,~] = cgs(speye(n)+BtB, g-lambda*Bt*(b-d),1e-3,100); 
    Bub = B*u+b;
    d = max(abs(Bub)-mu/lambda,0).*sign(Bub);
    b = Bub-d;
    err = norm(up-u)/norm(u);
    k = k+1;
end
    function [B Bt BtB] = DiffOper(N)
    D = spdiags([-ones(N,1) ones(N,1)], [0 1], N,N+1);
    D(:,1) = [];
    D(1,1) = 0;
    B = [ kron(speye(N),D) ; kron(D,speye(N)) ];
    Bt = B';
    BtB = Bt*B;
    end
end