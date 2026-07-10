%% Script that computes S_eps(x,t).
%  Written by Gabriel P. Langlois

%% Initialization -- Free parameters
% Randomization and reproducibility
rng(2,'v5normal');

% Variables - scale, noise, and smoothing parameters
scale = 255;            % grayscale = [0,255]
m = 2;               % m: Number of samples
sigma = 10/scale;
lambda = 32/scale;      % See table above
factor = 10;    
alpha = 0.2;

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

% Diagnostic
diag1 = (0.5/t)*(norm(x-u_map,'fro')^2) + ATV(u_map);
diag2 = ATV(x);


%% Algorithm
% Sample m points from {[max(0,xi-alpha),min(xi+alpha,1]}_{i=1}^{m}. Then
% Compute average and compute variance.

eps_log_average = 0;
eps_log_variance = 0;
sample_values = zeros(1,m);
c
terms = zeros(m,1);

for i=1:1:m
   temp = max(0,u_map-alpha) + (min(1,u_map+alpha)-max(0,u_map-alpha)).*rand(size,size);
   terms(i) = (0.5/t)*(norm(x-temp,'fro')^2) + ATV(temp) + (0.5*R*epsilon)*log(2*pi*t*epsilon);
   %sample_values(1,i)= (1/sqrt(2*pi*(t*epsilon)))*exp(-0.5/(t*epsilon)*(norm(xvec-temp,'fro')^2))*exp(-(1/epsilon)*ATV(temp));
end

min_terms = min(terms);
scaled_terms = terms - min_terms;
sum_exp = sum(exp(-(1/epsilon)*scaled_terms));
eps_log_average = min_terms - epsilon*log((1/m)*sum_exp);




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