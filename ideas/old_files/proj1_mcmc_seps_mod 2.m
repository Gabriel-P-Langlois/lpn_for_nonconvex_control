%%% Gibbs sampling algorithm for sampling from TV and computing S_eps for different images
% Multiple chains version
% Written by Gabriel P. Langlois

% Notes: n=50,alpha = 0:0.1:1 --> 19.2 seconds
% Notes: n=50,alpha = 0:0.025:1 --> 95 seconds

% lambda = 2, factor =  34
% lambda = 4, factor =  28
% lambda = 8, factor =  22
% lambda = 16, factor = 16
% lambda = 32, factor = 10
% lambda = 64, factor = 5
% lambda = 128, factor = 3

%% Initialization
% Variables - scale, noise, and smoothing parameters
scale = 255;            % grayscale = [0,255];
m = 1000;                % m: Number of chains
sigma = 10/scale;
lambda = 32/scale;
mult_factor = 0.25;
alpha_ind = [0:0.02:1];

% Randomization and reproducibility
rng(2,'v5normal');  % Use 2

% Load images
image1 = imresize(imread('barbara_512x512.png'),[256,256]); % Barbara
image2 = imread('cameraman.png');        % Cameraman
image_d_1 = im2double(image1);
image_d_2 = im2double(image2);

% Parameters of TV and probability distribution
size = length(image_d_1);
x = zeros(size,size);
epsilon = 2*(sigma^2)/lambda;
t = (sigma^2)/epsilon;

% Parameters for the algorithm
R = size^2;   % m = number of samples, R = subsampling rate ( = number of pixels).
alpha = mult_factor*sqrt(3)/size;       % Random perturbation. ~ 50/255

temp_ind_1 = zeros(R,2);
temp_ind_2 = zeros(R,2);

temp_rand_1 = zeros(R,1);
temp_rand_2 = zeros(R,1);

cur_image_1 = zeros(size,size);
cur_image_2 = zeros(size,size);

k_ind = 1;

% For calculating S_eps
s_zero = zeros(length(alpha_ind),1);
s_eps = zeros(length(alpha_ind),1);
act_terms = zeros(1,m);

% Diagnostic
accept_r_1 = 0;
accept_r_2 = 0;

Seps_quad = zeros(length(alpha_ind),1);

%% Algorithm
tic
for k=alpha_ind
    x = (1-k)*image_d_1 + k*image_d_2;
    u_map = reshape(SB_ATV_mod2(x(:),lambda*0.5),size,size);

    for i=1:1:m
        cur_image_1 = u_map;                  % Start everytime at umap
        temp_ind_1 = randi(size,[R,2]);               % Pick two indices
        temp_rand_1 = -alpha + (2*alpha)*rand(R,1);
        
        for j=1:1:R 
            % Chain one
            if((0<=cur_image_1(temp_ind_1(j,1),temp_ind_1(j,2))+ temp_rand_1(j)) && ... 
                    (cur_image_1(temp_ind_1(j,1),temp_ind_1(j,2))+ temp_rand_1(j) <= 1))
                 P = exp(-P_ratio(x(temp_ind_1(j,1),temp_ind_1(j,2)),t,temp_rand_1(j),cur_image_1(temp_ind_1(j,1),temp_ind_1(j,2)),[temp_ind_1(j,1),temp_ind_1(j,2)],size,cur_image_1)/epsilon);
                if(P>rand)
                    cur_image_1(temp_ind_1(j,1),temp_ind_1(j,2)) = cur_image_1(temp_ind_1(j,1),temp_ind_1(j,2))+ temp_rand_1(j);
                    accept_r_1=accept_r_1+1;
                end
            end
        end
        act_terms(i) = (0.5/t)*norm(x-cur_image_1,'fro')^2 + ATV(cur_image_1);
    end
    accept_r_1 = accept_r_1/(m*R);
    
      s_zero(k_ind) = (0.5/t)*norm(x-u_map,'fro')^2 + ATV(u_map);
      s_eps(k_ind) =  min(act_terms) +epsilon*log(m) - ... 
      epsilon*log(sum(exp(-(act_terms - min(act_terms))/epsilon))); % Actually S_eps - 0.5*R*epsilon*log(2*pi*t*epsilon)
      k_ind = k_ind + 1;
%     figure(2*k_ind-1)
%     imshow(x)
%     figure(k_ind*2)
%     imshow(u_pm)
end
toc

% Plot S_eps
figure(1)
plot(alpha_ind,s_eps, 'o')
hold
%figure(2)
plot(alpha_ind,s_zero, 'x')


%% Helper functions %%
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
    [u,~] = cgs(speye(n)+BtB, g-lambda*Bt*(b-d),1e-5,100); 
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