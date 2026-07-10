%% Gibbs sampling algorithm for anisotropic TV \w quadratic fidelity term 
% Written by Gabriel P. Langlois

%% Notes
% Values of lambda and factor used:
% lambda = 2, factor =  34
% lambda = 4, factor =  28
% lambda = 8, factor =  22
% lambda = 16, factor = 16
% lambda = 32, factor = 10
% lambda = 64, factor = 5
% lambda = 128, factor = 3

%% Initialization -- Free parameters
% Randomization and reproducibility
rng(2,'v5normal');

% Variables - scale, noise, and smoothing parameters
scale = 256;            % grayscale = [0,256]
m = 20000;               % m: Number of samples
sigma = 10/scale;
lambda = 32/scale;      % See table above
factor = 10;    

% Load images
image = imresize(imread('barbara_512x512.png'),[256,256]); % Barbara
image_d = im2double(image);
image_d_n_g = imnoise(image_d,'gaussian',0,(sigma)^2);

% Save noisy image
%imwrite(image_d_n_g, 'barbara_noisy_sigma=10_lambda=32.png');

%% Initialization -- Do not modify
% Parameters of TV and probability distribution
x = image_d_n_g;
epsilon = 2*(sigma^2)/lambda;
t = (sigma^2)/epsilon;

% Parameters for the algorithm
size = length(image_d);             % Size of the image (size x size)
R = size^2;                         % R: Subsampling rate
alpha = factor*sqrt(3)/size;        % Random perturbation. ~ 50/255

% MAP estimate
u_map = im2double(imresize(imread('barbara_denoised_sigma=10_lambda=32_umap_EXACT.png'),[256,256]));


% Allocation of variables and indices
u_pm_1 = zeros(size,size);
u_pm_2 = zeros(size,size);
cur_image_1 = u_map;                  % Start everytime at umap
cur_image_2 = u_map;                  % Start everytime at umap

temp_ind_1 = zeros(R,2);
temp_ind_2 = zeros(R,2);
temp_rand_1 = zeros(R,1);
temp_rand_2 = zeros(R,1);
k_ind = 1;

% Diagnostics
accept_r_1 = 0;
accept_r_2 = 0;
res_norm2_pm_m = zeros(m,1);
res_norminf_pm_m = zeros(m,1);

%% Algorithm
tic
for i=1:1:m
    temp_ind_1 = randi(size,[R,2]);                 % Pick two indices
    temp_rand_1 = -alpha + (2*alpha)*rand(R,1);     % Random perturbation
    temp_ind_2 = randi(size,[R,2]);                 
    temp_rand_2 = -alpha + (2*alpha)*rand(R,1);
        
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
        % Chain 2
        if((0<=cur_image_2(temp_ind_2(j,1),temp_ind_2(j,2))+ temp_rand_2(j)) && ... 
            (cur_image_2(temp_ind_2(j,1),temp_ind_2(j,2))+ temp_rand_2(j) <= 1))
            P = exp(-P_ratio(x(temp_ind_2(j,1),temp_ind_2(j,2)),t,temp_rand_2(j),cur_image_2(temp_ind_2(j,1),temp_ind_2(j,2)),[temp_ind_2(j,1),temp_ind_2(j,2)],size,cur_image_2)/epsilon);
            if(P>rand)
                cur_image_2(temp_ind_2(j,1),temp_ind_2(j,2)) = cur_image_2(temp_ind_2(j,1),temp_ind_2(j,2))+ temp_rand_2(j);
                accept_r_2=accept_r_2+1;
            end
        end
    end
    
    % Update (unnormalized) posterior mean estimates
    u_pm_1 = u_pm_1 + cur_image_1;
    u_pm_2 = u_pm_2 + cur_image_2;
    
    % Residual calculation (as a function of the # of samples)
    res_norm2_pm_m(i) = norm((u_pm_1+u_pm_2)/(2*i) - x,'fro');
    res_norminf_pm_m(i) = max(max(abs((u_pm_1+u_pm_2)/(2*i) - x)));
end
toc

% Compute (normalized) posterior mean estimates
u_pm_1 = (1/m)*(u_pm_1);
u_pm_2 = (1/m)*(u_pm_2);

u_pm = (u_pm_1+u_pm_2)/2;

% Compute l2 and linf norms between the pm estimate of the two chains
norm_euc = norm(u_pm_1-u_pm_2,'fro');
norm_abs = max(max(abs(u_pm_1-u_pm_2)));

% Compute the acceptance ratios (aim for ~ 0.278)
accept_r_1 = accept_r_1/(m*R);
accept_r_2 = accept_r_2/(m*R);

% Compute final residual norms
res_norm_pm = norm(u_pm-x,'fro');
res_norm_map = norm(u_map-x,'fro');
    
%% Generate figures - Full sized
figure(1)
imshow(image_d)
%title('Original image')

figure(2) 
imshow(x);
%title('Noisy image');

figure(3) 
imshow(u_map);
%title('Restored image with maximum a posterior estimate')

figure(4);
imshow(u_pm);
%title('Restored image with posterior mean estimate')

% Figures - Zoomed-in
figure(5)
imshow(image_d)
%title('Original image -- Zoom-in')
xlim([150,250])
ylim([0,110])

figure(6) 
imshow(x);
%title('Noisy image -- Zoom-in');
xlim([150,250])
ylim([0,110])

figure(7) 
imshow(u_map);
%title('Restored image with maximum a posterior estimate -- Zoom-in')
xlim([150,250])
ylim([0,110])

figure(8)
imshow(u_pm);
%title('Restored image with posterior mean estimate -- Zoom-in')
xlim([150,250])
ylim([0,110])

% Residual norms
figure(9)
plot(1:1:m,res_norm2_pm_m)
xlabel('Number of samples')
ylabel('Residual norm ||x-u_{pm}(x,t,\epsilon)||_2')

figure(10)
plot(1:1:m,res_norminf_pm_m)
xlabel('Number of samples')
ylabel('Residual norm ||x-u_{pm}(x,t,\epsilon)||_{\infty}')

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
tol = 1e-8;
lambda = 1;
while err > tol
    up = u;
    [u,~] = cgs(speye(n)+BtB, g-lambda*Bt*(b-d),1e-7,100); 
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