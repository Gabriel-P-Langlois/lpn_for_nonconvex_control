%%%%%% Gibbs sampling algorithm for sampling from TV and computing the posterior mean estimator %%%%%
% Written by Gabriel P. Langlois

% Randomization and reproducibility
rng(2,'v5normal');

% Noise
sigma = 20;

% Load image and add noise to it
lena_256x256 = imread('lena_gray_256.tif');
lena_256x256_d = im2double(lena_256x256);
lena_256x256_d_n_g = imnoise(lena_256x256_d,'gaussian',0,(sigma/255)^2);

% Reshape images to [0, 255] grayscale level
lena_256x256_d = rescale(lena_256x256_d,0,255);
lena_256x256_d_n_g = rescale(lena_256x256_d_n_g,0,255);

% Parameters of TV and probability distribution
%%% sigma=5, lambda/(2*sigma^2)=0.15 from Louchet paper.
x = lena_256x256_d_n_g;
lambda = 40;
epsilon = 2*(sigma^2)/lambda;
t = (sigma^2)/epsilon;
size = length(x);

% Compute the MAP estimate
u_map = reshape(SB_ATV_mod2(x(:),lambda*0.5),size,size);

% Choice of the images
cur_image = u_map;          % Most likely choice.

% Parameters for the algorithm (these appear enough for converence)
n = 1000;        % Maximal number of iterations
R = size^2;       % Subsampling rates
alpha = 10;     % Perturbation              % 10 --> accept ratio at 0.6
accept_r = 0;
b = 25;         % Burn-in

% MCMC Algorithm (Gibbs sampling)
u_pm = zeros(size,size);
temp_ind = zeros(R,2);
temp_rand = zeros(R,1);

for i=1:1:n
    temp_ind = randi(size,[R,2]);
    temp_rand = -alpha + (2*alpha)*rand(R,1);
   for j=1:1:R 
       P = P_ratio(cur_image,x,t,epsilon,size,[temp_ind(j,1),temp_ind(j,2)],temp_rand(j));
       if(P>rand)
           cur_image(temp_ind(j,1),temp_ind(j,2)) = cur_image(temp_ind(j,1),temp_ind(j,2))+ temp_rand(j);
           accept_r = accept_r+1;
       end
   end
   if(mod(i,b)==0)
       u_pm = u_pm + cur_image;
   end
end
accept_r = accept_r/(R*n);
u_pm = (b/n)*u_pm;

% Plot images
figure(1)
subplot(2,2,1); imshow(lena_256x256_d,[0 255]); 
title('Original image')

subplot(2,2,2); imshow(x,[0 255])
title(['Noisy image with \sigma = ',num2str(sigma),'.'])

subplot(2,2,3); imshow(u_map,[0 255])
title(['Denoised image with ROF model, with \lambda = ',num2str(lambda),'.'])

subplot(2,2,4); imshow(u_pm,[0 255])
title(['Denoised image with posterior mean estimate, with \lambda = ',num2str(lambda),'.'])

%%% Helper functions %%%
function val = P_ratio(A,x,t,epsilon,size,temp_ind,temp_rand)
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

xi_o = x(temp_ind(1),temp_ind(2));
yi_o = A(temp_ind(1),temp_ind(2));
yi_p = A(temp_ind(1),temp_ind(2)) + temp_rand;

val = (temp_rand/t)*(0.5*temp_rand + yi_o-xi_o);
if(temp_ind(1)>1)
    val = val + abs(A(temp_ind(1)-1,temp_ind(2))-yi_p) - ...
        abs(A(temp_ind(1)-1,temp_ind(2))-yi_o);
end
if(temp_ind(1)<size)
    val = val + abs(A(temp_ind(1)+1,temp_ind(2))-yi_p) - ...
        abs(A(temp_ind(1)+1,temp_ind(2))-yi_o);
end
if(temp_ind(2)>1)
    val = val + abs(A(temp_ind(1),temp_ind(2)-1)-yi_p) - ...
        abs(A(temp_ind(1),temp_ind(2)-1)-yi_o);
end
if(temp_ind(2)<size)
    val = val + abs(A(temp_ind(1),temp_ind(2)+1)-yi_p) - ...
        abs(A(temp_ind(1),temp_ind(2)+1)-yi_o);
end

val = exp(-val/epsilon);
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
tol = 1e-3;
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


