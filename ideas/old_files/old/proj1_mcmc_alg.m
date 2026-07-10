%%%%%% Gibbs sampling algorithm for sampling from TV and computing the posterior mean estimator %%%%%
% Written by Gabriel P. Langlois

% Variables - scale, noise, and smoothing parameters
scale = 255;          % grayscale =[0,255];
sigma = 20/scale;
lambda = 40/scale;

% Randomization and reproducibility
rng(2,'v5normal');

% Load image, add noise to it.
image = imresize(imread('barbara_512x512.png'),[256,256]); % Barbara - reshape to [0,255] grayscale level
%image = imread('cameraman.png');        % Cameraman

image_d = im2double(image);
image_d_n_g = image_d; %imnoise(image_d,'gaussian',0,(sigma)^2);

% Parameters of TV and probability distribution
x = image_d_n_g; size = length(x);
epsilon = 2*(sigma^2)/lambda;
t = (sigma^2)/epsilon;

% Compute the MAP estimate and S0(x,t)
u_map = reshape(SB_ATV_mod2(x(:),lambda*0.5),size,size);
S_zero = (0.5/t)*norm(x-u_map)^2 + ATV(u_map);

% Choice of the images
cur_image = u_map;          % Most likely choice.

% Parameters for the algorithm
m = 1000;        % Maximal number of iterations
n = size^2;       % Subsampling rates
alpha = 10/scale;     % Perturbation              % 10 --> accept ratio at 0.6
accept_r = 0;

% MCMC Algorithm (Gibbs sampling)
u_pm = zeros(size,size);
act_terms = zeros(1,m);
temp_ind = zeros(n,2);
temp_rand = zeros(n,1);

for i=1:1:m
    temp_ind = randi(size,[n,2]);
    temp_rand = -alpha + (2*alpha)*rand(n,1);
   for j=1:1:n 
       if((0<=cur_image(temp_ind(j,1),temp_ind(j,2))+ temp_rand(j)) && ... 
               (cur_image(temp_ind(j,1),temp_ind(j,2))+ temp_rand(j) <= 1))
           P = P_ratio(cur_image,x,t,epsilon,size,[temp_ind(j,1),temp_ind(j,2)],temp_rand(j));
           if(P>rand)
               cur_image(temp_ind(j,1),temp_ind(j,2)) = cur_image(temp_ind(j,1),temp_ind(j,2))+ temp_rand(j);
           accept_r = accept_r+1;
           end
       end
   end
   act_terms(i) = (0.5/t)*norm(x-cur_image)^2 + ATV(cur_image);
   u_pm = u_pm + cur_image;
end
accept_r = accept_r/(n*m);
u_pm = (1/m)*(u_pm);      % Initial choice is a sample point too.
S_eps = 0.5*n*epsilon*log(2*pi*t*epsilon) + min(act_terms) - ... 
    epsilon*log((1/m)*sum(exp(-(act_terms - min(act_terms))/epsilon)));
S_eps_2 = S_eps - 0.5*n*epsilon*log(2*pi*t*epsilon);

% % Plot images
figure(1) 
imshow(x);
%title('Noisy image');

figure(2) 
imshow(u_map);
%title('Restored image with maximum a posterior estimate')

figure(3);
imshow(u_pm);
%title('Restored image with posterior mean estimate')

% Note: Pictures are first saved in .m format, then saved with the same
% name in .png format using the export_fig command:
% export_fig 'NAME' -transparent -png

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