%%%%%% Simple Gibbs sampling algorithm for sampling from TV %%%%%
% Written by Gabriel P. Langlois

%%%%%%%%%%%%%%%%%%%%%%%% Notes: 
% (t=20, eps=0.25)--> MAP est is good, but u_pm is not when n = 2000. More
% samples are needed

% (t=20, eps=0.0125)--> MAP est is good, PM est not too good, but better
% than eps = 0.25 case with 2000 samples.

% (t=20, eps=0.0125,n=10000,b=50,alpha=0.55)--> Not too good
% (t=20, eps=0.0125,n=50000,b=100)-->



% n = 10000, b = 50 takes 3 minutes.

% There does not seem to be a point for doing this beyond n > 10000
%%%%%%%%%%%%%%%%%%%%%%%%%

% Randomization
rng(2,'v5normal');

% Parameters of the distribution + image
load('lena_256x256_d')
load('lena_256x256_d_n_g')
x = lena_256x256_d_n_g;
original_img = lena_256x256_d;
epsilon = 1/0.15;
t = 25/(255*epsilon);

size = length(x);

% Compute the MAP estimate of ATV
u_map = reshape(SB_ATV_mod(x(:),t),size,size);

% Choice of the images
cur_image = u_map;          % Most likely choice.

% Parameters for the algorithms
n = 200;        % Maximal number of iterations
R = size^2;       % Subsampling rates
alpha = 0.05;     % Perturbation            % Aim for accept_ratio = 0.234
b = 10;         % Burn-in

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
       end
   end
   if(mod(i,b)==0)
       u_pm = u_pm + cur_image;
   end
end
u_pm = (b/n)*u_pm;
u_pm_mod = cur_image;

% Plots
figure(1)
subplot(2,2,1)
imshow(original_img)
title('Uncorrupted image of Lena')
subplot(2,2,2)
imshow(x)
title('Noisy (Gaussian with \sigma = 0.015) images of Lena')
subplot(2,2,3)
imshow(u_map)
title(['MAP estimate (using anistropic TV code from Benjamin Tremoulheac) with t = ',num2str(t),' and \epsilon = ',num2str(epsilon)])
subplot(2,2,4)
imshow(u_pm)
title(['PM estimate with t = ',num2str(t),', \epsilon = ',num2str(epsilon),' and ',num2str(n/b),' samples.'])


%%% FUNCTIONS
function val = P_ratio(A,x,t,epsilon,size,temp_ind,temp_rand)
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