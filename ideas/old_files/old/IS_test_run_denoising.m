% Test script
load('lena_256x256_d')
load('lena_256x256_d_n_g')

rng(2,'v5normal');

% Parameters of the distribution
t = 1;
epsilon = 0.25;
x = lena_256x256_d_n_g;
original_img = lena_256x256_d;

clear lena_256x256_d lena_256x256_d_n_g % Rewrite this later


% Importance sampling: Sample from a Gaussian distribution of mean x
% and variance t*epsilon

J = 10; % Sample size
sum_num = zeros(length(x),length(x));
sum_den = 0;
std = sqrt(t*epsilon);

temp1 = zeros(length(x),length(x));
temp2 = 0;
for j=1:1:J
    temp1 = normrnd(x,std,[256,256]);               % Bottleneck
    temp2 = exp(-myTV(temp1)/epsilon);
    sum_num = sum_num + temp2*temp1;
    sum_den = sum_den + temp2;
end
u_pm = sum_num./sum_den;

% Show three images
figure(1)
subplot(1,3,1)
imshow(original_img)
title('Uncorrupted image of Lena')
subplot(1,3,2)
imshow(x)
title('Noisy (Gaussian with \sigma = 0.05) image of Lena')
subplot(1,3,3)
imshow(u_pm)
title('"Denoised" image of Lena')


%%% Anisotopic TV summation
function val = myTV(A)
% Input: A = n*x matrix
% Output: val = Anisotropic TV with 4-nearest neighbors with weights = 0.5
w = 0.5;
n = length(A);
val = 0;

for i=1:1:n
    for j=1:1:n
        if(i>1)
            val = val + w*abs(A(i-1,j)-A(i,j));
        end
        if(i<n)
            val = val + w*abs(A(i+1,j)-A(i,j));
        end
        if(j>1)
            val = val + w*abs(A(i,j-1)-A(i,j));
        end
        if(j<n)
            val = val + w*abs(A(i,j+1)-A(i,j));
        end
    end
end
end