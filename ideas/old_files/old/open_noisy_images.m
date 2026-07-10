% Open noisy images

load('lena_256x256_d')
load('lena_256x256_d_n_g')
load('cameraman_256x256_d')
load('cameraman_256x256_d_n_g')

figure(1)
subplot(2,2,1)
imshow(lena_256x256_d)
title('Uncorrupted image of Lena')
subplot(2,2,2)
imshow(lena_256x256_d_n_g)
title('Noisy (Gaussian with \sigma = 0.05) image of Lena')

subplot(2,2,3)
imshow(cameraman_256x256_d)
title('Uncorrupted image of the cameraman')
subplot(2,2,4)
imshow(cameraman_256x256_d_n_g)
title('Noisy (Gaussian with \sigma = 0.05) image of the cameraman')

% Converting matrix to vector: just write vector = matrix(:);
% Converting vector to matrix: matrix = reshape(vector,256,256);