% Load the lenna image (saved in .mat format) and transform it to double
% and generate noisy gaussian and salt & pepper images.

rng(3); % Reproducibility

% Load images and convert to double on a gray scale from 0 to 255
lena_256x256 = imread('lena_gray_256.tif');
lena_256x256_d = im2double(lena_256x256);
lena_256x256_d_n_g = imnoise(lena_256x256_d,'gaussian',0,25/255);

save('lena_256x256_d','lena_256x256_d')
save('lena_256x256_d_n_g','lena_256x256_d_n_g')

% cameraman_512x512 = imread('cameraman.tif');
% cameraman_512x512_d = im2double(cameraman_512x512);
% cameraman_512x512_d_n_g = imnoise(cameraman_512x512_d,'gaussian',0,5);
% 
% save('cameraman_512x512_d','cameraman_512x512_d')
% save('cameraman_512x512_d_n_g','cameraman_512x512_d_n_g')

clear lena_256x256 lena_256x256_d lena_256x256_d_n_g lena_256x256_d_n_sp
% clear cameraman_512x512 cameraman_512x512_d cameraman_512x512_d_n_g cameraman_512x512_d_n_sp

% lena512 = imread('lena512.png');
% lena_d = im2double(lena512);

% NOTE: images are rescaled from (0,255) to (0,1)