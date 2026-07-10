% compute the exact solution (up the numerical precision) of the Rudin-Osher-Fatemi model with the 4-nearest neighbors:
%  min_y 1/2 ||x-y||_2 + \sum_{i,j} |u_j - u_i|

  

%% read image
orig = double(imread('barbara_noisy_sigma=10_lambda=32.png'));


%% call TV minimization
lambda = 16;
output_tv = TVc4(orig, lambda);

% figure
% imshow(orig/256)
% figure
% imshow(output_tv/256)

%% write result
imwrite(output_tv/256, 'barbara_denoised_sigma=10_lambda=32_umap.pgm');

  
