%%% Generate the two images for the Bayesian PM estimator/HJ PDEs paper %%%
% Written by Gabriel P. Langlois


% Images of Barbara and the cameraman from https://homepages.cae.wisc.edu/~ece533/images/
% and converted to .png format (lossless compression).

barbara_256x256 = imresize(imread('barbara_512x512.png'),[256,256]);
barbara_256x256_d = im2double(barbara_256x256);
save('barbara_256x256_d','barbara_256x256_d')

cameraman_256x256 = imread('cameraman.png');
cameraman_256x256_d = im2double(cameraman_256x256);
save('cameraman_256x256_d','cameraman_256x256_d')

clear cameraman_256x256 barbara_256x256