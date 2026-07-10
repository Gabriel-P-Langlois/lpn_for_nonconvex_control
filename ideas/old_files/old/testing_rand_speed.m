%% Testing the speed of rand
rng(3,'twister')

n= 256;
k = 1000;
y = zeros(n^2,1);
z = zeros(n,n);
 
 %% Compute the randn function (vector form)
for i=1:1:k
   y = randn(n^2,1); 
end

%%
for i=1:1:k
   z = randn(n,n); 
end
 

%% Log
% v5normal is the fastest.
% simdtwister
% multFibonacci